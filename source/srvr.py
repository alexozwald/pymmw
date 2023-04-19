#!/usr/bin/env python3
import os.path
import threading

from collections import deque
import warnings
from concurrent.futures import ThreadPoolExecutor
from traceback import extract_tb, format_exception_only

import click
import hdbscan
import matplotlib.pyplot as plt
import orjson
import pandas as pd
import requests
import zmq
from seaborn import diverging_palette, set_theme  #cubehelix_palette

warn = warnings.warn
def tprint(*args, **kwargs):
    """ Wrapper for printing with threads """
    print(*args, flush=True, **kwargs)

class Plot3Dv:
    """
    Refs for improvement:
    [1] [Blitting Class sans-animation (Article)](https://coderslegacy.com/matplotlib-blitting-tutorial/)
    [2] [Matplotlibs official adv. example blitting class](https://matplotlib.org/stable/tutorials/advanced/blitting.html)
    """

    def __init__(self, frame_lag: int = 20):
        self.bounds = dict(x=[-2,25], y=[-2,25], z=[-5,15])
        init_v_bounds = dict(vmin=-40, vmax=40)
        view_angle_3d = dict(elev=14, azim=-135, roll=0)

        self.df = pd.DataFrame()
        self.frame_lag = frame_lag

        set_theme(style='darkgrid', context='notebook')
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.view_init(**view_angle_3d)
        self.ax.set(xlim=self.bounds['x'], ylim=self.bounds['y'], zlim=self.bounds['z'], xlabel='X', ylabel='Y', zlabel='Z')

        # pre-generate colorbar
        #self.cmap = cubehelix_palette(as_cmap=True)
        self.cmap = diverging_palette(h_neg=150, h_pos=276, s=73, l=30, sep=1, center='light', as_cmap=True)
        self.sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=plt.Normalize(**init_v_bounds))
        self.sm.set_array([])
        self.cbar = self.fig.colorbar(self.sm, ax=self.ax, pad=0.1)
        self.cbar.ax.set_title("Velocity (m/s)")

        # Launch Interactive Mode & GUI (non-blocking ig)
        plt.ion()
        plt.show(block=False)

    def check_max_bounds(self, df_new: pd.DataFrame):
        """ Update axis bounds from defaults if needed """
        bounds_updated = False

        for axis in ['x', 'y', 'z']:
            min_val, max_val = df_new[axis].agg(['min', 'max'])
            old_min, old_max = self.bounds[axis]

            if (new_bounds := (min(min_val, old_min), max(max_val, old_max))) != (old_min, old_max):
                self.bounds[axis] = new_bounds
                bounds_updated = True

        return bounds_updated

    def add_pts(self, df_new: pd.DataFrame):
        """ Add points from current df-reading to 3D Scatter """
        # clear axis of data
        self.ax.cla()
        #self.ax.lines.clear()

        # update & prune points
        fr_cutoff = df_new['frame'].max() - self.frame_lag
        self.df = pd.concat([self.df, df_new], ignore_index=True)
        self.df = self.df.loc[self.df['frame'] >= fr_cutoff]

        # re-normalize opacity
        min_opacity, max_opacity = 0.3, 1.0
        min_frame, max_frame = self.df['frame'].min(), self.df['frame'].max()
        normalized_opacity = self.df['frame'].sub(min_frame).div(max_frame - min_frame).mul(max_opacity - min_opacity).add(min_opacity).fillna(0.3)

        # compute contents of new new scatter plot
        x = self.df['x'].values
        y = self.df['y'].values
        z = self.df['z'].values
        v = self.df['v'].values
        self.ax.scatter(x, y, z, c=v, cmap='Purples', s=20, marker='o', edgecolor='k', linewidths=0.6, alpha=normalized_opacity)

        # Set limits, axes, and title+subtitle
        self.ax.set(xlim=self.bounds['x'], ylim=self.bounds['y'], zlim=self.bounds['z'], xlabel='X', ylabel='Y', zlabel='Z')
        time_str = pd.to_datetime(df_new['ts'].max(), unit='ms').strftime('%X.%f')[:-3]
        self.ax.set_title(f"Velocity Plot\nTime: {time_str} (fr: {df_new['frame'].max()})")

        # Write changes onto gui
        plt.draw()
        plt.pause(0.001)
        #self.fig.canvas.flush_events()

class DebugThreadPoolExecutor(ThreadPoolExecutor):
    """ Wrapper for ThreadPoolExecutor that delivers Warnings & Exceptions from Threads """

    def __init__(self, *args, non_blocking: bool, verbose: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.non_blocking = non_blocking
        self.verbose = verbose
        self._exceptions_queue = deque()
        self._warnings_queue = deque()

    def submit(self, fn, *args, **kwargs):
        def wrapped_fn(*args, **kwargs):
            with warnings.catch_warnings(record=True) as captured_warnings:
                warnings.simplefilter('always')
                try:
                    result = fn(*args, **kwargs)
                except Exception as e:
                    self._exceptions_queue.append(e)
                else:
                    for w in captured_warnings:
                        if 'hdbscan' not in w.category.__module__:
                            self._warnings_queue.append(w)
                    return result

        future = super().submit(wrapped_fn, *args, **kwargs)
        self.raise_exception()
        return future

    def raise_exception(self):
        while self._exceptions_queue:
            e = self._exceptions_queue.popleft()
            tb = extract_tb(e.__traceback__)[-1]
            exc_info = ''.join(format_exception_only(type(e), e)).strip()
            if self.non_blocking:
                # Print the exception without ending the program
                tprint(f"Exception occurred: '{exc_info}' => See: {os.path.basename(tb.filename)}:{tb.lineno}")
            else:
                # Raise the exception and end the program
                raise e
        if self.verbose > 0:
            while self._warnings_queue:
                w = self._warnings_queue.popleft()
                tprint(f"{pd.Timestamp.now().strftime('%X')} | WARN: {w.category.__name__}, '{w.message}'")

class ZmqListener:
    """ Listener class for ZMQ Port """

    def __init__(self, verbose: int, port: int = "5555"):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://localhost:{self.port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, '')

        self.verbose = verbose
        self.buffer = pd.DataFrame()

    def listen(self) -> pd.DataFrame:
        while True:
            message = self.socket.recv()
            msg_dict = orjson.loads(message)
            df_line = pd.DataFrame(msg_dict['xyzv'], columns=['x', 'y', 'z', 'v']).assign(ts=msg_dict['ts'], frame=msg_dict['frame'])
            # currently disabling pyarrow backend bc of weird error
            # <EOL>.convert_dtypes(dtype_backend='pyarrow', inplace=True)
            yield df_line

    def append_to_buffer(self, df_line: pd.DataFrame):
        self.buffer = pd.concat([self.buffer, df_line], ignore_index=True)

class DataAnalysis:
    """ Class containing all BBall Analysis Funcs & Methods """

    def __init__(self,
                 internet_input: bool,
                 mps_input: float,
                 outlier_score_input: float,
                 normalize_input: bool,
                 alpha_score_input: float,
                 minimum_cluster_size_input: int,
                 fr_stack_size_input: int,
                 min_fr_b2b_input: int,
                 verbose: int):
        self.df = pd.DataFrame()
        self.time = pd.Timestamp.now().strftime('%X')
        self.URL = "https://capstonebackend.herokuapp.com/api/playerData"
        self.HEADERS = { 'Content-type': 'application/json', 'Accept': 'text/plain' }
        self.sent_bbals = pd.Series(name='ts')
        self.tmp_detections = pd.DataFrame()

        # detection algo parameters
        self.STACK_SIZE = fr_stack_size_input # 20 (tested)
        self.MIN_FR_B2B = min_fr_b2b_input # 7 (tested); formerly: BBALL_FRAME_SCATTER_LEN
        self.MPS = mps_input # 17 m/s (tested)
        self.OUT_SCORE = outlier_score_input # 0.75 (tested)
        self.MIN_CLSTR = minimum_cluster_size_input # 3 (tested)
        self.ALPHA = alpha_score_input # 1.0 (tested)
        self.NORM = normalize_input # false (tested)

        self._thread_lock = threading.Lock()
        self.verbose = verbose
        self.internet = internet_input


    def handle_detections(self, detects: pd.DataFrame, curr_df: pd.DataFrame):
        """ Process & Narrow BBall Detections to one MSG per-ball """

        # return if no potential detections
        if len(detects) == 0:
            return

        # detect potential temporal clumps from one hit -- wait until MIN_FR_B2B cleared
        fr_cutoff = curr_df['frame'].min() - self.MIN_FR_B2B
        in_betweens_or_current = detects.loc[ detects['frame'] >= fr_cutoff ]
        if len(in_betweens_or_current) > 0:
            return
        
        # split detects past frame cutoff
        detects = detects.loc[ detects['frame'] < fr_cutoff ]

        # calc distance for al potential bballs
        detects['dist'] = (detects['x'].pow(2) + detects['y'].pow(2) + detects['z'].pow(2)).pow(0.5)

        # pick ball@max_v using closest point, fallback to max velocity (fallback unlikely)
        bball_v = detects.loc[detects['dist'] == detects['dist'].min()]
        bball_v = bball_v.loc[bball_v['v'] == bball_v['v'].max()].iloc[0] # iloc[0] => convert to series

        # send point to server & log (always)
        bball_v_ts = pd.Timestamp(bball_v['ts'], unit='ms')
        if bball_v['ts'] not in self.sent_bbals.values:
            bball_v['mph'] = bball_v['v'] / 2.23693629
            tprint(f"=({self.time})==> BBall Found! v={bball_v['v']:.3f} m/s ({bball_v['mph']:.1f} mph) @ d={bball_v['dist']:<2.2f}m & ර={bball_v['outlier']:.3f}० @ ({bball_v['x']:.1f} {bball_v['y']:.1f} {bball_v['z']:.1f})m @ {bball_v_ts.strftime('%X')} & □ ={bball_v['frame'].astype(int):<3} {len(detects):>2}=>1 pts")

            self.sent_bbals = pd.concat([self.sent_bbals, pd.Series(bball_v['ts'])], ignore_index=True)
            if self.internet:
                self.send_to_db(bball_v)

        # remove eval'd data from tmp_detections (thread-safe)
        with self._thread_lock:
            self.tmp_detections = self.tmp_detections.loc[ ~ self.tmp_detections['ts'].isin(detects['ts']) ]

        if (live_detect_lag := (pd.Timestamp.now() - bball_v_ts).seconds) > self.MIN_FR_B2B*2:
            warn(f"High load. -{live_detect_lag}s on live bball detections rn.", ResourceWarning)


    def send_to_db(self, bball: pd.Series):
        """ Send http POST to database / server """
        # Prepare the data for sending
        data = {'ts': bball['ts'], 'xyzv': bball.loc[['x', 'y', 'z', 'v']].tolist()}
        data = orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY).decode('utf-8')

        try:
            # Post to the server (header, json data, URL) & Track what frames are sent
            requests.post(url=self.URL, headers=self.HEADERS, data=data)
            self.log(f"BBall sent! @{pd.Timestamp(bball['ts'], unit='ms').strftime('%X')} v={bball['v']}")
        except Exception as e:
            self.log(f"FAILED TO POST. BALL_DATA={data}")
            raise e


    def log(self, s: str):
        tprint(f"{self.time} | {s}")


    def _get_df_minmax(self, dfA: pd.DataFrame, dfB: pd.DataFrame|str = None, col: str = 'frame'):
        """ Logging-aid to safely extract values for min/max in sticky situations """
        if dfB is not None:
            try:
                df_min, df_max = dfA[col].min(), dfB[col].max()
            except Exception:
                df_min, df_max = (0,0)
        else:
            try:
                df_min, df_max = dfA[col].min(), dfA[col].max()
            except Exception:
                df_min, df_max = (0,0)
        return df_min, df_max


    def exec_df_get_bball(self, buffer_data: pd.DataFrame):
        """ End-to-end func for BBall Detection, Data Processing, Model, and DB Communication """

        if self.verbose > 0: # verbose logging pre-reqs for later in func
            (buffer_max, df_og_max) = self._get_df_minmax(buffer_data, self.df, col='frame')
        self.time = pd.Timestamp.now().strftime('%X')

        # Scrub any negative velocities from buffer (+ prevent empty-df errors)
        # Early Exit: thread safety / return copies of relevant dataframes
        buffer_data = buffer_data.loc[buffer_data['v'] >= 0]
        if len(buffer_data) == 0:
            with self._thread_lock:
                return (self.tmp_detections.copy(deep=True), self.df.copy(deep=True))

        # import pre-processed buffer & prune stack.
        self.df = pd.concat([self.df, buffer_data])
        self.df = self.df.loc[self.df['frame'] >= (buffer_data['frame'].max() - self.STACK_SIZE)]
        if not self.internet:
            self.df = self.df.loc[self.df['frame'] <= (buffer_data['frame'].max())]

        if self.verbose > 1:
            df_min, df_max = self._get_df_minmax(self.df, col='frame')
            buffer_max, curr_fr_len, len_curr_stack, frame_jump, skipped_frames = (
                buffer_max, (df_max - df_min), len(self.df), (buffer_max - df_og_max),
                (0 if pd.isna(skf_diff := df_min - df_og_max) else max(0, skf_diff)) )
            self.log(f"□ ={buffer_max:<5} - □ Range={curr_fr_len:<2} - □ skipped={skipped_frames:<1} - □ jumped={frame_jump:<1} - #pts={len_curr_stack}")

        df = self.do_hdbscan(self.df, self.MIN_CLSTR, self.ALPHA, norm=self.NORM)

        # filter bballs
        df = df.loc[(df['v'] >= self.MPS) & (df['outlier'] >= self.OUT_SCORE)].drop_duplicates()

        if len(df) > 0:
            df = df.loc[(df['v'] == df['v'].max())]
            bball = df.iloc[0:]
            with self._thread_lock:
                self.tmp_detections = pd.concat([self.tmp_detections, bball], ignore_index=True).drop_duplicates()

        ## DEBUG & Stres Test: Add Delay
        #from time import sleep; from random import random; sleep(random() % 1.6);

        # thread safety / return copies of relevant dataframes
        with self._thread_lock:
            return (self.tmp_detections.copy(deep=True), self.df.copy(deep=True))
        

    def do_hdbscan(self, df_in: pd.DataFrame, min_clstrs: int, alpha: float, norm: bool = False) -> pd.DataFrame:
        """ Executes HDBSCAN algorithm returning `outlier` (& optionally `clstr`) column(s). """

        # disable hdbscan warning -- library bug
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        # copy-before-write (on slice) safety
        df = df_in.copy(deep=True)
        #df['v'] = df['v'].abs() # we filtered out all -v in pre-processing

        # add normalization (done by column, formerly per-stack)
        if norm:
            df[['x','y','z','v']] = df[['x','y','z','v']].apply(lambda x: (x - x.min()) / (x.max() - x.min())).fillna(0)

        if len(df) >= 4:
            scan = hdbscan.HDBSCAN(min_cluster_size=min_clstrs, algorithm='best', alpha=alpha, metric='euclidean', allow_single_cluster=True)
            model = scan.fit(df[['x','y','z','v']])
            df = df.assign(outlier=model.outlier_scores_, clstr=model.labels_)
            df['outlier'].fillna(0, inplace=True)
        else:
            df = df.assign(outlier=0, clstr=0)
            if self.verbose > 0:
                warn(f"hdbscan skipped, len too short -- len(df)={len(df_in)}, v_max={df_in['v'].max():<0.3f}", BytesWarning)

        return df


    def backtrace_initial_vel(self, df: pd.DataFrame, centroid: tuple) -> int:
        """ Expansion to get initial velocity from mid-air location -- **requires detection of batter location** """
        pass

@click.command(context_settings=dict(help_option_names=['-h','--help'], max_content_width=90))
@click.option("-m", "--mps", type=float, default=17, show_default=True, help="Set the minimum cutoff speed (m/s) at which outliers are tested to be baseballs")
@click.option("-o", "--outlier-score", type=click.FloatRange(0.01, 0.99), default=0.75, show_default=True, help="Set the cutoff minimum outllier score to test for baseballs with")
@click.option("-N", "--normalize", is_flag=True, default=False, show_default=True, help="Flag to normalize the dimensional readings before running HDBSCAN algo!")
@click.option("-a", "--alpha", type=click.FloatRange(0.0, 2.0), default=1.00, show_default=True, help="Set the cutoff minimum outllier score to test for baseballs with")
@click.option("-n", "--min-cluster-size", type=int, default=3, show_default=True, help="Set frame lag for the Live 3D plot")
@click.option("-fSTACK", "--frame-stack-size", type=int, default=20, show_default=True, help="Set the frame stack height of temporal points that will be concat'd")
@click.option("-fB2B", "--min-frames-ball2ball", type=int, default=7, show_default=True, help="Set the minimum # of frames between baseballs (to separate clusters)")
@click.option("-p", "--plot/--no-plot", is_flag=True, default=False, show_default=True, help="▲▽ Enable Live 3D Plot △▼")
@click.option("-fLAG", "--frame-lag", type=int, default=20, show_default=True, help="Set frame lag for the Live 3D plot")
@click.option("-i", "--internet/--no-internet", is_flag=True, default=False, show_default=True, help="Enable sending actual http.POSTs to the server")#default=True
@click.option("-v", "--verbose", count=True, default=False, help="Increase Output Verbosity")
def main(mps: float, outlier_score: float, normalize: bool, alpha: float, min_cluster_size: int, frame_stack_size: int, min_frames_ball2ball: int, plot: bool, frame_lag: int, internet: bool, verbose: int, **kwargs):
    """Launch ZMQ-Reader service to parse, analyze, & process . Plot data live in a colored, 3D Scatter plot.
        
    \b
    Requisite Launch Commands: 
    MacOS:      ./pymmw.py -c /dev/tty.SLAB_USBtoUART -d /dev/tty.SLAB_USBtoUART3
    Linux:      ./pymmw.py -c /dev/ttyUSB0 -d /dev/ttyUSB1
    TEST_SRVR:  python REPO_BASE/bball_data3b/zmq_server.py

    \b
    References:
    [1] HDBSCAN Parameter Selection - https://hdbscan.readthedocs.io/en/latest/parameter_selection.html
    [2] TI mmWave Demo - https://dev.ti.com/gallery/view/mmwave/mmWave_Demo_Visualizer/ver/3.6.0/
    [3] TI mmWave Sensing Estimator - https://dev.ti.com/gallery/view/mmwave/mmWaveSensingEstimator/ver/2.2.0/

    \b
    Tested Config:
    ./srvr.py  --no-internet --mps 17 --outlier-score 0.75 --alpha 1.0 --min-cluster-size 3 -fB2B 7 -fSTACK 20 --plot -fLAG 20 [-vv]
    
    """

    # generate
    zmq_listener = ZmqListener(verbose=verbose)
    data_analysis = DataAnalysis(
        internet_input=internet,
        mps_input=mps,
        outlier_score_input=outlier_score,
        normalize_input=normalize,
        alpha_score_input=alpha,
        minimum_cluster_size_input=min_cluster_size,
        fr_stack_size_input=frame_stack_size,
        min_fr_b2b_input=min_frames_ball2ball,
        verbose=verbose)
    analysis_executor = DebugThreadPoolExecutor(max_workers=1, non_blocking=False, verbose=verbose)
    handle_executor = DebugThreadPoolExecutor(max_workers=1, non_blocking=True, verbose=verbose)
    handle_future, analysis_future = None, None

    if plot:
        plotter = Plot3Dv(frame_lag=frame_lag)

    if verbose > 0:
        tprint(f"=({data_analysis.time})==> Entering loop...")

    for df_line in zmq_listener.listen():
        zmq_listener.append_to_buffer(df_line)

        if plot:
            plotter.add_pts(df_line)

        # If analysis_future has a response, submit handle_detections task
        if analysis_future and analysis_future.done():
            input_data_for_handle = analysis_future.result()
            handle_executor.submit(data_analysis.handle_detections, *input_data_for_handle)

        # Submit exec_df_get_bball task when there's no ongoing analysis
        if analysis_future is None or analysis_future.done():
            analysis_future = analysis_executor.submit(data_analysis.exec_df_get_bball, zmq_listener.buffer)
            zmq_listener.buffer = pd.DataFrame()


if __name__ == "__main__":
    main()
