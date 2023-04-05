import zmq
import orjson
import pandas as pd
import hdbscan
#import time
import threading
from numpy import sqrt
from concurrent.futures import ThreadPoolExecutor
#from warnings import filterwarnings
#from sys import stderr as sys_stderr
import requests
from collections import deque
import argparse
import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl



class Plot3Dv:
    def __init__(self, FRAME_LAG: int = 20):
        self.bounds = {'x': [-5, 25],
                       'y': [-5, 25],
                       'z': [-5, 15]}
        init_v_bounds = {'vmin': -40, 'vmax': 40}
        view_angle_3d = dict(elev=12, azim=-135, roll=0)
        
        self.df = pd.DataFrame()
        self.FRAME_LAG = FRAME_LAG

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.view_init(**view_angle_3d)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_xlim(*self.bounds['x'])
        self.ax.set_ylim(*self.bounds['y'])
        self.ax.set_zlim(*self.bounds['z'])

        # pre-generate colorbar
        self.sm = plt.cm.ScalarMappable(cmap='Purples', norm=plt.Normalize(**init_v_bounds))
        #self.sm.set_over('red')    # set past-limits colors
        #self.sm.set_under('black') # set past-limits colors
        self.sm.set_array([])
        self.cbar = self.fig.colorbar(self.sm, ax=self.ax, pad=0.1)
        self.cbar.ax.set_title(f"Velocity (m/s)")

        # launch gui
        #plt.show()
        # Launch gui
        plt.ion()  # Turn on interactive mode
        plt.show(block=False)  # Show plot non-blocking


    def check_max_bounds(self, df_new: pd.DataFrame):
        diff = False
        df_new_x_min = df_new['x'].min()
        df_new_x_max = df_new['x'].max()
        df_new_y_min = df_new['y'].min()
        df_new_y_max = df_new['y'].max()
        df_new_z_min = df_new['z'].min()
        df_new_z_max = df_new['z'].max()
        self.bounds['x'][0], diff = (df_new_x_min, True) if (df_new_x_min < self.bounds['x'][0]) else (self.bounds['x'][0], False)
        self.bounds['x'][1], diff = (df_new_x_max, True) if (df_new_x_max > self.bounds['x'][1]) else (self.bounds['x'][1], False)
        self.bounds['y'][0], diff = (df_new_y_min, True) if (df_new_y_min < self.bounds['y'][0]) else (self.bounds['y'][0], False)
        self.bounds['y'][1], diff = (df_new_y_max, True) if (df_new_y_max > self.bounds['y'][1]) else (self.bounds['y'][1], False)
        self.bounds['z'][0], diff = (df_new_z_min, True) if (df_new_z_min < self.bounds['z'][0]) else (self.bounds['z'][0], False)
        self.bounds['z'][1], diff = (df_new_z_max, True) if (df_new_z_max > self.bounds['z'][1]) else (self.bounds['z'][1], False)
        if diff:
            return True
        else:
            return False

    def add_pts(self, df_new: pd.DataFrame):
        # <insert code here to add points to the 3D scatter plot using df_line>
        #self.ax.scatter(...)

        #self.ax.clear()
        #self.ax.remove()
        #self.ax.clear()
        self.ax.cla()

        # update & prune points
        fr_cutoff = df_new['frame'].max() - self.FRAME_LAG
        self.df = pd.concat([self.df, df_new], ignore_index=True)
        self.df = self.df.loc[self.df['frame'] >= fr_cutoff]

        # re-normalize opacity
        min_opacity, max_opacity = 0.3, 1.0
        min_frame, max_frame = self.df['frame'].min(), self.df['frame'].max()
        normalized_opacity = self.df['frame'].sub(min_frame).div(max_frame - min_frame).mul(max_opacity - min_opacity).add(min_opacity).fillna(0.3)

        if self.check_max_bounds(df_new=df_new):
            self.ax.set_xlim(*self.bounds['x'])
            self.ax.set_ylim(*self.bounds['y'])
            self.ax.set_zlim(*self.bounds['z'])

        x = self.df['x'].values
        y = self.df['y'].values
        z = self.df['z'].values
        v = self.df['v'].values

        sc = self.ax.scatter(x, y, z, c=v, cmap='Purples', s=20, marker='o', edgecolor='k', linewidths=0.6, alpha=normalized_opacity)
        #ax.set_xlabel('X')
        #ax.set_ylabel('Y')
        #ax.set_zlabel('Z')

        # Set title and subtitle
        time_str = pd.to_datetime(df_new['ts'].max(), unit='ms').strftime('%X.%f')[:-3]
        self.ax.set_title(f"Velocity Plot\nTime: {time_str} (fr: {df_new['frame'].max()})")

        plt.draw()
        plt.pause(0.01)


class DebugThreadPoolExecutor(ThreadPoolExecutor):
    def __init__(self, *args, non_blocking: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self._exception = None
        self.non_blocking = non_blocking

    def submit(self, fn, *args, **kwargs):
        def wrapped_fn(*args, **kwargs):
            with warnings.catch_warnings(record=True) as captured_warnings:
                warnings.simplefilter("always")
                try:
                    result = fn(*args, **kwargs)
                except Exception as e:
                    self._exception = e
                else:
                    # Log captured warnings
                    for w in captured_warnings:
                        print(f"*** Warning: {w.message} ***", )
                    return result
        return super().submit(wrapped_fn, *args, **kwargs)

    def raise_exception(self):
        if self._exception:
            if self.non_blocking:
                # Print the exception without ending the program
                print(f"Exception occurred: {self._exception}")
            else:
                # Raise the exception and end the program
                raise self._exception


class ZmqListener:
    def __init__(self, port="5555"):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://localhost:{self.port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, '')
        
        self.buffer = pd.DataFrame()

    def listen(self) -> pd.DataFrame:
        while True:
            message = self.socket.recv()
            msg_dict = orjson.loads(message)
            df_line = pd.DataFrame(msg_dict['xyzv'], columns=['x', 'y', 'z', 'v']).assign(ts=msg_dict['ts'], frame=msg_dict['frame'])
            # currently disabling pyarrow backend bc of weird error
            #df_line.convert_dtypes(dtype_backend='pyarrow', inplace=True)
            yield df_line #msg_dict

    def append_to_buffer(self, df_line: pd.DataFrame):
        self.buffer = pd.concat([self.buffer, df_line], ignore_index=True)


class DataAnalysis:
    def __init__(self):
        self.df = pd.DataFrame()
        self.time = pd.Timestamp.now()
        self.URL = "https://capstonebackend.herokuapp.com/api/playerData"
        self.HEADERS = { 'Content-type': 'application/json', 'Accept': 'text/plain' }
        self.sent_bbals = pd.Series(dtype='Int64', name='ts')
        self.tmp_detections = pd.DataFrame()
        
        # detection algo parameters
        self.STACK_SIZE = 20
        self.BBALL_FRAME_SCATTER_LEN = 5  # lowkey this is rlly MIN_FR_BETWEEN_BBALLS
        self.MPH = 17
        self.OUT_SCORE = 0.75
        self.MIN_CLSTR = 3
        self.ALPHA = 1.0

        self._thread_lock = threading.Lock()


    def handle_detections(self, detects: pd.DataFrame, curr_df: pd.DataFrame):
        """ DESCRIPTION """

        # return if no potential detections
        if len(detects) == 0:
            return
        
        fr_cutoff = curr_df['frame'].min() - self.BBALL_FRAME_SCATTER_LEN
        min_fr_curr_df = curr_df['frame'].min()

        # detect potential temporal clumps from one hit -- wait until BBALL_FRAME_SCATTER_LEN cleared
        in_betweens = detects.loc[ (detects['frame'] <= fr_cutoff) & (detects['frame'] < min_fr_curr_df) ]
        if len(in_betweens) > 0:
            return

        # split detects past frame cutoff
        detects = detects.loc[ detects['frame'] < fr_cutoff ]

        # calc distance for al potential bballs
        detects['dist'] = sqrt(detects['x'].pow(2) + detects['y'].pow(2) + detects['z'].pow(2))

        # pick ball@max_v using closest point, fallback to max velocity (fallback unlikely)
        bball_v = detects.loc[detects['dist'] == detects['dist'].max()]
        bball_v = bball_v.loc[bball_v['v'] == bball_v['v'].max()].iloc[0] # iloc[0] => convert to series

        # send point to server
        #self.send_to_db(bball_v)
        print(f"=> Sending bball to server!  v = {bball_v['v']:.3f} m/s  @  ({bball_v['x']:.2f}m,{bball_v['y']:.2f}m,{bball_v['z']:.2f}m)")

        # remove eval'd data from tmp_detections (thread-safe)
        with self._thread_lock:
            self.tmp_detections = self.tmp_detections.loc[ ~ self.tmp_detections['ts'].isin(detects['ts']) ]

        return


    def send_to_db(self, bball: pd.Series):
        """ Send http POST to database / server """
        try:
            # post to server (header, json data, URL)
            data = { 'ts': bball['ts'], 'xyzv': bball[['x','y','z','v']].tolist() }
            data = orjson.dumps(data).decode('utf-8')
            requests.post(url=self.URL, headers=self.HEADERS, data=data)
            # track what frames are sent
            self.sent_bbals = pd.concat([self.sent_bbals, pd.Series(bball['ts'])], ignore_index=True)
            print(f"{self.time} | BBall sent. v={bball['v']}")
        except Exception as e:
            err_msg = f"{self.time} | FAILED TO POST -- err={e}({e.args})"
            if len(bball) > 0:      err_msg += f" -- bball={bball.to_dict()}"
            else:                   err_msg += f" -- bball=BBALL_NOT_FOUND"
            print(err_msg, flush=True)


    def exec_df_get_bball(self, buffer_data: pd.DataFrame):
        """ End-to-end func for BBall Detection, Data Processing, Model, and DB Communication """

        #print(f"=> STARTING `exec_df_get_bball` !!!", flush=True)
        #print(buffer_data.to_markdown(), flush=True)

        # logging pre-reqs
        self.time = pd.Timestamp.now().strftime('%X')
        buffer_min, buffer_max = (buffer_data['frame'].min(), buffer_data['frame'].max())
        try:
            df_og_min, df_og_max = (self.df['frame'].min(), self.df['frame'].max())
        except:
            df_og_min, df_og_max = (0,0)

        # scrub any negative velocities from buffer.  import buffer.  prune stack.
        buffer_data = buffer_data.loc[buffer_data['v'] >= 0]
        self.df = pd.concat([self.df, buffer_data])
        self.df = self.df.loc[self.df['frame'] >= (buffer_data['frame'].max() - self.STACK_SIZE)]
        
        #print(f"{self.time} | len(self.df) = {len(self.df)}", flush=True)

        # log frame-delay and relative processing time of `exec_df_get_bball`
        df_max, df_min = (self.df['frame'].min(), self.df['frame'].max())
        skipped_frames = -(df_min - df_og_max)
        try:
            skipped_frames = 0 if skipped_frames < 0 else skipped_frames
        except Exception as e:
            print(f"==> SKIPPED_FRAMES ERROR -- Current DF...\n{self.df.to_markdown()}")
            raise e
        frame_jump = buffer_max - df_og_max
        curr_fr_len = df_max - df_min
        len_curr_stack = len(self.df)
        print(f"{self.time} | □={buffer_max:<6} - □ Range={curr_fr_len} - □ skipped={skipped_frames:>2} - □ jumped={frame_jump:>2} - #pts={len_curr_stack}", flush=True)
        #print(self.df.to_markdown(), flush=True)

        #def get_ball(self, df, MPH, OUT_SCORE, MIN_CLSTR):
        #get_ball(df, 17, 0.75, 3)

        df = self.do_hdbscan(self.df, self.MIN_CLSTR, self.ALPHA, norm=False)

        # filter bballs
        df = df.loc[(df['v'] >= self.MPH) & (df['outlier'] >= self.OUT_SCORE)]
        #bballs = bballs.loc[bballs.loc[bballs['v'].abs() >= MPH][['frame','stack']].drop_duplicates()['frame'].drop_duplicates().index]
        #bballs = bballs.loc[bballs.loc[bballs['v'].abs() >= MPH][['frame','stack']].drop_duplicates()['frame'].drop_duplicates().index]

        if len(df) > 0:
            # MAYBE TAKE OUT THIS LINE -- logging for it instead.
            df = df.loc[(df['v'] == df['v'].max())]
            bball = df.iloc[0:1]
            if len(df) > 1:
                pass #print(f"BBall DF has multiple rows.  Using row of max v (). DF:\n{df}", flush=True)
            else:
                ...
            with self._thread_lock:
                self.tmp_detections = pd.concat([self.tmp_detections, bball], ignore_index=True).drop_duplicates()
                #print(f"NO 'INDEX' ?? cols={self.tmp_detections.columns.values}", flush=True)
                #.drop('index', axis=1).drop_duplicates()
            #print(f"=> FOUND BBALL! v={bball['v'].iloc[0]}")

        # thread safety / return copies of relevant dataframes
        with self._thread_lock:
            return_detects_and_df = (self.tmp_detections.copy(deep=True), self.df.copy(deep=True))
        return return_detects_and_df

    def do_hdbscan(self, df_in: pd.DataFrame, min_clstrs: int, alpha: float, norm: bool = False) -> pd.DataFrame:
        """ Executes HDBSCAN algorithm returning `outlier` (& optionally `clstr`) column(s). """
        
        # disable warning -- bug in hdbscan
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        df_in = df_in.copy(deep=True)
        df_out = pd.DataFrame()
        
        #df_in['v'] = df_in['v'].abs()

        # add normalization (done by column per-stack)
        if norm:
            df_in[['x','y','z','v']] = df_in[['x','y','z','v']].apply(lambda x: (x - x.min()) / (x.max() - x.min())).fillna(0)
        
        if len(df_in) >= 4:
            scan = hdbscan.HDBSCAN(min_cluster_size=min_clstrs, algorithm='best', alpha=alpha, metric='euclidean')
            model = scan.fit(df_in[['x','y','z','v']])
            df_in = df_in.assign(outlier=model.outlier_scores_, clstr=model.labels_)
        else:
            #print(f"{self.time} | WARN: hdbscan skipped, len too short -- len(df)={len(df_in)}, v_max={df_in['v'].max()}", flush=True)
            df_in = df_in.assign(outlier=0, clstr=0)

        df_out = pd.concat([df_out, df_in])

        df_out['outlier'].fillna(0, inplace=True)
        return df_out

    def backtrace_initial_vel(self, df: pd.DataFrame) -> int:
        pass


def main(mk_plot: bool):

    zmq_listener = ZmqListener()
    data_analysis = DataAnalysis()
    #executor = ThreadPoolExecutor(max_workers=1)
    # executor = DebugThreadPoolExecutor(max_workers=1)
    analysis_executor = DebugThreadPoolExecutor(max_workers=1, non_blocking=False)
    handle_executor = DebugThreadPoolExecutor(max_workers=1, non_blocking=True)

    if mk_plot:
        # TIL Matplotlib is not thread safe...
        # https://stackoverflow.com/questions/34764535/why-cant-matplotlib-plot-in-a-different-thread
        plot_executor = DebugThreadPoolExecutor(max_workers=1, non_blocking=False)
        plotter = Plot3Dv()

    analysis_future = None
    handle_future = None
    
    print("=> Entering loop...")

    for df_line in zmq_listener.listen():
        zmq_listener.append_to_buffer(df_line)
        #print(f"=> df_line = '{zmq_listener.buffer.to_dict('records')}'")
        #print(f"{pd.Timestamp.now().strftime('%X')} | frame = {zmq_listener.buffer['frame'].max()}")

        ## Submit exec_df_get_bball task when there's no ongoing analysis
        #if analysis_future is None or analysis_future.done():
        #    analysis_future = executor.submit(data_analysis.exec_df_get_bball, zmq_listener.buffer)
        #    zmq_listener.buffer = pd.DataFrame()

        # Raise any exceptions in worker threads
        analysis_executor.raise_exception()
        handle_executor.raise_exception()

        if mk_plot:
            plot_executor.raise_exception()
            plot_executor.submit(plotter.add_pts, df_line)

        # If analysis_future has a response, submit handle_detections task
        if analysis_future and analysis_future.done():
            input_data_for_handle = analysis_future.result()
            #handle_future = handle_executor.submit(data_analysis.handle_detections, *input_data_for_handle)

        # Submit exec_df_get_bball task when there's no ongoing analysis
        if analysis_future is None or analysis_future.done():
            analysis_future = analysis_executor.submit(data_analysis.exec_df_get_bball, zmq_listener.buffer)
            zmq_listener.buffer = pd.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Processing Module for IWR6843ISK Baseball Detection")
    parser.add_argument("--plot", action="store_true", help="Enable Live 3D plot")
    args = parser.parse_args()
    
    if args.plot:
        main(mk_plot=True)
    else:
        main(mk_plot=False)