#!/usr/bin/env python3
import os.path
import threading

from collections import deque
import warnings
from traceback import extract_tb, format_exception_only

import click
import hdbscan
import matplotlib.pyplot as plt
import orjson
import pandas as pd
from seaborn import diverging_palette, set_theme

#import requests
# instead, move to bball class. make it 

class HDBSCAN_Analysis():
    """ Class containing all BBall Analysis Funcs & Methods - using HDBSCAN """

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
        # V2: ADD PROCESSED DF AS POINT 
        with self._thread_lock:
            return (self.tmp_detections.copy(deep=True), self.df.copy(deep=True), df.copy(deep=True))
        

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

class PYTORCH_GNN_Analaysis():
    """ Class containing all BBall Analysis Funcs & Methods - using PyTorch & GNN Radar Point Cloud Analysis """

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

    def exec(self)
