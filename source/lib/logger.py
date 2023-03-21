import os
import json
import requests
import pandas as pd
from hdbscan import HDBSCAN
from warnings import filterwarnings
from sys import stderr as sys_stderr
from typing import Tuple


class Logger:

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.time = pd.Timestamp.now() # Timestamp replaced with Int (unix milliseconds)

        # use relative path everything in disgrace
        fileName = f"{self.time.strftime('%Y-%m-%d_%H-%M-%S')}.json.log"
        log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'log')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_file_path = os.path.join(log_dir, fileName)
        # generate log file & line-buffer
        self.log_out = open(self.log_file_path, 'a', buffering=1)

        self.frame = 0
        self.df = pd.DataFrame(columns=['ts','frame','x','y','z','v'])
        self.sent_bbals = pd.Series() # int (unix milliseconds)

        self.STACK_LEN = 20            # Num of frames to buffer / compute on
        self.TEST_SERVER_CONN = False  # True to debug & skip computes
        self.URL = "https://capstonebackend.herokuapp.com/api/playerData"
        self.HEADERS = { 'Content-type': 'application/json', 'Accept': 'text/plain' }
        self.HDBSCAN_PARAMS = {
            # REF: https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.hdbscan_.HDBSCAN
            'algorithm': 'best',
            'min_cluster_size': 3,
            'metric': 'euclidean',
            'alpha': 1.0,
            'cluster_selection_method': 'eom', # TODO: test 'leaf'
        }


    def message(self, dataFrame: dict):
        """ Writes the desired data into the logfile.  Called by mss/x8_mmw.py """
        # update current time
        self.time = int(pd.Timestamp.now().to_datetime64().astype(int) / 10**6)
        # rescale buffer / drop old frames
        self.df = self.df.loc[self.df['frame'] >= (self.frame - self.STACK_LEN)]

        # parse dataFrame & write log data
        log_str = self.parse_data_header(dataFrame)
        self.log_out.write(log_str + '\n')

        if self.TEST_SERVER_CONN:
            self.test_connection();  self.frame += 1;  return;

        # search for bball -> post if bball found
        SEND_BALL, bball = self.search_for_bball()
        if SEND_BALL:
            self.post_ball(bball)

        # Iterate frame
        self.frame += 1


    def parse_data_header(self, dataFrame: dict) -> str:
        """ Parse dataFrame Header.  Update rolling data stack.  Output logfile str """
        # update dataframe w/ new rows
        df = pd.DataFrame.from_dict(dataFrame['detected_points'], orient='index').reset_index(drop=True).assign(frame=self.frame, ts=self.time)
        self.df = pd.concat([self.df, df], ignore_index=True)
        # return log string
        log_dict = { 'ts': self.time, 'xyzv': df[['x','y','z','v']].to_numpy().tolist() }
        return json.dumps(log_dict)


    def search_for_bball(self) -> Tuple[bool, pd.Series]:
        """
        Search buffer for baseball via HDBSCAN.
        Return bool if ball is found & DF row of hit ball
        POTENTIAL FASTER ALTERNATIVE:
        1)  Run model in background (asyncio), saved to self.model
        2)  Run self.model.predict(NEW_XYZV_ARRAY)
        """
        # default response
        SEND_BBALL, bball_hit = False, pd.Series(index=['ts','frame','x','y','z','v','outlier','clstr'])

        # run hdbscan on dataframe
        df_clstrd = self.exec_hdbscan(self.df)

        # params for if point is a bball
        # 1)  outlier score >= 90%
        # 2)  not clustered (-1 = outlier on tree)
        # 3)  velocity >= 5 m/s OR 11.18 mph
        df_parsed = df_clstrd.loc[ (df_clstrd['outlier'] >= 0.90) & (df_clstrd['clstr'] == -1) & (df_clstrd['v'] >= 5) ]

        # parse to see if ball should be sent to server...
        if len(df_parsed) >= 1:
            # pick out highest match for simplicity
            # TODO: OR WE CAN PICK THE LAST MATCH TO RULE OUT THE PITCH & ADD A DELAY THAT A MATCH IN THE LAST 5 FRAMES IS VOID
            bball_hit = df_parsed.loc[df_parsed['v'] == df_parsed['v'].max()].iloc[0]
            SEND_BBALL = True if (bball_hit['ts'] not in self.sent_bbals) else False

        return SEND_BBALL, bball_hit

        # TODO:  IF WE ADD INITIAL VELOCITY INTERPOLATION
        # if SEND_BBALL:
        #     self.interpolate_initial_velocity(df_clstrd, bball_hit)
        # def interpolate_initial_velocity(self, df: pd.DataFrame, bball: pd.Series) -> list: ...
        # 1)  Find cluster of batter, dist from batter-cluster centroid to bball
        # 2)  Calc drag & initial velocity
        #   return bball[['x','y','z']].tolist() + [v_interpolated]


    def exec_hdbscan(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Execute HDBSCAN on copy of buffer DF -- return buffer w/ attrs """
        # disable warning -- bug in hdbscan
        filterwarnings('ignore', category=RuntimeWarning)

        # deep copy to preserve self.df
        df = df.copy(deep=True)
        # absolute value velocities
        df['v'] = df['v'].abs()
        # define params
        SCAN_COLS = ['x','y','z','v']
        min_df_length = self.HDBSCAN_PARAMS.get('min_cluster_size') + 1

        if len(df) >= min_df_length:
            scan = HDBSCAN(**self.HDBSCAN_PARAMS)
            model = scan.fit(df[SCAN_COLS])
            df = df.assign(outlier=model.outlier_scores_, clstr=model.labels_)
        else:
            print(f"\n{self.time} -- WARN: hdbscan skipped, len too short -- len(df)={len(df)}, v_max={df['v'].max()}", file=sys_stderr, flush=True)
            df = df.assign(outlier=0, clstr=0)

        return df


    def post_ball(self, bball: pd.Series) -> None:
        """ Send http POST to server """
        try:

            # post to server (header, json data, URL)
            data = { 'ts': bball['ts'], 'xyzv': bball[['x','y','z','v']].tolist() }
            requests.post(url=self.URL, headers=self.HEADERS, data=json.dumps(data))
            # track what frames are sent
            self.sent_bbals = pd.concat([self.sent_bbals, pd.Series(bball['ts'])], ignore_index=True)

        except Exception as e:

            err_msg = f"\n{self.time} -- FAILED TO POST BBALL -- err={e}({e.args})"
            if len(bball) > 0:
                err_msg += f" -- bball={bball.to_dict()}"
            else:
                err_msg += f" -- bball=BBALL_NOT_FOUND"
            print(err_msg, file=sys_stderr, flush=True)


    def test_connection(self):
        """ Test Http Connection To Server -- Refreshed From Integration Day 1 """
        # use fake data or max vel from current frame
        df_now = self.df.copy(deep=True).loc[self.df['frame'] == self.frame]
        df_now = df_now.loc[df_now['v'] == df_now['v'].max()]
        if len(df_now) >= 1:
            data = { 'ts': self.time, 'xyzv': df_now[['x','y','z','v']].iloc[0].tolist() }
        else:
            data = { 'ts': self.time, 'xyzv': [666, 666, 666, 666] }

        # try to post fake data -- every 2 seconds
        if self.frame % 60 == 0:
            try:
                requests.post(url=self.URL, data=json.dumps(data), headers=self.HEADERS)
                self.sent_bbals = pd.concat([self.sent_bbals, pd.Series(self.time)], ignore_index=True)
            except Exception as e:
                err_msg = f"\n{self.time} -- FAILED TO POST BBALL -- err={e}({e.args}) -- data={data}"
                print(err_msg, file=sys_stderr, flush=True)
