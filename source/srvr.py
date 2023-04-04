import zmq
import orjson
import pandas as pd
import time
import threading
from concurrent.futures import ThreadPoolExecutor

class ZmqListener:
    def __init__(self, port="5555"):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://localhost:{self.port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, '')

    def listen(self):
        while True:
            message = self.socket.recv()
            df_line = orjson.loads(message)
            yield df_line

class DataAnalysis:
    def __init__(self):
        self.buffer = pd.DataFrame()
        self.df = pd.DataFrame()
        self.URL = "https://capstonebackend.herokuapp.com/api/playerData"
        self.HEADERS = { 'Content-type': 'application/json', 'Accept': 'text/plain' }

    def append_to_buffer(self, df_line):
        temp_df = pd.DataFrame(df_line['xyzv'], columns=['x', 'y', 'z', 'v'])
        self.buffer = pd.concat([self.buffer, temp_df], ignore_index=True)

    def send_to_db(self, bball: pd.Series):
        """ Send http POST to database / server """
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
            print(err_msg, flush=True)


    def exec_df_analysis(self, buffer_data):
        # insert data analysis here
        pass
        self.df = pd.concat([self.df, buffer_data])
        self.df = self.df.loc[self.df['frame'] >= (self.df['frame'].max()+30)]


    def do_hdbscan(self, df_in: pd.DataFrame, MIN_CLSTRS: int, NORM: bool = False) -> pd.DataFrame:
        from warnings import filterwarnings; filterwarnings('ignore', category=RuntimeWarning);

        df_out = pd.DataFrame()
        df_in = df_in.copy(deep=True)
        df_in['v'] = df_in['v'].abs()

        # add normalization (done by column per-stack)
        if NORM:
            df_in[['stack','x','y','z','v']] = df_in[['stack','x','y','z','v']].groupby('stack').apply(lambda x: (x - x.min()) / (x.max() - x.min())).fillna(0).drop('stack',axis=1).reset_index('stack')

        for stack,g in df_in.groupby('stack'):
            scan = hdbscan.HDBSCAN(min_cluster_size=MIN_CLSTRS, algorithm='best', alpha=1.0, metric='euclidean')
            g = g.copy(deep=True)
            #print(f"Length of g[{stack}] = {len(g)}")
            model = scan.fit(g[['x','y','z','v']])
            g = g.assign(outlier=model.outlier_scores_, clstr=model.labels_)
            df_out = pd.concat([df_out, g])

        df_out['outlier'] = df_out['outlier'].fillna(0)
        return df_out
    
    def get_ball(self, df, MPH, OUT_SCORE, MIN_CLSTR):
        df = self.do_hdbscan(df, MIN_CLSTR)
        bballs = df.loc[(df['v'].abs() >= MPH) & (df['outlier'] >= OUT_SCORE)]
        #bballs = bballs.loc[bballs.loc[bballs['v'].abs() >= MPH][['frame','stack']].drop_duplicates()['frame'].drop_duplicates().index]
        bballs = bballs.loc[bballs.loc[bballs['v'].abs() >= MPH][['frame','stack']].drop_duplicates()['frame'].drop_duplicates().index]
        
        if len(bballs) < 1:
            return
        else:
            bballs

def main():
    zmq_listener = ZmqListener()
    data_analysis = DataAnalysis()
    executor = ThreadPoolExecutor(max_workers=1)

    analysis_future = None

    for df_line in zmq_listener.listen():
        data_analysis.append_to_buffer(df_line)

        if analysis_future is None or analysis_future.done():
            analysis_future = executor.submit(data_analysis.exec_df_analysis, data_analysis.buffer)
            data_analysis.buffer = pd.DataFrame()

if __name__ == "__main__":
    main()


import os
import json
import requests
import pandas as pd
from hdbscan import HDBSCAN
from warnings import filterwarnings
from sys import stderr as sys_stderr
from typing import Tuple

exit()

