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
            print(f"msg_received! ts={df_line['ts']}")
            yield df_line

class DataAnalysis:
    def __init__(self):
        self.buffer = pd.DataFrame()

    def append_to_buffer(self, df_line):
        temp_df = pd.DataFrame(df_line['xyzv'], columns=['x', 'y', 'z', 'v'])
        self.buffer = pd.concat([self.buffer, temp_df], ignore_index=True)

    def exec_df_analysis(self, data):
        time.sleep(3)
        print("Analysis completed.")
        # clear buffer
        self.buffer = self.buffer.loc[self.buffer['frame'] >= (self.buffer['frame'].max()+30)]
        # Analysis Goes Here
        pass

def main():
    zmq_listener = ZmqListener()
    data_analysis = DataAnalysis()
    executor = ThreadPoolExecutor(max_workers=1)

    df_buffer = 0
    analysis_future = None

    for df_line in zmq_listener.listen():
        data_analysis.append_to_buffer(df_line)

        if analysis_future is None or analysis_future.done():
            analysis_future = executor.submit(data_analysis.exec_df_analysis, data_analysis.buffer)
            data_analysis.buffer = pd.DataFrame()

if __name__ == "__main__":
    main()
