import os
from pandas import Timestamp, DataFrame
import orjson
from sys import stderr
import zmq as zmq
from traceback import extract_tb, format_exception_only

class Logger:

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.time = Timestamp.now() # Timestamp later replaced with Int (unix milliseconds)
        self.frame = 0 # Track frames

        # use relative path everything in disgrace
        fileName = f"{self.time.strftime('%Y-%m-%d_%H-%M-%S')}.json.log"
        log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'log')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_file_path = os.path.join(log_dir, fileName)
        # generate log file & ~line-buffer~
        self.log_out = open(self.log_file_path, 'w+b') # buffering=1 does nothing on bytes-obj 

        try:

            # ZeroMQ setup 
            self.zmq_context = zmq.Context()
            self.zmq_publisher = self.zmq_context.socket(zmq.PUB)
            self.zmq_port = "5555"
            self.zmq_publisher.bind(f"tcp://*:{self.zmq_port}")

        except Exception as e:

            tb = extract_tb(e.__traceback__)[-1]
            exc_info = ''.join(format_exception_only(type(e), e)).strip()
            print(f"Exception occurred: '{exc_info}' => See: {os.path.basename(tb.filename)}:{tb.lineno}", flush=True, file=stderr)
            raise e


    def message(self, dataFrame: dict):
        self.time = int(Timestamp.now().to_datetime64().astype(int) / 10**6)

        # check if there are points to record / send => SKIP OR SWIM
        if not dataFrame['detected_points']:
            self.frame += 1
            return
        else:
            df = DataFrame.from_dict(dataFrame['detected_points'], orient='index').reset_index(drop=True) #.assign(frame=self.frame, ts=self.time)

        # format / parse 'dataFrame' object into transmissible bytecode
        log_dict = { 'ts': self.time, 'frame': self.frame, 'xyzv': df[['x','y','z','v']].to_numpy().tolist() }
        log_str = orjson.dumps(log_dict) + b'\n'
        self.log_out.write(log_str)

        # Send log_str via ZeroMQ
        self.zmq_publisher.send(log_str)

        # log (NOTE: self.verbose is hard coded to False b/c upstream)
        if self.verbose:# or self.frame % 60 == 0:
            print(log_str.decode('utf-8'), file=stderr, flush=True)

        # iterate frame index tracker
        self.frame += 1
