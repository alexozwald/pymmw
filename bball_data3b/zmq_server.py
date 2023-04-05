import zmq
import orjson
from time import sleep
from pandas import Timestamp, Timedelta
from helper import repo_dir

class zmq_listener():
    def __init__(self, MSG_PER_SECOND=20) -> None:
    
        # ZeroMQ setup
        self.zmq_context = zmq.Context()
        self.zmq_publisher = self.zmq_context.socket(zmq.PUB)
        self.zmq_port = "5555"  # Choose an appropriate port number
        self.zmq_publisher.bind(f"tcp://*:{self.zmq_port}")
        self.msg_per_second = MSG_PER_SECOND
    
    def run(self, FILE_LIST: list[str]):
        # open infinite loop of regurgitating log2 file at port readable for testing
        i, start = (0, Timestamp.now())
        while True:
            for file_path in FILE_LIST:
                
                # log
                td = (Timestamp.now() - start).floor('ms').__str__()[7:-3]
                print(f"{td} - i={i:<6} - Opening '@GH/{file_path.replace(repo_dir,'')}'...")
                
                with open(file_path, "rb") as f:
                    for line in f.readlines()[:1500]:
                        self.zmq_publisher.send(line)
                        sleep(1 / self.msg_per_second)
                        i += 1

if __name__ == "__main__":
    #FILE_LIST = ["./data/2023.03.21_jc/log1.json", "./data/2023.03.21_jc/log2.json", "./data/2023.03.21_jc/log3.json", "./data/2023.03.21_jc/log4.json"]
    FILE_LIST = [repo_dir + "/bball_data3b/data/2023.03.21_jc/log2.json"]
    zmq_srvr = zmq_listener(MSG_PER_SECOND=30)
    zmq_srvr.run(FILE_LIST)
    
    exit()
