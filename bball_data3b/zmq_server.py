import zmq
import orjson
import time

class zmq_listener():
    def __init__(self, MSG_PER_SECOND=20) -> None:
    
        # ZeroMQ setup
        self.zmq_context = zmq.Context()
        self.zmq_publisher = self.zmq_context.socket(zmq.PUB)
        self.zmq_port = "5555"  # Choose an appropriate port number
        self.zmq_publisher.bind(f"tcp://*:{self.zmq_port}")
        self.msg_per_second = MSG_PER_SECOND
    
    def run(self, FILE_LIST: list[str]):
        while True:
            for file_path in FILE_LIST:
                with open(file_path, "rb") as f:
                    for line in f.readlines():
                        #log_str = orjson.loads(line)
                        #print(f"type(log_str)={type(log_str)}  &&  log_str={log_str}")
                        # Send log_str via ZeroMQ
                        self.zmq_publisher.send(line)
                        time.sleep(1 / self.msg_per_second)

if __name__ == "__main__":
    FILE_LIST = ["./data/2023.03.21_jc/log1.json", "./data/2023.03.21_jc/log2.json", "./data/2023.03.21_jc/log3.json", "./data/2023.03.21_jc/log4.json"]
    zmq_srvr = zmq_listener(MSG_PER_SECOND=10)
    zmq_srvr.run(FILE_LIST)
    
    exit()

'''
LOCAL_URL = "tcp://*:15667"
MSG_PER_SECOND = 20

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind(LOCAL_URL)

file_list = ["./data_2023.03.21/log1.json", "./data_2023.03.21/log2.json", "./data_2023.03.21/log3.json", "./data_2023.03.21/log4.json"]

while True:
    for file_path in file_list:
        with open(file_path, "rb") as f:
            for line in f.readlines():
                message = json.loads(line)
                socket.send_json(message)
                time.sleep(1 / MSG_PER_SECOND)
'''
