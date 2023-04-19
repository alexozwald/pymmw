import zmq
import orjson
from time import sleep
from pandas import Timestamp
from helper import REPO_DIR
from os.path import basename, dirname

class zmq_listener():
    def __init__(self, MSG_PER_SECOND=20) -> None:

        # ZeroMQ setup
        self.zmq_context = zmq.Context()
        self.zmq_publisher = self.zmq_context.socket(zmq.PUB)
        self.zmq_port = "5555"  # Choose an appropriate port number
        self.zmq_publisher.bind(f"tcp://*:{self.zmq_port}")
        self.msg_per_second = MSG_PER_SECOND

    def time(self):
        return (Timestamp.now() - self.start).floor('ms').__str__()[7:-3]

    def run(self, FILE_LIST: list[str], inf: bool = True):
        # open infinite loop of regurgitating log2 file at port readable for testing
        i, start = (0, Timestamp.now())
        self.start = Timestamp.now()
        while inf:
            for file_path in FILE_LIST:

                # log
                td = (Timestamp.now() - start).floor('ms').__str__()[7:-3]
                print(f"{td} - i={i:<6} - Opening '@GH/{file_path.replace(REPO_DIR,'')}'...")

                with open(file_path, "rb") as f:
                    for line in f.readlines()[:1500]:
                        self.zmq_publisher.send(line)
                        sleep(1 / self.msg_per_second)
                        i += 1


        # Not infinite Loop Data Run
        F_LEN = len(FILE_LIST)
        for f_num,file_path in enumerate(FILE_LIST):

            # log info

            f_path = file_path.replace(REPO_DIR,'')
            f_name, dir_group = basename(file_path), dirname(file_path).split('/')[-1][11:]

            print(f"[{self.time()}] '{f_name}' ({f_num}/{F_LEN}) :: i={i:<6} - Opening '@GH/{f_path}'...")

            with open(file_path, "rb") as f:
                start = i
                for line in f.readlines():
                    line = orjson.dumps(orjson.loads(line)|{'file':f_name,'dir':dir_group})
                    self.zmq_publisher.send(line)
                    sleep(1 / self.msg_per_second)
                    i += 1
                    if i % 100 == 0:
                        print(f"[{self.time()}] '{f_name}' :: idx={i:<6}  line={i-start :<5}")
        # after it's done...
        self.zmq_publisher.send(b'END_RUN')


if __name__ == "__main__":
    ALL_FILES = [f'{REPO_DIR}/bball_data3b/data/2023.03.21_jc/log1.json',f'{REPO_DIR}/bball_data3b/data/2023.03.21_jc/log2.json',f'{REPO_DIR}/bball_data3b/data/2023.03.21_jc/log3.json',f'{REPO_DIR}/bball_data3b/data/2023.03.21_jc/log4.json',f'{REPO_DIR}/bball_data3b/data/2023.03.30_lab_lobs/2023-03-30_09-56-36 -- OLD.json.log', f'{REPO_DIR}/bball_data3b/data/2023.03.30_lab_lobs/2023-03-30_09-38-00.json.log', f'{REPO_DIR}/bball_data3b/data/2023.03.30_lab_lobs/2023-03-30_09-40-23.json.log', f'{REPO_DIR}/bball_data3b/data/2023.03.30_lab_lobs/2023-03-30_09-46-36.json.log', f'{REPO_DIR}/bball_data3b/data/2023.03.30_lab_lobs/2023-03-30_09-50-42.json.log', f'{REPO_DIR}/bball_data3b/data/2023.03.30_lab_lobs/2023-03-30_10-33-06.json.log', f'{REPO_DIR}/bball_data3b/data/2023.03.31_pitch/2023-03-31_12-38-09.json.log', f'{REPO_DIR}/bball_data3b/data/2023.03.31_pitch/2023-03-31_12-39-26.json.log', f'{REPO_DIR}/bball_data3b/data/2023.03.31_pitch/2023-03-31_12-50-52.json.log', f'{REPO_DIR}/bball_data3b/data/2023.03.31_pitch/2023-03-31_13-04-42.json.log']
    FILE_LIST = [REPO_DIR + "/bball_data3b/data/2023.03.21_jc/log2.json"]

    zmq_srvr = zmq_listener(MSG_PER_SECOND=35)
    #zmq_srvr.run(FILE_LIST)
    zmq_srvr.run(ALL_FILES, inf=False)

    exit()
