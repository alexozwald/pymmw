import datetime
import time
import os
#import numpy as np
import json

class Logger:

    def __init__(self, verbose=False):
        # use relative path everything in disgrace
        fileName = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) +'.json.log'
        log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'log')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_file_path = os.path.join(log_dir, fileName)

        self.verbose = verbose
        self.idx = 0

        # generate log file & write first line of json
        self.log_out = open(self.log_file_path, 'a', buffering=1) # line buffered file write
        self.log_out.write("[\n")

    def __del__(self):
        """ destructor to properly close json
        caveat:     albeit with extra ',' on last array element
        issues:     interruption doesnt call __del__ dunder.
        """
        self.log_out.write(']')


    def message(self, dataFrame):
        """ writes the desired data into the logfile """
        #formatedData = "{\"ts\": " + str(int(time.time()*1000)) +", \"dataFrame\": " + str(dataFrame) + "}"
        data, arr = {}, []

        data['ts_ms'] = int(time.time()*1000)
        data['idx'] = self.idx
        for pt in dataFrame['detected_points'].values():
            arr.append([pt['x'], pt['y'], pt['z'], pt['v']]) #arr = np.append(arr, np.array([pt['x'], pt['y'], pt['z'], pt['v']]) )
        data['xyzv'] = arr

        data = json.dumps(data)
        data_fmt = str(data + ',' + '\n')

        print(data_fmt) if self.verbose else None
        self.log_out.write(data_fmt)
        self.idx += 1