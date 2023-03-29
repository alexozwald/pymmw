#!/bin/python3

import json
import matplotlib.pyplot as mpl

fname = "pymmw_2023-03-21_12-40-44.log.fixed"

file = open(fname , "rb")
json = json.load(file)

x_data = []
y_data = []

for frame in json:
    idx = frame["idx"]
    pcloud_json = frame["xyzv"]

    for point in pcloud_json:
        x = point[0]
        y = point[1]
        z = point[2]
        v = point[3]

        # print("idx: ", idx, "xyzv", x, ",", y, ",", z, ",", v)

        x_data.append(idx)
        y_data.append(v)

mpl.scatter(x_data, y_data)
mpl.show()

file.close()
