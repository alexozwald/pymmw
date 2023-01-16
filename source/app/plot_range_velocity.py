#
# Copyright (c) 2018, Manfred Constapel
# This file is licensed under the terms of the MIT license.
#

#
# CFAR detected objects - 3D plot
#

import math
import os
import sys

try:
       
    import numpy as np
    
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
    __base__ = os.path.dirname(os.path.abspath(__file__))
    while 'lib' not in [d for d in os.listdir(__base__) if os.path.isdir(os.path.join(__base__, d))]: __base__ = os.path.join(__base__, '..')
    if __base__ not in sys.path: sys.path.append(__base__)

    from lib.plot import * 

except ImportError as e:
    print(e, file=sys.stderr, flush=True)
    sys.exit(3)

# ------------------------------------------------

def update(data, history=40):
    if 'detected_points' not in data:
        return

    xm, ym = ax.get_xlim(), ax.get_ylim()

    for _, p in data['detected_points'].items():
        x, y, z, d = p['x'], p['y'], p['z'], p['v']
        radial_dist = math.sqrt(x**2 + y**2 + z**2)

        if xm[0] <= radial_dist <= xm[1] and ym[0] <= d <= ym[1]:
            if len(series) > history:
                series[0].remove()
                series.pop(0)

            path = ax.scatter(radial_dist, d, c='red', alpha=0.5)
            series.append(path)


if __name__ == "__main__":

    if len(sys.argv[1:]) != 1:
        print('Usage: {} {}'.format(sys.argv[0].split(os.sep)[-1], '<range_maximum>'))
        sys.exit(1)
        
    try:

        range_max = float(sys.argv[1])

        # ---

        series = []

        fig = plt.figure(figsize=(6, 6))
        ax = plt.subplot(1, 1, 1)  # rows, cols, idx
        
        fig.canvas.manager.set_window_title('...')
        
        ax.set_title('Doppler-Range'.format(), fontsize=10)
        ax.set_xlabel('Longitudinal distance [m]')
        ax.set_ylabel('Radial velocity [m/s]')
        
        move_figure(fig, (0 + 45*4, 0 + 45*4))
        
        ax.set_xlim([0, range_max])

        ax.set_ylim([-50, 50])
        ax.set_yticks(range(-50, 50 + 1, 10))

        fig.tight_layout(pad=2)
        
        ax.scatter([], [])
        ax.grid(color='black', linestyle=':', linewidth=0.5)

        start_plot(fig, ax, update, 25)
    
    except Exception as e:
        print(e, file=sys.stderr, flush=True)
        sys.exit(2)
