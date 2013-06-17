"""
Script to poll a function from inside the mayavi's event loop
"""


from rapprentice.colorize import *

import numpy as np
import time

from mayavi import mlab
from pyface.timer.api import Timer
from mayavi.scripts import mayavi2


class Pollster(object):
    def plot_points(self):
        data = np.random.randn(10,3)
        color = (1,0,0)
        scale_factor=1
        assert data.shape[1]==3, colorize("Plot data incorrect dimension", "red", True)
        mlab.points3d(data[:,0], data[:,1], data[:,2], figure=mlab.gcf(), color=color, scale_factor=scale_factor)

    def poll(self):
        self.plot_points()
        print colorize("polling..", "red", True)


@mlab.show
def main():
    mlab.figure()
    time.sleep(1)

    # setup polling
    p = Pollster()
    timer = Timer(100, p.poll)

    # Keep a reference on the timer
    mayavi2.savedtimerbug = timer

    # To stop polling the file do:
    #timer.Stop()

if __name__ == '__main__':
    main()
