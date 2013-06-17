"""
Script to poll a function from inside the mayavi's event loop
"""


from rapprentice.colorize import *

import numpy as np
import time

from mayavi import mlab
from pyface.timer.api import Timer
from mayavi.scripts import mayavi2




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
