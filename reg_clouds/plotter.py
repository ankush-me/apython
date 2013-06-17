import numpy as np
from mayavi import mlab
import threading 
import time
from rapprentice.colorize import *

from multiprocessing import Process,Pipe
from mayavi.plugins.app import main

import pickle


class PlotRequest():
    """
    Creates a plot request for mayavi.
    Sort of like a ros-message.
    
    request_type : a string. Can be in {'points', 'lines'}
    """
    def __init__(self, data, request_type='points', color=(1,0,0), scale_factor=1):
        assert isinstance(data, np.ndarray), colorize("Plot request data-type invalid", "red", True)
        assert data.shape[1]==3, colorize("Plot data incorrect dimension", "red", True)

        self.request = {}
        self.request['type']  = request_type
        self.request['color'] = color
        self.request['scale'] = scale_factor
        self.request['data']  = data 


class Plotter():
    """
    This class's check_and_process function is polled
    by the timer.
   
    A mayavi figure is created in a separate process and
    a timer is started in that same process.
   
    The timer periodically poll the check_and_process function
    which checks if any plotting request was sent to the process.
    If a request is found, the request is handled.
    """
    def __init__(self, in_pipe):
        self.request_pipe  = in_pipe
        self.plotting_funcs = {'points': self.plot_points, 'lines':self.plot_lines}

    def check_and_process(self):
        if self.request_pipe.poll():
            plot_request = self.request_pipe.recv()
            self.process_request(pickle.loads(plot_request))

    def process_request(self, req):
        self.plotting_funcs[req['type']](req)

    def plot_points(self, plot_request):
        d  = plot_request['data']
        mlab.points3d(d[:,0], d[:,1], d[:,2], color=plot_request['color'], scale_factor=plot_request['scale'])

    def plot_lines(self, req):
        pass


@mlab.show
def create_mayavi(pipe):
    mlab.figure()
    time.sleep(1)

    mayavi_app = Plotter(pipe)
    
    from pyface.timer.api import Timer
    from mayavi.scripts import mayavi2

    timer = Timer(50, mayavi_app.check_and_process) 
    mayavi2.savedtimerbug = timer

    

class PlotterInit(object):
    """
    Initializes Mayavi in a new process.
    """

    def __init__(self):
        self.mayavi_process  = None
        (self.pipe_to_mayavi, self.pipe_from_mayavi) = Pipe()
  
        self.mayavi_process = Process(target=create_mayavi, args=(self.pipe_from_mayavi,))
        self.mayavi_process.start()


    def terminate(self):
        try:
            self.mayavi_process.terminate()
        except:
            pass

    def request(self, plot_request):
        assert isinstance(plot_request, PlotRequest), colorize("Plot request-type invalid.", "red", True)
        self.pipe_to_mayavi.send(pickle.dumps(plot_request.request))


def gen_request():
    data = np.random.randn(5,3)
    color =  tuple(np.random.rand(1,3).tolist()[0])
    return PlotRequest(data, request_type='points', color=color)
    

if __name__=='__main__':
    # example usage below:
    p = PlotterInit()
    while True:
        p.request(gen_request())
        time.sleep(1)   
