import numpy as np
import mayavi_utils
from rapprentice.colorize import *
from mayavi import mlab
from multiprocessing import Process,Pipe
import cPickle
import time



def gen_custom_request(func_name, **kwargs):
    """
    Returns a plotting request to custom functions.
    func_name should be in : {'points', 'lines'}
    """
    req = {'type':'custom', 'func':func_name, 'data':kwargs}
    return cPickle.dumps(req)


def gen_mlab_request(func, *args, **kwargs):
    """"
    Pickles a call to a function of mlab and returns the pickled
    object. This object can be sent to the Mayavi process using pipes.   
    """
    req = {'type':'mlab', 'func':gen_mlab_request.func_to_str[func], 'data':(args, kwargs)}
    return  cPickle.dumps(req)
gen_mlab_request.func_to_str = {v:k for k,v in mlab.__dict__.iteritems() if not k.startswith('_')}


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
        self.plotting_funcs = {'lines': mayavi_utils.plot_lines}

    def check_and_process(self):
        if self.request_pipe.poll():
            plot_request = self.request_pipe.recv()
            self.process_request(cPickle.loads(plot_request))

    def process_request(self, req):
        if req['type'] == 'custom':
            f = self.plotting_funcs[req['func']]
            kwargs = req['data']
            f(**kwargs)
        else:
            f = getattr(mlab, req['func'])
            args, kwargs = req['data']
            f(*args, **kwargs)


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

    def request(self, plot_request):
        self.pipe_to_mayavi.send(plot_request)


if __name__=='__main__':

    # example usage below:
    p = PlotterInit()
    parity = False

    while True:
        req = None
        color =  tuple(np.random.rand(1,3).tolist()[0])

        if parity:
            # example of a custom request to the plotter
            N = 3
            line_points = [np.random.randn(4,3) for i in xrange(N)]
            req  = gen_custom_request('lines', lines=line_points, color=color, line_width=1, opacity=1)
        else:
            # example of how to request a mlab function to the plotter
            data  = np.random.randn(5,3)
            req   =  gen_mlab_request(mlab.points3d, data[:,0], data[:,1], data[:,2], color=color)

        parity = not parity
        p.request(req)
        time.sleep(1)   
