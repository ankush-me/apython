from rapprentice import registration
from rapprentice.colorize import *
from easyInput.slider import easyInput
import os.path as osp
import numpy as np
import time
from mayavi import mlab


def gen_grid(f, mins, maxes, xres = .01, yres = .01, zres = .01):
    """
    generate 3d grid and warps it using the function f.
    """    
    xmin, ymin, zmin = mins
    xmax, ymax, zmax = maxes

    nfine = 30
    ncoarse = 10

    xcoarse = np.linspace(xmin, xmax, ncoarse)
    ycoarse = np.linspace(ymin, ymax, ncoarse)
    zcoarse = np.linspace(zmin, zmax, ncoarse)
      
    xfine = np.linspace(xmin, xmax, nfine)
    yfine = np.linspace(ymin, ymax, nfine)
    zfine = np.linspace(zmin, zmax, nfine)
    
    lines = []
    if len(zcoarse) > 1:    
        for x in xcoarse:
            for y in ycoarse:
                xyz = np.zeros((nfine, 3))
                xyz[:,0] = x
                xyz[:,1] = y
                xyz[:,2] = zfine
                lines.append(f(xyz))

    for y in ycoarse:
        for z in zcoarse:
            xyz = np.zeros((nfine, 3))
            xyz[:,0] = xfine
            xyz[:,1] = y
            xyz[:,2] = z
            lines.append(f(xyz))
        
    for z in zcoarse:
        for x in xcoarse:
            xyz = np.zeros((nfine, 3))
            xyz[:,0] = x
            xyz[:,1] = yfine
            xyz[:,2] = z
            lines.append(f(xyz))

    return lines


def plot_lines(lines):
    """
    input  : a list of m matrices of shape nx3
             each list is interpreted as one line
    output : plot each line in mayavi
    
    adapted from : http://docs.enthought.com/mayavi/mayavi/auto/example_plotting_many_lines.html
    
    call
    mlab.show() to actually display the grid, after this function returns
    """    
    Ns   = np.cumsum(np.array([l.shape[0] for l in lines]))
    Ntot = Ns[-1]
    Ns   = Ns[:-1]-1
    connects  = np.vstack([np.arange(0, Ntot-1.5), np.arange(1,Ntot-0.5)]).T
    connects  = np.delete(connects, Ns, axis=0)
    
    pts = np.vstack(lines)
    s   = np.ones(pts.shape[0])
    
    # Create the points
    src = mlab.pipeline.scalar_scatter(pts[:,0], pts[:,1], pts[:,2], s)
    src.mlab_source.dataset.lines = connects
    lines = mlab.pipeline.stripper(src)
    
    # Finally, display the set of lines
    mlab.pipeline.surface(lines, colormap='Accent', line_width=1, opacity=.4)



def test_plotlines():
    def identity(xyz):
        return xyz
    lines = gen_grid(identity, [0,0,0], [1,1,1])
    plot_lines(lines)


def test_plot3d():
    x  = np.linspace(0,1,101)
    yz = np.zeros([101,2])
    p  = np.c_[x,yz]
    from numpy import cos, sin
    a = 0.1
    R = np.array([[cos(a), sin(a),0],[-sin(a), cos(a), 0],[0,0,1]])
    p = p.dot(R.T)
    l = mlab.plot3d(p[:,0], p[:,1], p[:,2], tube_radius=None)
    return l


def load_clouds(file_num=109):
    """
    Loads the point-clouds saved by surgical sim.    
    
    The .npz file saved by simulation has name: cloud_XXX.npz
    where XXX is a random number.
    """
    data_dir    = '/home/ankush/sandbox/bulletsim/src/tests/ravens/recorded/'
    clouds_file = 'clouds_%d.npz'%file_num 
    fname = osp.join(data_dir, clouds_file)
    
    """
    the .npz file of point-clouds saved by simulation has source and target 
    clouds as following: 
      src_0, src_1, ...
      target_0, target_1, ...
    
    Hence, we can easily load separate them into source/ target clouds.
    """
    clouds = np.load(fname)
    tclouds = [clouds[n] for n in clouds.files if n.startswith('target')]
    sclouds = [clouds[n] for n in clouds.files if n.startswith('src')]
    return (sclouds, tclouds)


def test_tps_rpm(src_cloud, target_cloud):
    print colorize("Fitting tps-rpm ...", 'yellow')
    f = registration.fit_ThinPlateSpline_RotReg(src_cloud, target_cloud, bend_coef = 0.05, rot_coefs = [.1,.1,0], scale_coef=1)
    f = registration.tps_rpm(src_cloud, target_cloud, f_init=None, n_iter=1000, rad_init=.05, rad_final=0.0001, reg_init=10, reg_final=0.01)
    #f = registration.tps_rpm_rot_reg(src_cloud, target_cloud, n_iter=100, rad_init=.50, rad_final=0.001, reg_init=10, reg_final=0.00001)
    
    print colorize("Plotting grid ...", 'yellow')
    mean = np.mean(src_cloud, axis=0)
    print '\tmean : ', mean
    print '\tmins : ', np.min(src_cloud, axis=0)
    print '\tmaxes : ', np.max(src_cloud, axis=0)

    mins  = mean + [-0.03, -0.03, 0]
    maxes = mean + [0.03, 0.03, 0.01]
    lines = gen_grid(f.transform_points, mins=mins, maxes=maxes)
    plot_lines(lines)
    return f


def fit_and_plot(file_num):
    """
    does tps-rpm on first pair of clouds in file_num .npz file and then plots the grid and src and target clouds.
    src cloud     : red
    target clouds : blue
    warped (src---> target) : green
    """
   
    # source clouds
    (sc, tc) = load_clouds(file_num)
    sc = np.concatenate(sc)

    # target clouds
    tc = np.concatenate(tc)
    #bias = [0.1, 0, 0]
    #tc += bias
    
    
    f = test_tps_rpm(sc, tc)
    warped = f.transform_points(sc)

    mlab.points3d(sc[:,0], sc[:,1], sc[:,2], color=(1,0,0), scale_factor=0.001)
    mlab.points3d(tc[:,0], tc[:,1], tc[:,2], color=(0,0,1), scale_factor=0.001)
    mlab.points3d(warped[:,0], warped[:,1], warped[:,2], color=(0,1,0), scale_factor=0.001)
    
    mlab.show()
    

def rot_reg(src, target):    
    f = registration.fit_ThinPlateSpline_RotReg(src, target, bend_coef = .1, rot_coefs = [.1,.1,0], scale_coef=1)
    print colorize("Linear part of the warping function is:\n", "blue"), f.lin_ag


#todo: implement tps-rpm/ see if john has already
