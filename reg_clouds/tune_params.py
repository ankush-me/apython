from rapprentice import registration
from rapprentice.colorize import *
from easyInput.slider import easyInput
import os.path as osp
import numpy as np
import time
from mayavi import mlab

from mayavi_utils import plot_lines
from mayavi_plotter import *

from sqpregpy import *

def gen_grid(f, mins, maxes, ncoarse=10, nfine=30):
    """
    generate 3d grid and warps it using the function f.
    The grid is based on the number of lines (ncoarse & nfine).
    """    
    xmin, ymin, zmin = mins
    xmax, ymax, zmax = maxes

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


def gen_grid2(f, mins, maxes, xres = .01, yres = .01, zres = .01):
    """
    generate 3d grid and warps it using the function f.
    
    The grid is based on the resolution specified.
    
    """    
    xmin, ymin, zmin = mins
    xmax, ymax, zmax = maxes

    xcoarse = np.arange(xmin, xmax+xres/10., xres)
    ycoarse = np.arange(ymin, ymax+yres/10., yres)
    zcoarse = np.arange(zmin, zmax+zres/10., zres)
    
    
    xfine = np.arange(xmin, xmax+xres/10., xres/5.)
    yfine = np.arange(ymin, ymax+yres/10., yres/5.)
    zfine = np.arange(zmin, zmax+zres/10., zres/5.)

    lines = []
    if len(zcoarse) > 1:    
        for x in xcoarse:
            for y in ycoarse:
                xyz = np.zeros((len(zfine), 3))
                xyz[:,0] = x
                xyz[:,1] = y
                xyz[:,2] = zfine
                lines.append(f(xyz))

    for y in ycoarse:
        for z in zcoarse:
            xyz = np.zeros((len(xfine), 3))
            xyz[:,0] = xfine
            xyz[:,1] = y
            xyz[:,2] = z
            lines.append(f(xyz))
        
    for z in zcoarse:
        for x in xcoarse:
            xyz = np.zeros((len(yfine), 3))
            xyz[:,0] = x
            xyz[:,1] = yfine
            xyz[:,2] = z
            lines.append(f(xyz))

    return lines


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
    
    Hence, we can easily load and separate them into source/ target clouds.
    """
    clouds = np.load(fname)
    tclouds = [clouds[n] for n in clouds.files if n.startswith('target')]
    sclouds = [clouds[n] for n in clouds.files if n.startswith('src')]
    return (sclouds, tclouds)



def plot_warping(f, src, target, fine=True, draw_plinks=True):
    """
    function to plot the warping as defined by the function f.
    src : nx3 array
    target : nx3 array
    fine   : if fine grid else coarse grid.
    """
    print colorize("Plotting grid ...", 'blue', True)
    mean = np.mean(src, axis=0)

    print '\tmean : ', mean
    print '\tmins : ', np.min(src, axis=0)
    print '\tmaxes : ', np.max(src, axis=0)

    mins  = mean + [-0.1, -0.1, -0.01]
    maxes = mean + [0.1, 0.1, 0.01]

    grid_lines = []
    if fine:
        grid_lines = gen_grid2(f, mins=mins, maxes=maxes, xres=0.005, yres=0.005, zres=0.002)
    else:
        grid_lines = gen_grid(f, mins=mins, maxes=maxes)

    
    plotter_requests = []
    plotter_requests.append(gen_mlab_request(mlab.clf))
    plotter_requests.append(gen_custom_request('lines', lines=grid_lines, color=(0,0.5,0.3)))
    
    warped = f(src)
    
    plotter_requests.append(gen_mlab_request(mlab.points3d, src[:,0], src[:,1], src[:,2], color=(1,0,0), scale_factor=0.001))
    plotter_requests.append(gen_mlab_request(mlab.points3d, target[:,0], target[:,1], target[:,2], color=(0,0,1), scale_factor=0.001))
    plotter_requests.append(gen_mlab_request(mlab.points3d, warped[:,0], warped[:,1], warped[:,2], color=(0,1,0), scale_factor=0.001))

    if draw_plinks:
        plinks = [np.c_[ps, pw].T for ps,pw in zip(src, warped)]
        plotter_requests.append(gen_custom_request('lines', lines=plinks, color=(0.5,0,0), line_width=2, opacity=1))
                                
    return plotter_requests



def test_tps_rpm_regrot_multi(src_clouds, target_clouds, fine=False, augment_coords=False, scale_down=500.0):
    """
    FINE: set to TRUE if you want to plot a very fine grid.
    """
    print colorize("Fitting tps-rpm ...", 'green', True)
    #f = registration.fit_ThinPlateSpline_RotReg(src_cloud, target_cloud, bend_coef = 0.05, rot_coefs = [.1,.1,0], scale_coef=1)
    #f = registration.tps_rpm(src_cloud, target_cloud, f_init=None, n_iter=1000, rad_init=.05, rad_final=0.0001, reg_init=10, reg_final=0.01)

    plotter = PlotterInit()

    def plot_cb(f):
        plot_requests = plot_warping(f.transform_points, np.concatenate(src_clouds), np.concatenate(target_clouds), fine)
        for req in plot_requests:
            plotter.request(req)


    x_aug = {}
    y_aug = {}
    if augment_coords:
        for i,c in enumerate(src_clouds):
            x_aug[i] = np.abs(np.arange(len(c)) - len(c)/2)/scale_down
        for i,c in enumerate(target_clouds):
           y_aug[i] = np.abs(np.arange(len(c)) - len(c)/2)/scale_down
#         for c in src_clouds:
#             c[:,2] = np.abs(np.arange(len(c)) - len(c)/2)/scale_down
#         for c in target_clouds:
#             c[:,2] = np.abs(np.arange(len(c)) - len(c)/2)/scale_down


    f, info = registration.tps_rpm_regrot_multi(src_clouds, target_clouds,
                                    x_aug=x_aug, y_aug=y_aug,
                                    n_iter=15,
                                    n_iter_powell_init=50, n_iter_powell_final=50,
                                    rad_init=0.3, rad_final=0.0001, # if testing for box-holes points, rad_final=0.00001
                                    bend_init=10, bend_final=0.00001,
                                    rot_init = (0.01,0.01,0.0025), rot_final=(0.00001,0.00001,0.0000025),
                                    scale_init=10, scale_final=0.0000001,
                                    return_full=True,
                                    plotting_cb=plot_cb, plotter=plotter)

    print "(src, w, aff, trans) : ", f.x_na.shape, f.w_ng.shape, f.lin_ag.shape, f.trans_g.shape

    
    plot_requests = plot_warping(f.transform_points,np.concatenate(src_clouds), np.concatenate(target_clouds), fine)
    for req in plot_requests:
        plotter.request(req)

    return f


def fit_and_plot(file_num, draw_plinks=True, fine=False, augment_coords=False):
    """
    params:
      - draw_plinks [bool] : draws a line b/w each point in the source-cloud and its transformed location.

    does tps-rpm on first pair of clouds in file_num .npz file and then plots the grid and src and target clouds.
    src cloud     : red
    target clouds : blue
    warped (src---> target) : green
    """
    (sc, tc) = load_clouds(file_num)

    test_tps_rpm_regrot_multi(sc, tc, fine=fine, augment_coords=augment_coords)



def test_sqpregrot(src, target,  bend_coeff=0.00001,  rot_coeff=np.array((0.00001,0.00001,0.0000025)), scale_coeff=0.0000001, corres_coeff=0.0001):
    A, B, c = fit_tps_sqp(src, target, rot_coeff, scale_coeff, bend_coeff, corres_coeff, True)
    c = c.flatten()
    print A.shape, B.shape, c.shape

    f = registration.ThinPlateSpline()
    f.x_na = src
    f.w_ng = A
    f.lin_ag = B
    f.trans_g = c.T
    
    plotter = PlotterInit()
    plot_requests = plot_warping(f.transform_points,src, target, True)
    for req in plot_requests:
        plotter.request(req)


def fit_and_plot_sqp(file_num, draw_plinks=True, fine=False):
    (sc, tc) = load_clouds(file_num)
    sc = sc[0]
    tc = tc[0]
    test_sqpregrot(sc, tc)


def rot_reg(src, target):    
    f = registration.fit_ThinPlateSpline_RotReg(src, target, bend_coef = .1, rot_coefs = [.1,.1,0], scale_coef=1)
    print colorize("Linear part of the warping function is:\n", "blue"), f.lin_ag



























