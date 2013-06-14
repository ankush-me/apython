from rapprentice import registration
from rapprentice.colorize import *
from easyInput.slider import easyInput
import os.path as osp
import numpy as np
import time
from mayavi import mlab


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




def plot_lines(lines, color=(1,1,1), line_width=1, opacity=0.4):
    """
    input  :
    
      - lines :  a list of m matrices of shape nx3
                 each list is interpreted as one line
                 
      - color : (r,g,b) values for the lines
      - line_width : width of the lines
      - opacity    : opacity of the lines

             
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
    surf = mlab.pipeline.surface(lines, line_width=line_width, opacity=opacity)
    
    # set the color of the lines
    r,g,b = color
    color = 255*np.array((r,g,b, 1))
    surf.module_manager.scalar_lut_manager.lut.table = np.array([color, color])
    
    

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



def test_tps_rpm_regrot_multi(src_cloud, target_cloud, fine=False):
    """
    FINE: set to TRUE if you want to plot a very fine grid.
    """
    
    print colorize("Fitting tps-rpm ...", 'green', True)
    #f = registration.fit_ThinPlateSpline_RotReg(src_cloud, target_cloud, bend_coef = 0.05, rot_coefs = [.1,.1,0], scale_coef=1)
    #f = registration.tps_rpm(src_cloud, target_cloud, f_init=None, n_iter=1000, rad_init=.05, rad_final=0.0001, reg_init=10, reg_final=0.01)

    f, info = registration.tps_rpm_regrot_multi(src_cloud, target_cloud, n_iter=50,
                                    rad_init=0.3, rad_final=0.0001, 
                                    bend_init=10, bend_final=0.00001,
                                    scale_init=1, scale_final=0.0001,
                                    return_full=True)

    print colorize("Plotting grid ...", 'yellow')
    mean = np.mean(np.concatenate(src_cloud), axis=0)

    print '\tmean : ', mean
    print '\tmins : ', np.min(np.concatenate(src_cloud), axis=0)
    print '\tmaxes : ', np.max(np.concatenate(src_cloud), axis=0)

    mins  = mean + [-0.2, -0.2, 0]
    maxes = mean + [0.2, 0.2, 0.01]

    lines = []
    if fine:
        lines = gen_grid2(f.transform_points, mins=mins, maxes=maxes, xres=0.005, yres=0.005, zres=0.002)
    else:
        lines = gen_grid(f.transform_points, mins=mins, maxes=maxes)

    plot_lines(lines, color=(0,0.5,0.3))

    return f


def fit_and_plot(file_num, draw_plinks=True, fine=False):
    """
    params:
      - draw_plinks [bool] : draws a line b/w each point in the source-cloud and its transformed location.

    does tps-rpm on first pair of clouds in file_num .npz file and then plots the grid and src and target clouds.
    src cloud     : red
    target clouds : blue
    warped (src---> target) : green
    """

    (sc, tc) = load_clouds(file_num)
    f = test_tps_rpm_regrot_multi(sc, tc, fine=fine)

    # plot the points    
    sc = np.concatenate(sc)
    tc = np.concatenate(tc)

    warped = f.transform_points(sc)
    mlab.points3d(sc[:,0], sc[:,1], sc[:,2], color=(1,0,0), scale_factor=0.001)
    mlab.points3d(tc[:,0], tc[:,1], tc[:,2], color=(0,0,1), scale_factor=0.001)
    mlab.points3d(warped[:,0], warped[:,1], warped[:,2], color=(0,1,0), scale_factor=0.001)

    if draw_plinks:
        plinks = [np.c_[ps, pw].T for ps,pw in zip(sc, warped)]
        plot_lines(plinks, color=(0.5,0,0), line_width=2, opacity=1)

    mlab.show()
    


def rot_reg(src, target):    
    f = registration.fit_ThinPlateSpline_RotReg(src, target, bend_coef = .1, rot_coefs = [.1,.1,0], scale_coef=1)
    print colorize("Linear part of the warping function is:\n", "blue"), f.lin_ag


#todo: implement tps-rpm/ see if john has already
