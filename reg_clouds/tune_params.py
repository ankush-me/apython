from rapprentice import registration
from rapprentice.colorize import *
from easyInput.slider import easyInput
import os.path as osp
import numpy as np
import time
from mayavi import mlab


def gen_grid(f, mins, maxes, xres = .1, yres = .1, zres = .04):
    """
    generate 3d grid and warps it using the function f.
    """
    xmin, ymin, zmin = mins
    xmax, ymax, zmax = maxes

    nfine = 30
    xcoarse = np.arange(xmin, xmax, xres)
    ycoarse = np.arange(ymin, ymax, yres)
    if zres == -1: zcoarse = [(zmin+zmax)/2.]
    else: zcoarse = np.arange(zmin, zmax, zres)
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


def plotLines(lines):
    """
    input  : a list of m matrices of shape nx3
             each list is interpreted as one line
    output : plot each line in mayavi
    
    adapted from : http://docs.enthought.com/mayavi/mayavi/auto/example_plotting_many_lines.html
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
    mlab.show()


def test_plotlines():
    def identity(xyz):
        return xyz
    lines = gen_grid(identity, [0,0,0], [1,1,1])
    plotLines(lines)

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

#########################################

FILE_NUM = 109

data_dir  = '/home/ankush/sandbox/bulletsim/src/tests/ravens/recorded/'
clouds_file = 'clouds_%d.npz'%FILE_NUM 
fname = osp.join(data_dir, clouds_file)

clouds = np.load(fname)

tclouds = [clouds[n] for n in clouds.files if n.startswith('target')]
sclouds = [clouds[n] for n in clouds.files if n.startswith('src')]


def rot_reg_works():
    from np import sin,cos
    from rapprentice import registration
    import fastrapp
    x = np.random.randn(100,3)
    a = .1
    R = np.array([[cos(a), sin(a),0],[-sin(a), cos(a), 0],[0,0,1]])
    y = x.dot(R.T)
    f = registration.fit_ThinPlateSpline_RotReg(x, y, bend_coef = .1, rot_coefs = [.1,.1,0], scale_coef = 1)
    assert np.allclose(R.T, f.lin_ag, atol = 1e-4)
#     

def rot_reg(src, target):    
    f = registration.fit_ThinPlateSpline_RotReg(src, target, bend_coef = .1, rot_coefs = [.1,.1,0], scale_coef=1)
    print colorize("Linear part of the warping function is:\n", "blue"), f.lin_ag


# rot_reg(sclouds[0], tclouds[0])
#test_plot3d()

#mlab.show()

test_plotlines()
test_plot3d()










