import numpy as np

from rapprentice import tps
from rapprentice import registration
from rapprentice.colorize import *

import os.path as osp
import time
from mayavi import mlab
from mayavi_utils import plot_lines
from mayavi_plotter import *

import sqpregpy 



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



def test_tps_mix(src_clouds, target_clouds, fine=False, augment_coords=False, scale_down=500.0):
    """
    FINE: set to TRUE if you want to plot a very fine grid.
    """
    print colorize("Fitting tps-rpm ...", 'green', True)
    
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

    start = time.time()
    f, info = tps_multi_mix(src_clouds, target_clouds,
                                    x_aug=x_aug, y_aug=y_aug,
                                    n_iter=20,
                                    rad_init=0.3, rad_final=0.0001, # if testing for box-holes points, rad_final=0.00001
                                    bend_init=10, bend_final=0.00001,
                                    rot_init = (0.01,0.01,0.0025), rot_final=(0.00001,0.00001,0.0000025),
                                    scale_init=50, scale_final=0.00001,
                                    return_full=True,
                                    plotting_cb=plot_cb, plotter=plotter)

    print "(src, w, aff, trans) : ", f.x_na.shape, f.w_ng.shape, f.lin_ag.shape, f.trans_g.shape

    end = time.time()
    print colorize("Iterative : took : %f seconds."%(end - start), "red", True)

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
    test_tps_mix(sc, tc, fine=fine, augment_coords=augment_coords)


def tps_multi_mix(x_clouds, y_clouds,
                   n_iter = 100,
                   bend_init = 0.05, bend_final = .0001, 
                   rot_init = (0.1,0.1,0.025), rot_final=(0.001,0.001,0.00025),
                   scale_init=1, scale_final=0.001, 
                   rad_init = .5, rad_final = .0005,
                   x_aug=None, y_aug=None,
                   verbose=False, f_init = None, return_full = False,
                   plotting_cb=None, plotter=None):
    """
    x_aug : dict of matrices of extra coordinates for x_clouds. The key is the index of the cloud.
    y_aug : similar to x_aug for y_clouds
    

    Similar to tps_rpm_regrot except that it accepts a 
    LIST of source and target point clouds and registers 
    a cloud in the source to the corresponding one in the target.  
    
    For details on the various parameters check the doc of tps_rpm_regrot.
    """  

    assert len(x_clouds)==len(y_clouds), "Different number of point-clouds in source and target."

    if x_aug==None or y_aug==None:
        x_aug = y_aug = {}
  
    #flatten the list of point clouds into one big point cloud
    combined_x = np.concatenate(x_clouds) 
    combined_y = np.concatenate(y_clouds)

    # concatenate the clouds into one big cloud
    _,d  = combined_x.shape

    regs     = registration.loglinspace(bend_init, bend_final, n_iter)
    rads     = registration.loglinspace(rad_init, rad_final, n_iter)
    scales   = registration.loglinspace(scale_init, scale_final, n_iter)
    rots     = registration.loglinspace_arr(rot_init, rot_final, n_iter)

    # initialize the function f.
    if f_init is not None: 
        f = f_init  
    else:
        f         = registration.ThinPlateSpline(d)
        f.trans_g = np.median(combined_y,axis=0) - np.median(combined_x,axis=0)

    # iterate b/w calculating correspondences and fitting the transformation.
    for i in xrange(n_iter):
        target_pts   = []
        good_inds    = []
        wt           = []

        for j in xrange(len(x_clouds)): #process a pair of point-clouds
            x_nd = x_clouds[j]
            y_md = y_clouds[j]

            assert x_nd.ndim==y_md.ndim==2, "tps_rpm_reg_rot_multi : Point clouds are not two dimensional arrays"
                        
            xwarped_nd = f.transform_points(x_nd)
            
            # use augmented coordinates.
            if x_aug.has_key(j) and y_aug.has_key(j):
                corr_nm = registration.calc_correspondence_matrix(np.c_[xwarped_nd, x_aug[j]], np.c_[y_md, y_aug[j]], r=rads[i], p=.2)
            else:
                corr_nm = registration.calc_correspondence_matrix(xwarped_nd, y_md, r=rads[i], p=.2)

            wt_n = corr_nm.sum(axis=1) # gives the row-wise sum of the corr_nm matrix
            goodn = wt_n > 0.1
            targ_Nd = np.dot(corr_nm[goodn, :]/wt_n[goodn][:,None], y_md) # calculate the average points based on softmatching

            target_pts.append(targ_Nd)
            good_inds.append(goodn)  
            wt.append(wt_n[goodn])

        target_pts = np.concatenate(target_pts)
        good_inds  = np.concatenate(good_inds)
        source_pts = combined_x[good_inds]
        wt         = np.concatenate(wt)

        assert len(target_pts)==len(source_pts)==len(wt), "Lengths are not equal. Error!"
        ## USE SQP BASED FITTING:       
        f      = registration.fit_ThinPlateSpline(source_pts, target_pts, bend_coef = regs[i], wt_n = wt_n[good_inds], rot_coef = 10*regs[i])

#         A, B, c = sqpregpy.fit_sqp(source_pts, target_pts, wt_n[good_inds], rots[i], scales[i], regs[i], True, False)
#         c  = c.flatten()
#         f = registration.ThinPlateSpline()
#         f.x_na    = source_pts
#         f.w_ng    = A
#         f.lin_ag  = B
#         f.trans_g = c


        mscore = registration.match_score(source_pts, target_pts)
        tscore = tps.tps_cost(f.lin_ag, f.trans_g, f.w_ng, source_pts, target_pts, regs[-1])
        print colorize("\ttps-mix : iter : %d | fit distance : "%i, "red") , colorize("%g"%mscore, "green"), colorize(" | tps score: %g"%tscore, "blue")

        if False and plotting_cb and i%5==0:
            plotting_cb(f)

        # just plots the "target_pts" : the matched up points found by the correspondences.
        if False and plotter:
            plotter.request(gen_mlab_request(mlab.points3d, target_pts[:,0], target_pts[:,1], target_pts[:,2], color=(1,1,0), scale_factor=0.001))

        # return if source and target match up well
        if tscore < 1e-6:
            break

    if return_full:
        info = {}
        info["cost"] = tps.tps_cost(f.lin_ag, f.trans_g, f.w_ng, source_pts, target_pts, regs[-1])
        return f, info
    else:
        return f
