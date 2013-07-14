from __future__ import division
import numpy as np

from rapprentice import tps
from rapprentice import registration
from rapprentice.colorize import *

import os.path as osp
import time
from mayavi import mlab
from mayavi_utils import plot_lines
from mayavi_plotter import *

from shape_context import *
from dtw.warp import *

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

    start = time.time()
    f, info = tps_sc_multi(src_clouds, target_clouds,
                           n_iter=20,
                           rad_init=0.3, rad_final=0.0001, # if testing for box-holes points, rad_final=0.00001
                           bend_init=10, bend_final=0.000005,
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


def fit_and_plot_dtw(file_num, draw_plinks=True, fine=False, augment_coords=False):
    """
    params:
      - draw_plinks [bool] : draws a line b/w each point in the source-cloud and its transformed location.

    does tps-rpm on first pair of clouds in file_num .npz file and then plots the grid and src and target clouds.
    src cloud     : red
    target clouds : blue
    warped (src---> target) : green
    """
    (sc, tc) = load_clouds(file_num)
    sc = sc[0]
    tc = tc[0]
    
    sc_src    = shape_context(sc)
    sc_target = shape_context(tc)  
    sc_dist   = shape_distance2d(sc_src, sc_target)
    sc_min_dist = np.min(sc_dist) + np.spacing(1.)
    dists     = np.exp(sc_dist/sc_min_dist)
    dtw_match = dtw_path(dtw_cumm_mat(dists))
    print dtw_match
    raw_input()

    # plot stuff
    plotter = PlotterInit()
    plot_reqs = []
    plot_reqs.append(gen_mlab_request(mlab.points3d, sc[:,0], sc[:,1], sc[:,2], color=(1,0,0), scale_factor=0.001))
    plot_reqs.append(gen_mlab_request(mlab.points3d, tc[:,0], tc[:,1], tc[:,2], color=(0,0,1), scale_factor=0.001))
    
    si, ti = np.nonzero(dtw_match)
    plinks = [np.c_[sc[si[i],:], tc[ti[i],:]].T for i in xrange(len(si))]
    plot_reqs.append(gen_custom_request('lines', lines=plinks, color=(0,1,0), line_width=2, opacity=1))

    Ts = pca_frame(sc)
    Tt = pca_frame(tc)
    plot_reqs.append(gen_custom_request('transform', Ts, size=0.01))
    plot_reqs.append(gen_custom_request('transform', Tt, size=0.01))

    for req in plot_reqs:
        plotter.request(req)
            

def tps_dtw(x_nd, y_md, n_iter = 100, bend_init=100, bend_final=.000001,
            plotter = None, plot_cb = None):
    
    regs = loglinspace(bend_init, bend_final, n_iter)
    
    src_nd  = x_nd
    targ_md = y_md
    sc_targ = shape_context(targ_md)
    
    for i in xrange(n_iter):
        sc_src = shape_context(src_nd)
        
        # do DTW:
        sc_dist   = shape_distance2d(sc_src, sc_targ)
        #sc_min_dist = np.min(sc_dist) + np.spacing(1.)
        #dists     = np.exp(sc_dist/sc_min_dist)
        dtw_match = dtw_path(dtw_cumm_mat(sc_dist)).toarray()

        dtw_rowsum = dtw_match.sum(axis=1)
        dtw_match  = dtw_match/dtw_rowsum[:,None]      
        tps_targ   = dtw_match.dot(targ_md)
          
        tps_src    = x_nd

        # for each match, put an error-term:
        #si, ti = np.nonzero(dtw_match)
        #tps_src = src_nd[si,:]
        #tps_targ = targ_md[ti,:]
        
        f = registration.fit_ThinPlateSpline(tps_src, tps_targ, bend_coef = regs[i], rot_coef = 10*regs[i])
        src_nd = f.transform_points(x_nd)

        
        if plot_cb and i%5==0:
            plot_cb(f)
            si, ti = np.nonzero(dtw_match)
            plinks = [np.c_[src_nd[si[id],:], targ_md[ti[id],:]].T for id in xrange(len(si))]
            plotter.request(gen_custom_request('lines', lines=plinks, color=(1,0,1), line_width=2, opacity=1))
            

        if plotter:
            plotter.request(gen_mlab_request(mlab.points3d, tps_targ[:,0], tps_targ[:,1], tps_targ[:,2], color=(1,1,0), scale_factor=0.001))
 
    return f


def test_tps_dtw(file_num, fine=False):
    (sc, tc) = load_clouds(file_num)
    x_nd = sc[0]
    y_md = tc[0]
    print colorize("Fitting tps-DTW ...", 'green', True)
    plotter = PlotterInit()

    def plot_cb(f):
        plot_requests = plot_warping(f.transform_points, x_nd, y_md, fine)
        for req in plot_requests:
            plotter.request(req)

    start = time.time()
    f = tps_dtw(x_nd, y_md, plot_cb=plot_cb, plotter=plotter)
    end = time.time()

    print colorize("TPS-DTW : took : %f seconds."%(end - start), "red", True)
    return f



def calc_corr_matrix(x_nd, y_md, r, p, dmult=None, n_iter=20):
    """
    sinkhorn procedure. see tps-rpm paper
    """
    n = x_nd.shape[0]
    m = y_md.shape[0]
    
    if dmult==None:
        dmult = np.ones((n,m))
    
    dist_nm = ssd.cdist(x_nd, y_md,'euclidean')
    prob_nm = np.exp(-dist_nm / r)
    prob_nm *= dmult

    prob_nm_orig = prob_nm.copy()
    for _ in xrange(n_iter):
        prob_nm /= (p*((n+0.)/m) + prob_nm.sum(axis=0))[None,:]  # cols sum to n/m
        prob_nm /= (p + prob_nm.sum(axis=1))[:,None] # rows sum to 1

    prob_nm = np.sqrt(prob_nm_orig * prob_nm)
    prob_nm /= (p + prob_nm.sum(axis=1))[:,None] # rows sum to 1
    return prob_nm


def calc_dist_matrix(x_n3, y_m3, r, p=.2, tech='exp'):
    """
    Combines the shape-context histogram distances with euclidean distances.
    """
    assert x_n3.ndim==y_m3.ndim==2, "Distance matrix error: inputs are not two dimensional"
    assert x_n3.shape[1]==y_m3.shape[1]==3, "Distance matrix error: pts are not three dimensional"
    
    dsc_nm = shape_distance2d(shape_context(x_n3), shape_context(y_m3))
    #dsc_nm = np.reciprocal(np.square(dsc_nm))
    dsc_nm = np.exp(-dsc_nm)
    #return calc_corr_matrix(x_n3, y_m3, r, p, dmult=None)
    return calc_corr_matrix(x_n3, y_m3, r, p, dmult=dsc_nm)


def tps_sc_multi(x_clouds, y_clouds,
                 n_iter = 100,
                 bend_init = 0.05, bend_final = .0001, 
                 rot_init = (0.1,0.1,0.025), rot_final=(0.001,0.001,0.00025),
                 scale_init=1, scale_final=0.001, 
                 rad_init = .5, rad_final = .0005,
                 verbose=False, f_init = None, return_full = False,
                 plotting_cb=None, plotter=None):
    """
    Combines the shape-context distances with tps-rpm.
    """  

    assert len(x_clouds)==len(y_clouds), "Different number of point-clouds in source and target."

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

            corr_nm = calc_dist_matrix(xwarped_nd, y_md, r=rads[i], p=.2)

            wt_n = corr_nm.sum(axis=1) # gives the row-wise sum of the corr_nm matrix
            goodn = wt_n > 0.
            targ_Nd = np.dot(corr_nm[goodn, :]/wt_n[goodn][:,None], y_md) # calculate the average points based on softmatching

            target_pts.append(targ_Nd)
            good_inds.append(goodn)  
            wt.append(wt_n[goodn])

        target_pts = np.concatenate(target_pts)
        good_inds  = np.concatenate(good_inds)
        source_pts = combined_x[good_inds]
        wt         = np.concatenate(wt)

        assert len(target_pts)==len(source_pts)==len(wt), "Lengths are not equal. Error!"
        f      = registration.fit_ThinPlateSpline(source_pts, target_pts, bend_coef = regs[i], wt_n = wt_n[good_inds], rot_coef = 10*regs[i])

        mscore = registration.match_score(source_pts, target_pts)
        tscore = tps.tps_cost(f.lin_ag, f.trans_g, f.w_ng, source_pts, target_pts, regs[-1])
        print colorize("\ttps-mix : iter : %d | fit distance : "%i, "red") , colorize("%g"%mscore, "green"), colorize(" | tps score: %g"%tscore, "blue")

        if plotting_cb and i%5==0:
            plotting_cb(f)

        # just plots the "target_pts" : the matched up points found by the correspondences.
        if plotter:
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
