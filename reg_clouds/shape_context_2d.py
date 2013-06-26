import numpy as np
import scipy.spatial.distance as ssd
from rapprentice.registration import loglinspace
from scipy.sparse import *
import os.path as osp
import scipy.io as sio
from shape_context import load_clouds


def shape_context2d(p, mean_dist=None, r_inner=1./8, r_outer=2., nbins_r=5, nbins_theta=12, outliers=None, sparse=False):
    """
    Computes the shape-context log-polar histograms at each point in p -- the point cloud.
    p is a Nxd matrix of points.
    """

    N, d = p.shape
    assert d==2, "shape_context2d requires 2D point data."
    pt_nd   = p

    # compute the coordinates : r,theta, phi
    dists_nn    = ssd.cdist(pt_nd, pt_nd)
    if mean_dist==None:
        mean_dist      = np.mean(dists_nn)
        dists_nn       = dists_nn/mean_dist

    # theta_nn are in [0,2pi)
    dx_nn, dy_nn    = pt_nd.T[:,None,:]-pt_nd.T[:,:,None]
    theta_nn        = np.arctan2(dy_nn, dx_nn)
    theta_nn        = np.fmod(np.fmod(theta_nn,2*np.pi)+2*np.pi,2*np.pi)

    # define histogram edges
    r_edges     = np.concatenate(([0], loglinspace(r_inner, r_outer, nbins_r)))
    theta_edges = np.linspace(0, 2*np.pi, nbins_theta+1)
    
    combined_2nn = np.array([dists_nn, theta_nn])

    # compute the bins : 4 dimensional matrix.
    # r,t,p are the number of bins of radius, theta, phi
    sc_nrt = np.zeros((N, nbins_r, nbins_theta))
    for i in xrange(N):
        hist, edges = np.histogramdd(combined_2nn[:,i,:].T, bins=[r_edges, theta_edges])
        sc_nrt[i,:,:] = hist

    sc_nrt = sc_nrt.reshape(N, nbins_r*nbins_theta)

    # just r binning:
    rbins_nr = np.zeros((N, nbins_r))
    for i in xrange(N):
        dat = combined_2nn[0,i,:].T
        hist, edges = np.histogramdd(dat, bins=[r_edges])
        rbins_nr[i,:] = hist

    # just theta binning
    tbins_nt = np.zeros((N, nbins_theta))
    for i in xrange(N):
        dat = combined_2nn[1,i,:].T
        hist, edges = np.histogramdd(dat, bins=[theta_edges])
        tbins_nt[i,:] = hist

    return (sc_nrt, mean_dist, dists_nn, theta_nn, rbins_nr, tbins_nt) 


def test_shape_context_2d(file_num):
    """
    Loads a point-cloud from simulation data.
    - Discards the z-axis.
      - Computes the 2d shape-context.
        - Saves the shape-context matrix in matlab format.
    """
    (src, target) = load_clouds(file_num)
    src    = src[0][:,0:2]
    target = target[0][:,0:2]

    sc_src, mdist, r_nn, theta_nn, r_bins, theta_bins  = shape_context2d(src)
    sc_target, _, _, _, _,_,                           = shape_context2d(target)
    
    sio.savemat('/home/ankush/Desktop/shape_context/sc_%d.mat'%file_num, 
                {'src':src, 'src_mean_dist':mdist, 'sc_src':sc_src,
                 'src_r':r_nn, 'src_theta':theta_nn, 'r_bins':r_bins, 't_bins':theta_bins, 
                 'sc_target':sc_target})
