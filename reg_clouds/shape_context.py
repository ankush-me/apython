import numpy as np
import scipy.spatial.distance as ssd
from mayavi_utils import *
from mayavi import mlab
from rapprentice.registration import loglinspace


def shape_context(p, median_dist=None, r_inner=1./8, r_outer=2, nbins_r=5, nbins_theta=12, nbins_phi=6, outliers=None):
    """
    Computes the shape-context log-polar histograms at each point in p -- the point cloud.
    
    p is a Nxd matrix of points.
    """
    
    N, d = p.shape
    assert d==3, "shape_context is implemented only for three dimensions"
    
    p_mean = np.mean(p, axis=0)
    p_centered = p - p_mean
    _,_,v      = np.linalg.svd(p_centered, full_matrices=True)
    pt_nd      = np.dot(p_centered, v.T)
    
    # compute the coordinates : r,theta, phi
    dists    = ssd.pdist(pt_nd, 'euclidean')
    if median_dist==None:
        median_dist = np.median(dists)
        dists       = dists/median_dist
    dists_nn        = ssd.squareform(dists)

    # theta_nn are in [0,2pi)
    dx_nn, dy_nn, dz_nn  = pt_nd.T[:,:,None]-pt_nd.T[:,None,:]
    theta_nn        = np.arctan2(dy_nn, dx_nn)
    theta_nn        = np.mod(np.mod(theta_nn,2*np.pi)+2*np.pi,2*np.pi)

    # phi_nn are in [-pi/2, pi/2]
    dist_xy_nn    = (dx_nn**2 + dy_nn**2)**0.5
    phi_nn        = np.arctan2(dz_nn, dist_xy_nn)

    # define histogram edges
    r_edges     = np.concatenate(([0], loglinspace(r_inner, r_outer, nbins_r)))
    theta_edges = np.linspace(0, 2*np.pi, nbins_theta+1)
    phi_edges   = np.linspace(-np.pi/2, np.pi/2, nbins_phi+1) 


    combined_3nn = np.array([dists_nn, theta_nn, phi_nn])
    
    # compute the bins : 4 dimensional matrix.
    # r,t,p are the number of bins of radius, theta, phi
    sc_nrtp = np.zeros((N, nbins_r, nbins_theta, nbins_phi))
    
    for i in xrange(N):
        hist, edges = np.histogramdd(combined_3nn[:,i,:].T, bins=[r_edges, theta_edges, phi_edges])
        sc_nrtp[i,:,:,:] = hist 

    return sc_nrtp 



if __name__=='__main__':
    N = 100
    p = np.zeros((N,3))
    for i in xrange(N):
        p[i,:] = (i/(N+0.0),0,0)    
    
    noise = np.random.randn(100,3)
    noise = np.c_[noise[:,0]/10, noise[:,0]/5, noise[:,0]/8]
    p = p + noise    

    mlab.points3d(p[:,0], p[:,1], p[:,2], scale_factor=0.1)
    shape_context(p)
