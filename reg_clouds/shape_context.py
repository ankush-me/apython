import numpy as np
import scipy.spatial.distance as ssd
from mayavi_utils import *
from mayavi import mlab
from rapprentice.registration import loglinspace
from scipy.sparse import *

def shape_context(p, median_dist=None, r_inner=1./8, r_outer=2, nbins_r=5, nbins_theta=12, nbins_phi=6, outliers=None, sparse=False):
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
    if sparse:
        sc_nrtp = lil_matrix((N, nbins_r*nbins_theta*nbins_phi))
        for i in xrange(N):
            hist, edges = np.histogramdd(combined_3nn[:,i,:].T, bins=[r_edges, theta_edges, phi_edges])
            hist = csc_matrix(hist.flatten())
            sc_nrtp[i,:] = hist
        sc_nrtp = csc_matrix(sc_nrtp)
    else:
        sc_nrtp = np.zeros((N, nbins_r, nbins_theta, nbins_phi))
        for i in xrange(N):
            hist, edges = np.histogramdd(combined_3nn[:,i,:].T, bins=[r_edges, theta_edges, phi_edges])
            sc_nrtp[i,:,:,:] = hist

    return sc_nrtp 


def plot_shape_context(index, sc):
    """
    Plots the index-th shape-context in the 4-dimensional shape-context.
    Plots the r,theta,phi counts in 3 dimensional space.
    
    get the non-zero elements and their indices. 
    then pass the count as the scalar.
    """
    pass


def shape_distance(sc1, sc2):
    """
    Computes the Chi-squared distance b/w shape-contexts sc1 and sc2.
    returns an sc1.len x sc2.len distance matrix
    """
    assert sc1.ndim==sc2.ndim==4, "shape contexts are not four-dimensionsal"    

    n1, r1, t1, p1 = sc1.shape
    n2, r2, t2, p2 = sc2.shape
    
    assert r1==r2 and t1==t2 and p1==p2, "shape-contexts have different bin-sizes."
    
    # flatten the shape-contexts:
    sc1_flat = sc1.reshape((n1, r1*t1*p1))
    sc2_flat = sc2.reshape((n2, r2*t2*p2))

    # normalize
    eps = np.spacing(1)
    sc1_norm = sc1_flat/ np.c_[eps + np.sum(sc1_flat, axis=1)]
    sc2_norm = sc2_flat/ np.c_[eps + np.sum(sc2_flat, axis=1)]
    
    sc1_norm = sc1_norm[:,None,:]
    sc2_norm = sc2_norm[None,:,:]
    
    dist = 0.5* np.sum( (sc1_norm - sc2_norm)**2 / (sc1_norm + sc2_norm + eps) , axis=2)

    assert dist.shape==(n1,n2), "distance metric shape mis-match. Error in code."
    return dist


def gen_rand_data(N=200, s=1.):
    p = np.zeros((N,3))
    for i in xrange(N):
        p[i,:] = (i/(N+0.0),0,0)    
    
    noise = np.random.randn(N,3)
    noise = s*np.c_[noise[:,0]/10, noise[:,0]/5, noise[:,0]/8]
    p = p + noise    

    #mlab.points3d(p[:,0], p[:,1], p[:,2], scale_factor=0.1)
    return p
    

if __name__ == '__main__':
    p1 = gen_rand_data()
    p2 = gen_rand_data(100, s=1.2)
    
    sc1 = shape_context(p1, sparse=False)
    sc2 = shape_context(p2, sparse=False)
    
    shape_distance(sc1, sc2)
