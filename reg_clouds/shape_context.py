import numpy as np
import scipy.spatial.distance as ssd
from mayavi_utils import *
from mayavi import mlab
from rapprentice.registration import loglinspace


def shape_context(p, median_dist=None, r_inner=1./8, r_outer=2, nbins_r=5, nbins_theta=12, nbins_phi=5, outliers=None):
    """
    Computes the shape-context log-polar histograms at each point in p -- the point cloud.
    
    p is a Nxd matrix of points.
    """
    N, d = p.shape
    
    assert d==3, "shape_context is implemented only for three dimensions"
    
    p_mean = np.mean(p, axis=0)
    p_centered = p - p_mean
    _,_,v      = np.linalg.svd(p_centered, full_matrices=1)
    pt_nd      = np.dot(p_centered, v.T)
    

    dx_nn, dy_nn, dz_nn  = pt_nd.T[:,:,None]-pt_nd.T[:,None,:]
    
    dists    = ssd.cdist(pt_nd, 'euclidean')
    if median_dist==None:
        median_dist = np.median(dists)
        dists       = dists/median_dist

    # compute bin-membership for lengths
    r_bin_edges  = np.concatenate(([-1], loglinspace(r_inner, r_outer, nbins_r)))
    r_bin_id_nn  = ssd.squareform(np.digitize(dists, bins=r_bin_edges))
    fz_nn = r_bin_id_nn < nbins_r # flag all points inside outer boundary


    # theta_nn are in [0,2pi)
    theta_nn = np.arctan2(dy_nn, dx_nn)
    theta_nn = np.mod(np.mod(theta_nn,2*np.pi)+2*np.pi,2*np.pi)


    # phi_nn are in [-pi/2, pi/2]
    dist_xy_nn  = (dx_nn**2 + dy_nn**2)**0.5
    phi_nn = np.arctan2(dz_nn, dist_xy_nn)


    
    
    nbins  = nbins_r * nbins_theta * nbins_phi
    sc     = np.zeros(N,nbins) 
    

    # quantize to a fixed set of angles (bin edges lie on 0,(2*pi)/k,...2*pi
    theta_array_q = 1+floor(theta_array_2/(2*pi/nbins_theta));

    nbins=nbins_theta*nbins_r;
    BH=zeros(nsamp,nbins);
    for n=1:nsamp
        fzn=fz(n,:)&in_vec;
        Sn=sparse(theta_array_q(n,fzn),r_array_q(n,fzn),1,nbins_theta,nbins_r);
        BH(n,:)=Sn(:)';

    
    
    
    
        

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
    
    
    
