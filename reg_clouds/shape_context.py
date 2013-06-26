import numpy as np
import scipy.spatial.distance as ssd
from mayavi_utils import *
from mayavi_plotter import *
from mayavi import mlab
from rapprentice.registration import loglinspace
import scipy.sparse as sparse
import os.path as osp
import scipy.io as sio


def shape_context(p, median_dist=None, r_inner=1./8, r_outer=2., nbins_r=5, nbins_theta=12, nbins_phi=6, outliers=None, sparse=False):
    """
    Computes the shape-context log-polar histograms at each point in p -- the point cloud.
    p is a Nxd matrix of points.
    """
    N, d = p.shape
    assert d==3, "shape_context is implemented only for three dimensions"

    p_mean = np.mean(p, axis=0)
    p_centered = p - p_mean
    T = pca_frame(p)
    R = T[0:3,0:3]
    pt_nd      = np.dot(p_centered, R)
    #pt_nd      = p_centered

    # compute the coordinates : r,theta, phi
    dists    = ssd.pdist(pt_nd, 'euclidean')
    if median_dist==None:
        median_dist = np.median(dists) 
    dists       = dists/median_dist
    dists_nn    = ssd.squareform(dists)

    # theta_nn are in [0,2pi)
    dx_nn, dy_nn, dz_nn  = pt_nd.T[:,None,:]-pt_nd.T[:,:,None]
    theta_nn             = np.arctan2(dy_nn, dx_nn)
    theta_nn             = np.fmod(np.fmod(theta_nn,2*np.pi)+2*np.pi,2*np.pi)

    # phi_nn are in [-pi/2, pi/2]
    dist_xy_nn    = np.sqrt(np.square(dx_nn) + np.square(dy_nn))
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

    if sparse: # convert to sparse representation
        sc_nrtp = sc_nrtp.reshape(N, nbins_r*nbins_theta*nbins_phi)
        sc_nrtp = sparse.csc_matrix(sc_nrtp)

    return sc_nrtp 


def plot_shape_context(index, sc):
    """
    Plots the index-th shape-context in the 4-dimensional shape-context.
    Plots the r,theta,phi counts in 3 dimensional space.
    
    get the non-zero elements and their indices. 
    then pass the count as the scalar.
    """
    pass


def shape_distance(sc1, sc2, do_fft=False):
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
    
    if do_fft: # gives rotational invariance
        sc1_norm = np.abs(np.fft.fft2(sc1_norm))
        sc2_norm = np.abs(np.fft.fft2(sc2_norm))

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


def save_cloud_as_mat(file_num):
    data_dir    = '/home/ankush/sandbox/bulletsim/src/tests/ravens/recorded/'
    clouds_file = 'clouds_%d.npz'%file_num 
    fname = osp.join(data_dir, clouds_file)

    clouds = np.load(fname)
    tclouds = [clouds[n] for n in clouds.files if n.startswith('target')]
    tcloud  = tclouds[0][:,0:2]

    sclouds = [clouds[n] for n in clouds.files if n.startswith('src')]
    scloud  = sclouds[0][:,0:2]

    sio.savemat('reg_data_%d.mat'%file_num, {'x':scloud, 'y':tcloud})


def test_shape_context(file_num):
    (src, target) = load_clouds(file_num)
    src = src[0]
    
    th = np.pi/6
    from numpy import sin, cos
    R = np.array([[cos(th), -sin(th), 0],
                 [sin(th), cos(th), 0],
                 [0,0,1]])
    target = np.dot(target[0], R.T)
    
    sc_src    = shape_context(src)
    sc_target = shape_context(target)  
    dists    = shape_distance(sc_src, sc_target)
    argmins  = np.argmin(dists, axis=1)

    # plot stuff
    plotter = PlotterInit()
    plot_reqs = []
    plot_reqs.append(gen_mlab_request(mlab.points3d, src[:,0], src[:,1], src[:,2], color=(1,0,0), scale_factor=0.001))
    plot_reqs.append(gen_mlab_request(mlab.points3d, target[:,0], target[:,1], target[:,2], color=(0,0,1), scale_factor=0.001))
    
    plinks = [np.c_[src[i,:], target[ti,:]].T for i,ti in enumerate(argmins)]
    plot_reqs.append(gen_custom_request('lines', lines=plinks, color=(0,1,0), line_width=2, opacity=1))
    
    Ts = pca_frame(src)
    Tt = pca_frame(target)
    plot_reqs.append(gen_custom_request('transform', Ts, size=0.01))
    plot_reqs.append(gen_custom_request('transform', Tt, size=0.01))

    for req in plot_reqs:
        plotter.request(req)

       
def pca_frame(p, tol=1e-6):
    """
    Given an Nx3 matrix of points, it returns
    the frame defined by the principal components
    of the data.
    """
    N,d = p.shape
    p_mean = np.mean(p, axis=0)
    p_centered = p - p_mean
    _,s,v      = np.linalg.svd(p_centered, full_matrices=True)
    R   = v.T

    # make right-handed
    if not np.allclose(np.cross(R[:,0], R[:,1]), R[:,2]):
        R[:,2] *= -1

    # a hack to get consistent transforms
    if s[2] < tol and R[2,2] < 0:
        R[:,1] *= -1
        R[:,2] *= -1

    T = np.eye(d+1)
    T[0:d,0:d] = R
    T[0:d,d]   = p_mean
    return T

    
def test_shape_context2(file_num):
    """
    Exports the shape_context matrix summed along two of the bins
    in matlab format for comparison against the original code.
    """
    (src, target) = load_clouds(file_num)
    src = src[0]
    target = target[0]
    
    sc_src = shape_context(src)
    sc_src = np.sum(np.sum(sc_src, axis=3), axis=1)
    sio.savemat('/home/ankush/Desktop/shape_context/sc_3d_%d.mat'%file_num, {'sc':sc_src})
