import numpy as np
import scipy.sparse as ss

def dtw_cumm_mat(c_mn):
    """
    Calculates the dynamic-time warping cummulative
    cost matrix, given the cost matrix [mxn]
    """
    print c_mn.shape
    m,n = c_mn.shape
    d = np.empty(c_mn.shape)
    d[0,:] = np.cumsum(c_mn[0,:])
    d[:,0] = np.cumsum(c_mn[:,0])
    for i in xrange(1,m):
        for j in xrange(1,n):
            d[i,j] = c_mn[i,j] + min(d[i-1,j], d[i,j-1], d[i-1, j-1])
    return d

def dtw_path(cmat_mn):
    """
    Returns the dynamic-time warped path p, of length m+n-1.
    The path goes from (0,0) to (m-1,n-1).
    """
    m,n = cmat_mn.shape

    nsteps =m+n-1  
    dtw_path  = ss.dok_matrix(cmat_mn.shape) 
    dtw_path[m-1,n-1] = 1
    
    pm,pn = m-1,n-1
    for _ in xrange(nsteps-2, -1,-1):
        if pm==pn==0:
            return dtw_path
        if pm == 0:
            dtw_path[pm,pn-1] = 1
            pn -= 1
        elif pn == 0:
            dtw_path[pm-1,pn] = 1
            pm -= 1
        else:
            arg_min = np.argmin((cmat_mn[pm-1, pn], cmat_mn[pm, pn-1], cmat_mn[pm-1, pn-1]))
            if arg_min==0:
                dtw_path[pm-1, pn] = 1
                pm -=1
            if arg_min==1:
                dtw_path[pm, pn-1] = 1
                pn -= 1
            elif arg_min==2:
                dtw_path[pm-1, pn-1] = 1
                pn -= 1
                pm -= 1
    return dtw_path
    

if __name__=='__main__':
    cmat = np.ones((4,3))
    cmat[0,:] = 0
    cmat[:,0] = 0
    #cmat = np.eye(3)
    dtw_mat = dtw_cumm_mat(cmat)
    print 'cost mat : \n', cmat
    print 'dtw cummulative mat: \n', dtw_mat
    print 'dtw path : \n', dtw_path(dtw_mat) 