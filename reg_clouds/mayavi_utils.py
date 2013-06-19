import numpy as np
from mayavi import mlab



def plot_lines(lines, color=(1,1,1), line_width=1, opacity=0.4):
    """
    input  :
    
      - lines :  a LIST of m matrices of shape n_ix3
                 each matrix is interpreted as one line
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