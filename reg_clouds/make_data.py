from mayavi import mlab
import numpy as np
import os.path as osp

def disp_pts(pts1, pts2, color1, color2):
    figure = mlab.gcf()
    mlab.clf()
    figure.scene.disable_render = True

    pts1_glyphs = mlab.points3d(pts1[:,0], pts1[:,1], pts1[:,2], color=color1, resolution=20, scale_factor=0.001)
    pts2_glyphs = mlab.points3d(pts2[:,0], pts2[:,1], pts2[:,2], color=color2, resolution=20, scale_factor=0.001)
    glyph_points1 = pts1_glyphs.glyph.glyph_source.glyph_source.output.points.to_array()
    glyph_points2 = pts2_glyphs.glyph.glyph_source.glyph_source.output.points.to_array()

    dd = 0.001

    outline1 = mlab.outline(pts1_glyphs, line_width=3)
    outline1.outline_mode = 'full'
    p1x, p1y, p1z = pts1[0,:]
    outline1.bounds = (p1x-dd, p1x+dd,
                       p1y-dd, p1y+dd,
                       p1z-dd, p1z+dd)

    pt_id1 = mlab.text(0.8, 0.2, '0 .', width=0.1, color=color1)

    outline2 = mlab.outline(pts2_glyphs, line_width=3)
    outline2.outline_mode = 'full'
    p2x, p2y, p2z = pts2[0,:]
    outline2.bounds = (p2x-dd, p2x+dd,
                       p2y-dd, p2y+dd,
                       p2z-dd, p2z+dd)  
    pt_id2 = mlab.text(0.8, 0.01, '0 .', width=0.1, color=color2)
    
    figure.scene.disable_render = False


    def picker_callback(picker):
        """ Picker callback: this get called when on pick events.
        """
        if picker.actor in pts1_glyphs.actor.actors:
            point_id = picker.point_id/glyph_points1.shape[0]
            if point_id != -1:
                ### show the point id
                pt_id1.text = '%d .'%point_id
                #mlab.title('%d'%point_id)
                x, y, z = pts1[point_id,:]
                outline1.bounds = (x-dd, x+dd,
                                   y-dd, y+dd,
                                   z-dd, z+dd)
        elif picker.actor in pts2_glyphs.actor.actors:
            point_id = picker.point_id/glyph_points2.shape[0]
            if point_id != -1:
                ### show the point id
                pt_id2.text = '%d .'%point_id
                x, y, z = pts2[point_id,:]
                outline2.bounds = (x-dd, x+dd,
                                   y-dd, y+dd,
                                   z-dd, z+dd)


    picker = figure.on_mouse_pick(picker_callback)
    picker.tolerance = dd/2.
    mlab.show()
    
def load_clouds(file_num=109):
    data_dir    = '/home/ankush/sandbox/bulletsim/src/tests/ravens/recorded/'
    clouds_file = 'clouds_%d.npz'%file_num 
    fname = osp.join(data_dir, clouds_file)

    clouds = np.load(fname)
    tclouds = [clouds[n] for n in clouds.files if n.startswith('target')]
    sclouds = [clouds[n] for n in clouds.files if n.startswith('src')]
    return (sclouds, tclouds)


def open_scene(file_num):
    (sc, tc) = load_clouds(file_num)
    src = sc[0]
    target = tc[0]
    disp_pts(src, target, color1=(1,0,0), color2=(0,1,0))
    

def open_scene2(fil1, fil2):
    data_dir    = '/home/ankush/sandbox/bulletsim/src/tests/ravens/recorded/scene_pts'
    f1name = osp.join(data_dir, fil1)
    f2name = osp.join(data_dir, fil2)
    sc  = np.loadtxt(f1name)
    tc  = np.loadtxt(f2name)
    disp_pts(sc, tc, color1=(1,0,0), color2=(0,1,0))


def save_npz_as_txt(file_num):
    sc, tc = load_clouds(file_num)
    sc = sc[0]
    tc = tc[0]
    
    save_dir    = '/home/ankush/sandbox/bulletsim/src/tests/ravens/recorded/scene_pts/rope-tests'
    src_fname   = osp.join(save_dir, 'rope-%d-src.txt'%file_num)
    targ_fname   = osp.join(save_dir, 'rope-%d-targ.txt'%file_num)
    
    np.savetxt(src_fname, sc)
    np.savetxt(targ_fname, tc)


if __name__=='__main__':
    open_scene(22)