from sv_ml.components.common import BasePredictor
import sv_ml.modules.vessel_regression as vessel_regression
import skimage.filters as filters
import numpy as np
from sv_ml.modules.io import mkdir, write_json, load_json
from sv_ml.modules.vessel_regression import pred_to_contour
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def log_prediction(yhat,x,c,p,meta,path, config):
    cpred = pred_to_contour(yhat)
    ctrue = pred_to_contour(c)
    scale  = meta['dimensions']*meta['spacing']/2

    new_meta = {}
    for k in meta: new_meta[k] = meta[k]

    new_meta['center']   = p.tolist()
    new_meta['yhat_raw'] = yhat.tolist()
    new_meta['c_raw']    = c.tolist()

    new_meta['yhat_centered_unscaled'] = cpred.tolist()
    new_meta['c_centered_unscaled']    = ctrue.tolist()

    cpred_pos = (cpred+p)*scale
    ctrue_pos = (ctrue+p)*scale

    #edge fitting
    I = filters.gaussian(x)
    E = filters.sobel(I)

    interp = vessel_regression.Interpolant(E, np.array([-scale, -scale]), config['SPACING'])

    R = np.mean(np.sqrt(np.sum(cpred**2,axis=1)))
    new_meta['R_pred'] = R
    Nsteps             = config['N_STEPS']
    alpha              = config['EDGE_RADIUS_RATIO']
    angles = np.arctan2(cpred[:,1],cpred[:,0])
    new_meta['angles'] = angles.tolist()
    
    z = vessel_regression.edge_fit(interp, cpred_pos, angles, alpha, R, Nsteps)

    new_meta['yhat_pos_noedge'] = cpred_pos.tolist()
    new_meta['yhat_pos'] = z.tolist()
    new_meta['c_pos']    = ctrue_pos.tolist()

    new_meta['radius_pixels'] = meta['radius']
    new_meta['radius'] = meta['radius']*meta['spacing']

    name = meta['image']+'.'+meta['path_name']+'.'+str(meta['point'])

    write_json(new_meta, path+'/predictions/{}.json'.format(name))

    plt.figure()
    plt.imshow(E,cmap='gray',extent=[-scale,scale,scale,-scale])
    plt.colorbar()
    plt.scatter(cpred_pos[:,0], cpred_pos[:,1], color='r', label='predicted',s=4)
    plt.scatter(ctrue_pos[:,0], ctrue_pos[:,1], color='y', label='true', s=4)
    plt.scatter(z[:,0], z[:,1], color='pink', label='edge fit', s=4)
    plt.legend()
    plt.savefig(path+'/images/{}_edge.png'.format(name),dpi=200)
    plt.close()

    plt.figure()
    plt.imshow(x,cmap='gray',extent=[-scale,scale,scale,-scale])
    plt.colorbar()
    plt.scatter(cpred_pos[:,0], cpred_pos[:,1], color='r', label='predicted',s=4)
    plt.scatter(ctrue_pos[:,0], ctrue_pos[:,1], color='y', label='true', s=4)
    plt.scatter(z[:,0], z[:,1], color='pink', label='edge fit', s=4)
    plt.legend()
    plt.savefig(path+'/images/{}.png'.format(name),dpi=200)
    plt.close()

class EdgeFittingPredictor(BasePredictor):
    def setup(self):
        res_dir = self.config['RESULTS_DIR']
        name    = self.config['NAME']

        self.root           = os.path.join(res_dir,name)
        self.log_dir        = os.path.join(self.root,'log')
        self.model_dir      = os.path.join(self.root,'model')
        self.val_dir        = os.path.join(self.root,'val')
        self.val_image_dir  = os.path.join(self.root,'val','images')
        self.val_pred_dir   = os.path.join(self.root,'val','predictions')
        self.test_dir       = os.path.join(self.root,'test')
        self.test_image_dir = os.path.join(self.root,'test','images')
        self.test_pred_dir  = os.path.join(self.root,'test','predictions')

        mkdir(self.root)
        mkdir(self.log_dir)
        mkdir(self.model_dir)
        mkdir(self.val_dir)
        mkdir(self.val_image_dir)
        mkdir(self.val_pred_dir)
        mkdir(self.test_dir)
        mkdir(self.test_image_dir)
        mkdir(self.test_pred_dir)

    def predict(self):
        predictions = self.model.predict(self.X)

        path = self.config['RESULTS_DIR']+'/'+self.config['NAME']
        if self.data_key == "VAL":
            path = path+'/val'
        elif self.data_key == "TEST":
            path = path+'/test'

        for i in tqdm(range(predictions.shape[0])):
            x = self.X[i]
            c = self.C[i]
            p = self.points[i]
            meta = self.meta[i]
            yhat = predictions[i]

            log_prediction(yhat,x,c,p,meta,path, self.config)