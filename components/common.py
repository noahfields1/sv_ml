#import joblib
import numpy as np
import os
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor

import matplotlib
# #matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
from medpy.metric.binary import hd, assd, dc
import skimage.filters as filters

import sv_ml.modules.vascular_data as sv
from sv_ml.modules.io import mkdir, write_json, load_json
from sv_ml.modules.vessel_regression import pred_to_contour
import sv_ml.modules.vessel_regression as vessel_regression

from sv_ml.base.train import AbstractTrainer
from sv_ml.base.predict import AbstractPredictor
from sv_ml.base.evaluation import AbstractEvaluation
from sv_ml.base.preprocessor import AbstractPreProcessor
from sv_ml.base.postprocessor import AbstractPostProcessor
EPS = 1e-5

def outlier(c):
    if np.amax(np.abs(c)) > 1:
        return True
    return False

class BasePreProcessor(AbstractPreProcessor):
    def __call__(self, x):
        if not 'IMAGE_TYPE' in self.config:
            mu  = 1.0*np.mean(x)
            sig = 1.0*np.std(x)+EPS
            x_   = (x-mu)/sig
        else:
            if self.config['IMAGE_TYPE'] == 'EDGE':
                x_ = filters.sobel(x)
                ma = np.amax(x_)
                mi = np.amin(x_)
                x_ = (x_-mi)/(ma-mi+EPS)

            if self.config['IMAGE_TYPE'] == 'HESSIAN':
                x_ = filters.gaussian(x, sigma=self.config['BLUR_SIGMA'])
                x_ = filters.sobel(x_)
                x_ = filters.sobel(x_)
                mu  = 1.0*np.mean(x_)
                sig = 1.0*np.std(x_)+EPS
                x_   = (x_-mu)/sig
                c = self.config['CLIP_VAL']
                x_[x_>c] = c
                x_[x_<-c] = -c

            if self.config['IMAGE_TYPE'] == 'CLIP':
                mu  = 1.0*np.mean(x)
                sig = 1.0*np.std(x)+EPS
                x_   = (x-mu)/sig
                c = self.config['CLIP_VAL']
                x_[x_>c] = c
                x_[x_<-c] = -c

            if self.config['IMAGE_TYPE'] == 'BLUR':
                x_ = filters.gaussian(x, sigma=self.config['BLUR_SIGMA'])
                mu  = 1.0*np.mean(x_)
                sig = 1.0*np.std(x_)+EPS
                x_   = (x_-mu)/sig

            if self.config['IMAGE_TYPE'] == 'FRANGI':
                x_ = filters.frangi(x)
                mu  = 1.0*np.mean(x_)
                sig = 1.0*np.std(x_)+EPS
                x_   = (x_-mu)/sig

            if self.config['IMAGE_TYPE'] == 'GABOR':
                x_ = filters.gabor(x, frequency = 0.3)
                print("Gabor Shape: " + str(x_.shape))
                mu  = 1.0*np.mean(x_)
                sig = 1.0*np.std(x_)+EPS
                x_   = (x_-mu)/sig

        #Gabriel's Code
        #x_ = x_.reshape(self.config['INPUT_DIMS'])
        #return x_.copy()
        #Noah Added Code
        cropx,cropy,cropz = self.config['INPUT_DIMS']
        x,y= x_.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)
        x_new = x_[starty:starty+cropy,startx:startx+cropx]
        #x_new = x_.reshape((x[0],x[1],1))
        print(x_new.shape)
        return x_new.copy()

class BasePostProcessor(AbstractPostProcessor):
    def setup(self):
        self.scale = self.config['CROP_DIMS']*self.config['SPACING']/2

    def __call__(self,y):
        c = pred_to_contour(y)
        return c*self.scale

class PointPostProcessor(AbstractPostProcessor):
    def setup(self):
        self.scale = self.config['CROP_DIMS']*self.config['SPACING']/2

    def __call__(self,y):
        c = vessel_regression.point_pred_to_contour(y)
        return c*self.scale

class EdgePostProcessor(AbstractPostProcessor):
    def setup(self):
        self.scale = self.config['CROP_DIMS']*self.config['SPACING']/2

    def set_inputs(self, T):
        x = T[0]
        I = filters.gaussian(x)
        E = filters.sobel(I)

        self.interp = vessel_regression.Interpolant(E,
            np.array([-self.scale, -self.scale]), self.config['SPACING'])

    def __call__(self,y):
        cpred  = pred_to_contour(y)*self.scale
        R      = np.mean(np.sqrt(np.sum(cpred**2,axis=1)))
        Nsteps = self.config['N_STEPS']
        alpha  = self.config['EDGE_RADIUS_RATIO']
        angles = np.arctan2(cpred[:,1],cpred[:,0])

        z = vessel_regression.edge_fit(self.interp, cpred, angles, alpha, R, Nsteps)

        return z

def log_prediction(yhat,x,c,meta,path,config):
    ctrue = c
    scale  = config['CROP_DIMS']*meta['spacing']/2

    new_meta = {}
    for k in meta: new_meta[k] = meta[k]

    new_meta['c_raw']    = c.tolist()

    new_meta['c_centered_unscaled']    = ctrue.tolist()

    ctrue_pos = ctrue

    new_meta['yhat_pos'] = yhat.tolist()
    new_meta['c_pos']    = ctrue_pos.tolist()

    new_meta['radius_pixels'] = meta['radius']
    new_meta['radius'] = meta['radius']*meta['spacing']

    name = meta['image']+'.'+meta['path_name']+'.'+str(meta['point'])

    write_json(new_meta, path+'/predictions/{}.json'.format(name))

    S = x.shape
    if len(S) == 1:
        w = int(S[0]**0.5)
        x_ = x.reshape((w,w))
    else:
        #print(x.shape, type(x))
        x_ = x[:,:,0]
        x_ = x
    plt.figure()
    plt.imshow(x_,cmap='gray',extent=[-scale,scale,scale,-scale])
    plt.colorbar()
    plt.scatter(yhat[:,0], yhat[:,1], color='r', label='predicted',s=4)
    plt.scatter(ctrue_pos[:,0], ctrue_pos[:,1], color='y', label='true', s=4)
    plt.legend()
    plt.savefig(path+'/images/{}.png'.format(name),dpi=200)
    plt.close()

class BaseTrainer(AbstractTrainer):
    def setup_directories(self):
        mkdir(self.root)
        mkdir(self.log_dir)
        mkdir(self.model_dir)
        mkdir(self.val_dir)
        mkdir(self.val_image_dir)
        mkdir(self.val_pred_dir)
        mkdir(self.test_dir)
        mkdir(self.test_image_dir)
        mkdir(self.test_pred_dir)

    def set_data(self, data, data_key):
        """
        data is a tuple (X,C,points,meta)
        """
        self.X      = data[0]
        self.C      = data[1]
        self.meta   = data[2]
        self.data_key = data_key

    def set_preprocessor(self,preprocessor):
        self.preprocessor = preprocessor

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

        self.preprocessor = None

    def train(self):
        X = self.X

        if not self.preprocessor == None:
            X = np.array([self.preprocessor(x) for x in self.X])
            C = np.array([self.preprocessor.preprocess_label(c) for c in self.C])
        #Noah's Code###########
        X = X.reshape((X.shape[0],X.shape[1], X.shape[2], 1))
        ###############
        self.model.train(X, C)

    def save(self):
        self.model.save()
    def load(self):
        self.model.load()

class BasePredictor(AbstractPredictor):
    def set_data(self, data, data_key):
        """
        data is a tuple (X,C,points,meta)
        """
        self.X      = data[0]
        self.C      = data[1]
        self.meta   = data[2]
        self.data_key = data_key
        self.preprocessor = None

    def set_preprocessor(self, preprocessor):
        self.preprocessor = preprocessor

    def set_postprocessor(self,postprocessor):
        self.postprocessor = postprocessor

    def predict(self):
        X = self.X.copy()
        
        if not self.preprocessor == None:
            X = np.array([self.preprocessor(x) for x in X])
        X = X.reshape((X.shape[0],X.shape[1],X.shape[2],1))
        predictions = self.model.predict(X)
        path = self.config['RESULTS_DIR']+'/'+self.config['NAME']
        if self.data_key == "VAL":
            path = path+'/val'
        elif self.data_key == "TEST":
            path = path+'/test'

        for i in tqdm(range(predictions.shape[0])):
            x = self.X[i]
            x_ = X[i]
            c = self.C[i]
            
            meta = self.meta[i]
            yhat = predictions[i]
            self.postprocessor.set_inputs((x,meta))
            yhat = self.postprocessor(yhat)
            c    = self.postprocessor(c)
            log_prediction(yhat,x_,c,meta,path,self.config)

    def load(self):
        self.model.load()

class BaseEvaluation(AbstractEvaluation):
    def setup(self):
        self.results_dir = self.config['RESULTS_DIR']
    def evaluate(self, data_key):
        
        name = self.config['NAME']
        self.out_dir    = os.path.join(self.results_dir, name,data_key.lower())
        self.pred_dir   = os.path.join(self.out_dir, 'predictions')

        if not os.path.isdir(self.out_dir):
            raise RuntimeError("path doesnt exist {}".format(self.out_dir))

        if not os.path.isdir(self.pred_dir):
            raise RuntimeError("path doesnt exist {}".format(self.pred_dir))

        pred_files = os.listdir(self.pred_dir)
        pred_files = [os.path.join(self.pred_dir,f) for f in pred_files]

        preds = [load_json(f) for f in pred_files]

        results = []

        SPACING = [self.config['SPACING']]*2
        DIMS    = [self.config['CROP_DIMS']]*2
        ORIGIN  = [0,0]

        for i,d in tqdm(enumerate(preds)):
            cpred = np.array(d['yhat_pos'])
            ctrue = np.array(d['c_pos'])

            if outlier(np.array(d['c_raw'])) and not "OUTLIER" in self.config:
                print("outlier")
                continue

            cp_seg = sv.contourToSeg(cpred, ORIGIN, DIMS, SPACING)
            ct_seg = sv.contourToSeg(ctrue, ORIGIN, DIMS, SPACING)
            print(np.sum(ct_seg))
            o = {}
            o['image'] = d['image']
            o['path_name'] = d['path_name']
            o['point'] = d['point']
            o['model_name'] = self.config['NAME']
            #o['HAUSDORFF'] = hd(cp_seg,ct_seg, SPACING)
            #o['ASSD'] = assd(cp_seg, ct_seg, SPACING)
            o['dice'] = dc(cp_seg, ct_seg)
            o['radius'] = d['radius']
            results.append(o)
        
        df = pd.DataFrame(results)
        df_fn = os.path.join(self.out_dir,'{}.csv'.format(data_key))
        df.to_csv(df_fn)
