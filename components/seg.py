import joblib
import numpy as np
import os
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
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

class SegPreProcessor(AbstractPreProcessor):
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

        x_ = x_.reshape(self.config['INPUT_DIMS'])

        return x_.copy()

    def preprocess_label(self,y):
        return y.reshape(self.config['LABEL_SHAPE'])

class SegPostProcessor(AbstractPostProcessor):
    def setup(self):
        self.cd = self.config['CROP_DIMS']

    def __call__(self,y):
        yhat = y.reshape((cd,cd))
        return yhat

def log_prediction(yhat,x,y,meta,path,config):
    scale  = config['CROP_DIMS']*meta['spacing']/2

    new_meta = {}
    for k in meta: new_meta[k] = meta[k]

    new_meta['radius_pixels'] = meta['radius']
    new_meta['radius'] = meta['radius']*meta['spacing']

    name = meta['image']+'.'+meta['path_name']+'.'+str(meta['point'])

    write_json(new_meta, path+'/predictions/{}.json'.format(name))

    S = x.shape
    if len(S) == 1:
        w = int(S[0]**0.5)
        x_ = x.reshape((w,w))
    else:
        x_ = x[:,:,0]

    plt.figure()
    plt.imshow(x_,cmap='gray',extent=[-scale,scale,scale,-scale])
    plt.colorbar()
    plt.savefig(path+'/images/{}.x.png'.format(name),dpi=200)
    plt.close()

    plt.figure()
    plt.imshow(yhat,cmap='gray',extent=[-scale,scale,scale,-scale])
    plt.colorbar()
    plt.savefig(path+'/images/{}.yhat.png'.format(name),dpi=200)
    plt.close()

    plt.figure()
    plt.imshow(y,cmap='gray',extent=[-scale,scale,scale,-scale])
    plt.colorbar()
    plt.savefig(path+'/images/{}.y.png'.format(name),dpi=200)
    plt.close()

class SegPredictor(AbstractPredictor):
    def set_data(self, data, data_key):
        """
        data is a tuple (X,C,points,meta)
        """
        self.X      = data[0]
        self.Y      = data[1]
        self.Yc     = data[2]
        self.meta   = data[3]
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

        predictions = self.model.predict(X)

        path = self.config['RESULTS_DIR']+'/'+self.config['NAME']
        if self.data_key == "VAL":
            path = path+'/val'
        elif self.data_key == "TEST":
            path = path+'/test'

        for i in tqdm(range(predictions.shape[0])):
            x    = self.X[i]
            x_   = X[i]
            y    = self.Y[i]
            meta = self.meta[i]
            yhat = predictions[i]

            self.postprocessor.set_inputs((x,meta))
            yhat = self.postprocessor(yhat)

            log_prediction(yhat,x_,y,meta,path,self.config)

    def load(self):
        self.model.load()

class SegEvaluation(AbstractEvaluation):
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

            if outlier(np.array(d['c_raw'])):
                print("outlier")
                continue

            cp_seg = sv.contourToSeg(cpred, ORIGIN, DIMS, SPACING)
            ct_seg = sv.contourToSeg(ctrue, ORIGIN, DIMS, SPACING)

            o = {}
            o['image'] = d['image']
            o['path_name'] = d['path_name']
            o['point'] = d['point']
            o['model_name'] = self.config['NAME']
            o['HAUSDORFF'] = hd(cp_seg,ct_seg, SPACING)
            o['ASSD'] = assd(cp_seg, ct_seg, SPACING)
            o['dice'] = dc(cp_seg, ct_seg)
            o['radius'] = d['radius']
            results.append(o)

        df = pd.DataFrame(results)
        df_fn = os.path.join(self.out_dir,'{}.csv'.format(data_key))
        df.to_csv(df_fn)