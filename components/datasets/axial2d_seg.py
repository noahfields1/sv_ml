import numpy as np
import random
from tqdm import tqdm
import skimage.filters as filters
import sv_ml.modules.dataset as dataset
from sv_ml.modules.io import load_yaml
import sv_ml.modules.vascular_data as sv
import sv_ml.modules.vessel_regression as vessel_regression

from sv_ml.base.dataset import AbstractDataset
EPS=1e-5

def read_T(id):
    meta_data = load_yaml(id)
    X         = np.load(meta_data['X'])
    Y         = np.load(meta_data['Y'])
    Yc        = np.load(meta_data['Yc'])
    return (X,Y,Yc,meta_data)

def radius_balance(X,Y,Yc,meta, r_thresh, Nsample):
    N = X.shape[0]
    radiuses = [m['radius']*m['spacing'] for m in meta]

    i_sm     = [i for i in range(N) if radiuses[i] <= r_thresh]
    i_lg     = [i for i in range(N) if radiuses[i] > r_thresh]

    index    = random.choices(i_sm,k=Nsample)+random.choices(i_lg,k=Nsample)

    X_  = [X[i] for i in index]
    Y_  = [Y[i] for i in index]
    Yc_ = [Yc[i] for i in index]
    m_  = [meta[i] for i in index]

    return X_,Y_,Yc_,m_

def outlier(y):
    if np.sum(y) < 10:
        return True
    return False

def distance_contour(yc,cd, nc):
    c = sv.marchingSquares(yc, iso=0.5)
    c = sv.reorder_contour(c)

    c = (1.0*c-cd)/(cd)
    p = np.mean(c,axis=0)
    c_centered = c

    c_centered = c_centered[:,:2]
    p = p[:2]

    c_reorient = sv.interpContour(c_centered, num_pts=nc)

    c_dist = np.sqrt(np.sum(c_reorient**2,axis=1))

    return c_dist, p

def get_dataset(config, key="TRAIN"):
    """
    setup and return requested dataset
    args:
        config - dict   - must containt FILES_LIST
        key    - string - either TRAIN, VAL, or TEST
    """

    files = open(config['FILE_LIST']).readlines()
    files = [s.replace('\n','') for s in files]

    if key == "TRAIN":
        patterns = config['TRAIN_PATTERNS']
    elif key == "VAL":
        patterns = config['VAL_PATTERNS']
    elif key == "TEST":
        patterns = config['TEST_PATTERNS']
    else:
        raise RuntimeError("Unrecognized data key {}".format(key))

    files = [f for f in files if any([s in f for s in patterns])]

    if "OTHER_PATTERNS" in config:
        p = config['OTHER_PATTERNS']

        files = [f for f in files if any([s in f.lower() for s in p])]

    data = [read_T(s) for s in files]

    meta = [d[3] for d in data]

    X    = np.array([d[0] for d in data])

    N    = X.shape[0]
    cr   = int(X.shape[1]/2)
    cd   = int(config['CROP_DIMS']/2)

    Y    = np.array([d[1] for d in data])
    Yc   = np.array([d[2] for d in data])

    if config['BALANCE_RADIUS'] and key=='TRAIN':
        X,Y,Yc,meta = radius_balance(X,Y,Yc,meta,
        config['R_SMALL'], config['N_SAMPLE'])

    if "AUGMENT" in config and key=='TRAIN':
        aug_x = []
        aug_y = []
        aug_m = []
        for k in range(config['AUGMENT_FACTOR']):
            for i in tqdm(range(len(X))):
                x = X[i]
                y = Y[i]

                x,y = sv.random_rotate((x,y))

                rpix = config['MAX_PIX_SHIFT']
                lim  = int(rpix/np.sqrt(2))
                x_shift = np.random.randint(-lim,lim)
                y_shift = np.random.randint(-lim,lim)

                aug_x.append( x[cr+y_shift-cd:cr+y_shift+cd, cr+x_shift-cd:cr+x_shift+cd] )
                aug_y.append( y[cr+y_shift-cd:cr+y_shift+cd, cr+x_shift-cd:cr+x_shift+cd] )
                aug_m.append( meta[i] )

        X_ = np.array(aug_x)
        Y_ = np.array(aug_y)
        meta = aug_m
    else:
        X_ = np.array(X)[:,cr-cd:cr+cd, cr-cd:cr+cd]
        Y_ = np.array(Y)[:,cr-cd:cr+cd, cr-cd:cr+cd]

    Yc = np.array(Yc)[:,cr-cd:cr+cd, cr-cd:cr+cd]

    Y_ = (1.0*Y_ - np.amin(Y_,axis=(1,2),keepdims=True))/(
        np.amax(Y_,axis=(1,2),keepdims=True) - np.amin(Y_,axis=(1,2),keepdims=True)+EPS)
    print(Yc.shape)
    Yc = (1.0*Yc - np.amin(Yc,axis=(1,2),keepdims=True))/(
        np.amax(Yc,axis=(1,2),keepdims=True) - np.amin(Yc,axis=(1,2),keepdims=True)+EPS)

    return X_, Y_, Yc, meta
