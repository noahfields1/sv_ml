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
    meta_data['radius_real'] = meta_data['radius']*meta_data['spacing']
    X         = np.load(meta_data['X'])
    Y         = np.load(meta_data['Y'])
    Yc        = np.load(meta_data['Yc'])
    return (X,Y,Yc,meta_data)

def radius_balance(X,Y,meta, r_thresh, Nsample):
    N = X.shape[0]
    radiuses = [m['radius']*m['spacing'] for m in meta]

    i_sm     = [i for i in range(N) if radiuses[i] <= r_thresh]
    i_lg     = [i for i in range(N) if radiuses[i] > r_thresh]

    index    = random.choices(i_sm,k=Nsample)+random.choices(i_lg,k=Nsample)

    X_ = [X[i] for i in index]
    Y_ = [Y[i] for i in index]
    m_ = [meta[i] for i in index]

    return X_,Y_,m_

def outlier(y):
    c = vessel_regression.pred_to_contour(y)
    if np.amax(np.abs(c)) > 1:
        return True
    return False

def distance_contour(yc,cd, nc):
    c = sv.marchingSquares(yc, iso=0.5)
    c = sv.reorder_contour(c)

    c = (1.0*c-cd)/(cd)
    p = np.mean(c,axis=0)
    c_centered = c-p

    c_centered = c_centered[:,:2]
    p[:2] = p[:2]/2+0.5

    c_reorient = sv.interpContour(c_centered, num_pts=nc)

    c_dist = np.sqrt(np.sum(c_reorient**2,axis=1))
    d = np.mean(c_dist)*2
    p[2] = d
    return c_dist/d, p

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

    cr   = int(X.shape[1]/2)
    cc   = int(config['CENTER_DIMS']/2)
    cd   = int(config['CROP_DIMS']/2)

    Yc   = np.array([d[2] for d in data])

    r_thresh = config['R_SMALL']
    radiuses = [m['radius']*m['spacing'] for m in meta]

    if 'SIZE_SPLIT' in config:
        print("splitting size")
        if config.get('SIZE_SPLIT') == 'LARGE':
            indexes = [i for i in range(X.shape[0]) if radiuses[i] > r_thresh]

        if config.get('SIZE_SPLIT') == 'SMALL':
            indexes = [i for i in range(X.shape[0]) if radiuses[i] <= r_thresh]

        X        = np.array([X[i] for i in indexes])
        Yc       = np.array([Yc[i] for i in indexes])
        meta     = [meta[i] for i in indexes]



    N    = X.shape[0]
    X_center = np.zeros((N,config['CENTER_DIMS'],config['CENTER_DIMS']))
    Y_center = np.zeros((N,config['CENTER_DIMS'],config['CENTER_DIMS']))

    print("centering images")
    for i,yc in tqdm(enumerate(Yc)):
        if key == 'TRAIN' and config['CENTER']:
            contour = sv.marchingSquares(yc, iso=0.5)
            contour = sv.reorder_contour(contour)

            cx = int(np.mean(contour[:,0]))
            cy = int(np.mean(contour[:,1]))
            if cx > cc+2*(cr-cc): cx = int(cc+2*(cr-cc))
            if cx < cc: cx=cc

            if cy > cc+2*(cr-cc): cy = int(cc+2*(cr-cc))
            if cy < cc: cy=cc

            X_center[i] = X[i,cy-cc:cy+cc, cx-cc:cx+cc].copy()
            Y_center[i] = Yc[i,cy-cc:cy+cc, cx-cc:cx+cc].copy()
        else:
            X_center[i] = X[i,cr-cc:cr+cc, cr-cc:cr+cc].copy()
            Y_center[i] = Yc[i,cr-cc:cr+cc, cr-cc:cr+cc].copy()

    X  = None
    Yc = None

    if config['BALANCE_RADIUS'] and key=='TRAIN':
        X_center,Y_center,meta = radius_balance(X_center,Y_center,meta,
        config['R_SMALL'], config['N_SAMPLE'])

    if "AUGMENT" in config and key == 'TRAIN':
        aug_x = []
        aug_y = []
        aug_m = []
        for k in range(config['AUGMENT_FACTOR']):
            for i in tqdm(range(len(X_center))):
                x = X_center[i]
                y = Y_center[i]

                x,y = sv.random_rotate((x,y))

                rpix = meta[i]['radius']
                lim  = int(config['AUGMENT_R_SCALE']*rpix/np.sqrt(2))+1
                x_shift = np.random.randint(-lim,lim)
                y_shift = np.random.randint(-lim,lim)

                x_ = x[cc+y_shift-cd:cc+y_shift+cd, cc+x_shift-cd:cc+x_shift+cd]
                if not x_.shape[1] == config['CROP_DIMS'] or not x_.shape[0] == config['CROP_DIMS'] or len(x.shape) < 2:
                    continue

                aug_x.append( x_ )
                aug_y.append( y[cc+y_shift-cd:cc+y_shift+cd, cc+x_shift-cd:cc+x_shift+cd] )
                aug_m.append( meta[i] )

        X_ = np.array(aug_x)
        Y_ = np.array(aug_y)
        meta = aug_m
    else:
        X_ = np.array(X_center)[:,cc-cd:cc+cd, cc-cd:cc+cd]
        Y_ = np.array(Y_center)[:,cc-cd:cc+cd, cc-cd:cc+cd]


    #get contours
    contours = []
    x_final  = []
    m_final  = []
    for i in tqdm(range(X_.shape[0])):
        try:
            T = distance_contour(Y_[i],cd,config['NUM_CONTOUR_POINTS'])
            # if outlier(T[0]) and not "OUTLIER" in config:
            #    print("outlier")
            #    continue

            contours.append(T)
            x_final.append(X_[i])
            m_final.append(meta[i])
        except:
            print("failed")
    #contours = np.array(contours)
    X_ = np.array(x_final)

    meta = m_final

    return X_,contours,meta
