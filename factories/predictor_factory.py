import sv_ml.components.common as common
import sv_ml.components.edge as edge
from sv_ml.components.seg import SegPredictor

def get(config):
    if not "PREDICTOR" in config:
        raise RuntimeError("PREDICTOR key missing in config")

    predictor = config['PREDICTOR']

    if predictor == "base":
        return common.BasePredictor(config)
    elif predictor == "edge_fit":
        return edge.EdgeFittingPredictor(config)
    elif predictor == 'SEG':
        return SegPredictor(config)
    else:
        raise RuntimeError("Unrecognized predictor {}".format(predictor))