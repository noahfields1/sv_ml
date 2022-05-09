import sv_ml.components.common as common

def get(config):
    if not "EVALUATION" in config:
        raise RuntimeError("EVALUATION key missing from config")

    ev = config['EVALUATION']

    if ev == "base":
        return common.BaseEvaluation(config)
    else:
        raise RuntimeError("Unrecognized evaluation {}".format(ev))