import sv_ml.components.common as common

def get(config):
    if not "TRAINER" in config:
        raise RuntimeError("TRAINER key missing in config")

    train = config['TRAINER']

    if train == "base":
        return common.BaseTrainer(config)
    else:
        raise RuntimeError("Unrecognized trainer {}".format(train))


