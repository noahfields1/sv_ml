class AbstractEvaluation(object):
    def __init__(self, config):
        self.config = config
        self.setup()
    def setup(self):
        pass
    def evaluate(self, data_key):
        pass
