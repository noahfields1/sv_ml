class AbstractDataset(object):
    def __init__(self, raw_data):
        self.raw_data = raw_data
    def configure(self,config, key):
        pass
    def get(self, index):
        pass
    def get_batch(self, batch_size=16):
        pass
    def shape(self):
        pass
