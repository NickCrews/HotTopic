

class Dataset(object):

    def __init__(self, days=None):
        if days is None:
            days = set()
        self.days = days

    
