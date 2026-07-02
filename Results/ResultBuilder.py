from Scene import Scene

class ResultBuilder:
    def __init__(self, path):
        self.path = path
        self.dataset = []
        self.df = None