from config import Config
from Results.ResultBuilder import ResultBuilder

if __name__ == '__main__':
    cfg = Config()
    res = ResultBuilder(cfg)
    res.build()
    db = 0

