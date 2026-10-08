import json

from config import Config
from Results.ResultBuilder import ResultBuilder

if __name__ == '__main__':
    chosen = json.load(open('selected_scenes.json'))
    cfg = Config(selected_scenes=tuple(chosen), output_dir = r'E:/scenes/val_scenes/output_10scenes', visualize=True)
    # cfg = Config()
    res = ResultBuilder(cfg)
    res.build()
    db = 0

