import os

from Results.ResultBuilder import ResultBuilder

if __name__ == '__main__':
    path = r"D:\Programiki do nauki i inne\Szkolne\Studia\Projekt inzynierski\Logi NuScenes"
    res = ResultBuilder(path, vis=True)
    res.build()
    res.calculate_metrics(output_dir=os.path.join(res.path, 'output'))
    db = 0

