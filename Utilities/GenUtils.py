import os 
from glob import glob

def CreateDirectory(Path):
    if not os.path.exists(Path):
        os.makedirs(Path)

def JoinPaths(*Paths):
    return os.path.normpath(os.path.sep.join(path.rstrip(r"\/") for path in Paths))

def ResizeImage(Img, Height, Width, ResizeMethod = cv2.INTER_CUBIC):
    return cv2.resize(Img, dsize = (Width, Height), interpolation = ResizeMethod)