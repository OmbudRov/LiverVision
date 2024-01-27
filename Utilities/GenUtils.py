import os 
import tensorflow
import cv2
from glob import glob

def CreateDirectory(Path):
    if not os.path.exists(Path):
        os.makedirs(Path)

def JoinPaths(*Paths):
    return os.path.normpath(os.path.sep.join(path.rstrip(r"\/") for path in Paths))

def SetGPUs(GPU_IDs):
    AllGPUs = tensorflow.config.experimental.list_physical_devices('GPU')
    AllGPUsLength = len(AllGPUs)
    if isinstance(GPU_IDs, int):
        if GPU_IDs == -1:
            GPU_IDs = range(AllGPUsLength)
        else:
            GPU_IDs = min(GPU_IDs, AllGPUsLength)
            GPU_IDs = range(GPU_IDs)
    
    SelectedGPUs = [AllGPUs[GPU_ID] for GPU_ID in GPU_IDs if GPU_ID < AllGPUsLength]

    try:
        tensorflow.config.experimental.set_visible_devices(SelectedGPUs, 'GPU')
    except RuntimeError as E:
        print(E)

def ResizeImage(Img, Height, Width, ResizeMethod = cv2.INTER_CUBIC):
    return cv2.resize(Img, dsize = (Width, Height), interpolation = ResizeMethod)

def GetGPUsCount():
    return len(tensorflow.config.experimental.list_logical_devices('GPU'))

def ImageToMaskName(ImageName : str):
    return ImageName.replace('image', 'mask')

def GetDataPaths(Mode : str):
    ImagesPaths = os.listdir(
        JoinPaths(
            "Data\\"+Mode+"\Images"
        )
    )

    MaskPaths = [
        ImageToMaskName(ImageName) for ImageName in ImagesPath
    ]
    MaskPaths = [
        JoinPaths(
            "Data\\"+Mode+"\Mask",
            MaskName
        ) for MaskName in MaskPaths
    ]

    ImagesPaths = [
        JoinPaths(
            "Data\\"+Mode+"\Images",
            ImageName
        ) for ImageName in ImagesPath
    ]

def ReadImage(ImgPath, ColorMode):
    return cv2.imread(ImgPath, ColorMode)

def PrepareImage(Path : str, NormalizeType : str):
    Image = ReadImage(Path, cv2.IMREAD_COLOR)
    Image = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)

    if NormalizeType == "Normalize":
        Image = Image / 255.0
    
    Image = Image.astype(numpy.float32)

    return Image

def PrepareMask(Path : str):
    Mask = ReadImage(Path, cv2.IMREAD_GRAYSCALE)
    Mask = Mask.astype(numpy.int32)
    return Mask