import os
import sys
from glob import glob
from pathlib import Path
from tqdm import tqdm
import numpy
import cv2
import nibabel as nib
import matplotlib.pyplot as plt

from Utilities.GenUtils import CreateDirectory, JoinPaths, ResizeImage

def ReadNii(filepath):
    CTScan = nib.load(filepath).get_fdata()
    CTScan = numpy.rot90(numpy.array(CTScan))
    return CTScan

def CropCenter(Img, CropH, CropW):
    Height, Width = Img.shape[:2]
    StartH = Height // 2 - (CropH // 2)
    StartW = Width // 2 - (CropW // 2)
    return Img[StartH:StartH + CropH, StartW:StartW + CropW, :]

def LinearScale(Img):
    Img = (Img - Img.min(axis = (0, 1))) / (Img.max(axis = (0, 1)) - Img.min(axis = (0, 1)))
    return Img*255

def ClipScan(Img, MinValue, MaxValue):
    return numpy.clip(Img, MinValue, MaxValue)

def ResizeScan(Scan, NewHeight, NewWidth, ScanType):
    ScanShape = Scan.shape
    ResizedScan = numpy.zeros((NewHeight, NewWidth, ScanShape[2]), dtype=Scan.dtype)
    ResizeMethod = cv2.INTER_CUBIC if ScanType == "image" else cv2.INTER_NEAREST
    for Start in range(0, ScanShape[2], ScanShape[1]):
        End = Start + ScanShape[1]
        if End >= ScanShape[2]: End = ScanShape[2]
        for i in range(ScanShape[2]):
            ResizedScan[:, :, i] = ResizeImage(
                Scan[:, :, i],
                NewHeight, NewWidth,
                ResizeMethod
            )
    return ResizedScan

def SaveImages(Scan, save_path, ImgIndex):
    ScanShape = Scan.shape
    for Index in range(ScanShape[-1]):
        BeforeIndex = max(Index - 1, 0)
        AfterIndex = min(Index + 1, ScanShape[-1] - 1)

        NewImgPath  = os.path.join(save_path, f"image_{ImgIndex}_{Index}.png")
        NewImage = Scan[:, :, [BeforeIndex, Index, AfterIndex]]
        NewImage = cv2.cvtColor(NewImage, cv2.COLOR_RGB2BGR)  # RGB to BGR
        cv2.imwrite(NewImgPath, NewImage)  # save the images as .png

def SaveMask(Scan, SavePath, MaskIndex):
    for Index in range(Scan.shape[-1]):
        NewMaskPath = os.path.join(SavePath, f"mask_{MaskIndex}_{Index}.png")
        plt.imshow(Scan[:, :, Index], cmap = 'gray', aspect='equal')
        plt.axis('off')
        plt.tight_layout(pad = 0)
        plt.savefig(NewMaskPath, bbox_inches = 'tight', pad_inches = 0)
        plt.close()
        ScannedImage = cv2.imread(NewMaskPath)
        ResizedImage = cv2.resize(ScannedImage, (320, 320))
        cv2.imwrite(NewMaskPath, ResizedImage)
        
def ExtractImage(ImagePath, SavePath, ScanType = "image", ):
    _, Index = str(Path(ImagePath).stem).split("-")
    Scan = ReadNii(ImagePath)
    Scan = ResizeScan(
        Scan,
        320,
        320,
        ScanType
    )
    if ScanType == "image":
        Scan = ClipScan(
            Scan,
            -200,
            250
        )
        Scan = LinearScale(Scan)
        Scan = numpy.uint8(Scan)
        SaveImages(Scan, SavePath, Index)
    else:
        SaveMask(Scan, SavePath, Index)

def ExtractImages(ImagesPath, SavePath, ScanType = "image", ):
    for ImagePath in tqdm(ImagesPath):
        ExtractImage(ImagePath, SavePath, ScanType)

def ProcessLiTSData():
    TrainImagesNames = glob(
        JoinPaths(

            "Data\Training Batch 2",
            "volume-*.nii"
            )
    )
    TrainMaskNames = glob(
        JoinPaths(
            "Data\Training Batch 2",
            "segmentation-*.nii"
        )
    )

    assert len(TrainImagesNames) == len(TrainMaskNames), \
        "Train volumes and segmentations are not same in length"

    ValImagesNames = glob(
        JoinPaths(
            "Data\Training Batch 1",
            "volume-*.nii"
        )
    )
    ValMaskNames = glob(
        JoinPaths(
            "Data\Training Batch 1",
            "segmentation-*.nii"
        )
    )
    assert len(ValImagesNames) == len(ValMaskNames), \
        "Validation volumes and segmentations are not same in length"
    
    TrainImagesNames = sorted(TrainImagesNames)
    TrainMaskNames = sorted(TrainMaskNames)
    ValImagesNames = sorted(ValImagesNames)
    ValMaskNames = sorted(ValMaskNames)

    TrainImagesPath = JoinPaths(
        "Data\Train\Images"
    )
    TrainMaskPath = JoinPaths(
        "Data\Train\Mask"
    )
    ValImagesPath = JoinPaths(
        "Data\Val\Images"
    )
    ValMaskPath = JoinPaths(
        "Data\Val\Mask"
    )

    CreateDirectory(TrainImagesPath)
    CreateDirectory(TrainMaskPath)
    CreateDirectory(ValImagesPath)
    CreateDirectory(ValMaskPath)

    print("\nExtracting Train Images")
    ExtractImages(TrainImagesNames, TrainImagesPath, ScanType = "image")

    print("\nExtracting Train Mask")
    ExtractImages(TrainMaskNames, TrainMaskPath, ScanType = "mask")

    print("\n Extracting Val Images")
    ExtractImages(ValImagesNames, ValImagesPath, ScanType = "image")

    print("\n Extracting Val Mask")
    ExtractImages(ValMaskNames, ValMaskPath, ScanType = "mask")

if __name__ == '__main__':
    ProcessLiTSData()