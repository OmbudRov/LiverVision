import tensorflow
import numpy
import os

from Utilities.GenUtils import JoinPaths, SetGPUs, GetGPUsCount, GetDataPaths

class DataGenerator(tensorflow.keras.utils.Sequence):
    def __init__(Self, Mode : str):
        Self.Mode = Mode
        Self.BatchSize = 2
        numpy.random.seed(73)
        DataPaths = GetDataPaths(Mode)

        Self.ImagesPaths = DataPaths[0]
        Self.MaskPaths = DataPaths[1]

        Self.on_epoch_end()

    def __len__(Self):
        Self.on_epoch_end()
        return int(numpy.floor(
            len(Self.ImagesPaths) / Self.BatchSize
        ))
    
    def on_epoch_end(Self):
        Self.Indexes = numpy.arange(len(Self.ImagesPaths))
        if(Self.Mode == "Image"):
            numpy.random.shuffle(Self.Indexes)
    
    def __getitem__(Self, Index):
        Indexes = Self.Indexes[
            Index * Self.BatchSize : (Index+1) * Self.BatchSize
        ]
        return Self.__data_generation(Indexes)

    def __data_generation(Self, Indexes): 
        BatchImages = numpy.zeros((
            2,
            320,
            320,
            3
        )).astype(numpy.float32)
        BatchMasks = numpy.zeros((
            2,
            320,
            320,
            2
        ))

        for i, Index in enumerate(Indexes):
            ImgPath = Self.ImagesPaths[int(Index)]
            MaskPath = Self.MaskPaths[int(Index)]

            Image = PrepareImage(
                ImgPath,
                "Normalize"
            )
            Mask = PrepareMask(
                MaskPath
            )

            Image, Mask = tensorflow.numpy_function(
                Self.tf_func,
                [Image, ],
                [tensorflow.float32, ]
            )

            #Setting Shape incase if it was lost during tensorflow conversion
            Image.set_shape([
                320,
                320,
                3
            ])
            BatchImages[i] = Image

            Mask = tensorflow.one_hot(
                Mask,
                2,
                dtype = tensorflow.int32
            )
            Mask.set_shape([
                320,
                320,
                2
            ])
            BatchMasks[i] = Mask
            return BatchImages, BatchMasks
    
    @staticmethod
    def tf_func(*args):
        return args