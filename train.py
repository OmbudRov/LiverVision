from datetime import datetime
import tensorflow
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
from Utilities.GenUtils import CreateDirectory, JoinPaths, SetGPUs, GetGPUsCount
import Utilities.DataGenerator as DataGen
from Utilities.loss import *
from Unet import PrepareModel

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def CreateTrainingFolders():
    CreateDirectory("Checkpoints")
    CreateDirectory("Checkpoints/Tensorboard")

def Train():
    print("Verifying the data...")
    
    CreateTrainingFolders()
    TrainingGenerator = DataGen.DataGenerator(Mode = "Train")
    ValGenerator = DataGen.DataGenerator(Mode = "Val")

    Optimizer = tensorflow.keras.optimizers.Adam(
        learning_rate = 3e-4
    )

    model = PrepareModel()
    model.compile(
        optimizer = Optimizer,
        loss = DiceCoefLoss,
        metrics = [DiceCoef],
    )
    model.summary()

    TensorboardLogDirectory = JoinPaths(
        "Checkpoints/Tensorboard",
        "{}".format(datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
        )
    print("TensorBoard Directory: "+TensorboardLogDirectory+"\n")
    
    CheckpointPath = JoinPaths(
        "Checkpoints",
        "UNet.hdf5"
        )
    print("Weights Path: "+CheckpointPath+"\n")

    CSVLogPath = JoinPaths("Checkpoints", "TrainingLogs_UNet.csv")
    print("Logs Path: "+CSVLogPath+"\n")

    EvaluationMetric = "val_DiceCoef"
    Callbacks = [
        TensorBoard(
            log_dir = TensorboardLogDirectory,
            write_graph = False,
            profile_batch = 0,
            update_freq='batch'
        ),
        EarlyStopping(
            patience = 20,
            verbose = 1
        ),
        ModelCheckpoint(
            CheckpointPath,
            verbose = 1,
            save_weights_only = True,
            save_best_only = True,
            monitor = EvaluationMetric,
            mode = "max"
        ),
        CSVLogger(
            CSVLogPath,
            append = False
        )
    ]

    TrainingSteps = TrainingGenerator.__len__()
    ValidationSteps = ValGenerator.__len__()
    model.fit(
        x = TrainingGenerator,
        steps_per_epoch = TrainingSteps,
        validation_data = ValGenerator,
        validation_steps = ValidationSteps,
        epochs = 10,
        batch_size = 2,
        callbacks = Callbacks,
    )
def main():
    Train()

if __name__ == "__main__":
    main()