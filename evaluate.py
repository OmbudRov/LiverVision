import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow

import Utilities.DataGenerator as DataGen
from Utilities.GenUtils import JoinPaths
from Unet import PrepareModel
from Utilities.loss import *

def Evaluate():
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

    CheckpointPath = JoinPaths(
        "Checkpoints",
        "UNet.hdf5"
    )

    assert os.path.exists(CheckpointPath), \
        f"Model Weight's File does not exist at \n {CheckpointPath}"

    model.load_weights(CheckpointPath, by_name = True, skip_mismatch = True)
    model.summary()

    EvaluationMetric = "DiceCoef"

    Result = model.evaluate(
        x = ValGenerator,
        batch_size = 2,
        return_dict = True
    )
    return Result, EvaluationMetric

def main():
    Result, EvaluationMetric = Evaluate()
    with open("Checkpoints/UNet_Evaluation.txt", "w") as f:
        print(Result)    
        print(f"Validation Dice Coefficient: {Result[EvaluationMetric]}")
        
        f.write(str(Result) + "\n")
        f.write(f"Validation Dice Coefficient: {Result[EvaluationMetric]}\n")
if __name__ == "__main__":
    main()    