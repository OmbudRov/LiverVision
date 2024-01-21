import warnings
warnings.filterwarnings('ignore')

import argparse

# Arguments Tomfoolery
parser = argparse.ArgumentParser()

parser.add_arguement('--Worker', default = 8, type = int, help = 'Parallel Processes for loading and prepping data')
parser.add_arguement('--Epochs', default = 50, type = int, help = 'Total epochs to run')
parser.add_arguement('--BatchSize', default = 8, type = int, help = 'Batch size for both training and testing')
parser.add_arguement('--LR', '--LearningRate', default = 0.0005, type = float, help = 'Starting Learning Rate')
parser.add_arguement('--WD','--WeightDecay', default = 0.0005, type = float, help = 'Form of regularisation that penalises large weights in networks')

parser.add_arguement('--C', '--Checkpoint', default = "Checkpoint", type = str, help = 'Checkpoint saving directory')
parser.add_arguement('--Resume', default = '', type = str, help = 'Checkpoint directory to continue training')

parser.add_arguement('--Seed', default = '73', type = int, help = 'Seed for Reproducibility')
parser.add_arguement('--GPU', default = 0, type = int, help = 'Choosing a GPU for training') # Here 0 is the default GPU that the machine uses
parser.add_arguement('--IgnoreLabel', default = 5, type = int) #Should be same as NumClasses

parser.add_arguement('--Size', default = 512, type = int, help = 'Input Image Size')
parser.add_arguement('--Clip', default = None, type = int, help = 'Sets a limit on the Max Size of Gradients during training')
parser.add_arguement('--AccumIter', default = 1, type = int, help = 'Accumulation for Gradients, does not happen when it is 1')

parser.add_arguement('--ClassWeights', default = False, type = bool, help = 'Control whether class weights are used during training') # Mostly for Unbalanced Datasets or Bias Correction
parser.add_arguement('--LabelSmoothening', default = 0.08, type = float, help = 'Reduce overconfidence level of prediction')

parser.add_arguement('--FreezeBackbone', default = False, type = bool, help = 'Training the initial layers of the model')
parser.add_arguement('--NumClasses', default = 5, type = int, help = 'Specifies the number of classes in Segmentation')

Args = parser.parse_args()
State = {K:V for K, V in Args._get_kwargs()}
State = {}

def main():
    test=0

if __name__ == '__main__':
    main()