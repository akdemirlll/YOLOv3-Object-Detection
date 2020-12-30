import sys
import os
sys.path.append('.')
sys.path.append('..')
from model import yolov3 as yv3
from model import utils
RAW_WEIGHTS = os.environ['rawweights']
SETUP_WEIGHTS = os.environ['modelweights']

if __name__ == '__main__':
    # define the model
    model = yv3.make_yolov3_model()
    # load the model weights
    weight_reader = utils.WeightReader(RAW_WEIGHTS)
    # set the model weights into the model
    weight_reader.load_weights(model)
    # save the model to file
    model.save(SETUP_WEIGHTS)
