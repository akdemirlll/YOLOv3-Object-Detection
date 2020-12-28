import sys
sys.path.append('.')
sys.path.append('..')
from model import yolov3 as yv3

if __name__ == '__main__':
    # define the model
    model = yv3.make_yolov3_model()
    # load the model weights
    weight_reader = yv3.WeightReader('modelweights/yolov3.weights')
    # set the model weights into the model
    weight_reader.load_weights(model)
    # save the model to file
    model.save('modelweights/model.h5')
