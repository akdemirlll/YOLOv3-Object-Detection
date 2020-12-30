import sys
import os
sys.path.append('.')
import cv2
from model.utils import decode_netout, correct_yolo_boxes, do_nms, get_boxes, draw_boxes_cv2
from keras.models import load_model
from model.yolov3 import LABELS, ANCHORS

WEIGHTSDIR = os.environ['modelweights']
TESTIMG = os.environ['testimg']

if __name__ == '__main__':
    model = load_model(WEIGHTSDIR)

    input_w, input_h = 416, 416
    image = cv2.imread(TESTIMG)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_w, image_h = image.shape[:2][::-1]
    oimg = image.copy()
    image = cv2.resize(image, (input_w, input_h))
    image = image.reshape(1, input_w, input_h, 3)
    yhat = model.predict(image / 255.)
    class_threshold = 0.6
    boxes = list()
    for i in range(len(yhat)):
        # decode the output of the network
        boxes += decode_netout(yhat[i][0], ANCHORS[i],
                               class_threshold, input_h, input_w)
    correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
    do_nms(boxes, 0.5)
    v_boxes, v_labels, v_scores = get_boxes(boxes, LABELS, class_threshold)
    for i in range(len(v_boxes)):
        print(v_labels[i], v_scores[i])

    draw_boxes_cv2(oimg, v_boxes, v_labels, v_scores)
    print("Displaying detected objects in a new window.")
    print("Press any key to continue.")
    cv2.imshow('frame', oimg)
    cv2.waitKey(0)
