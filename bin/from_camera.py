import sys
sys.path.append('.')
import numpy as np
from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from model.utils import load_image_pixels, decode_netout, correct_yolo_boxes, do_nms, get_boxes, draw_boxes
import cv2

if __name__ == '__main__':
    # load yolov3 model
    model = load_model('modelweights/model.h5')
    # define the expected input shape for the model
    # %% codecell
    input_w, input_h = 416, 416
    # define our new photo


    vid = cv2.VideoCapture(0)

    # for filename in os.listdir('source_images'):
    while True:
        ret, image = vid.read()
        # image = cv2.imread(os.path.join('source_images', filename))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, (256, 128))
        image_w, image_h = image.shape[:2][::-1]
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        oimg = image.copy()
        image = cv2.resize(image, (input_w, input_h))
        image = image.reshape(1, input_w, input_h, 3)

        yhat = model.predict(image/255.)
        anchors = [[116, 90, 156, 198, 373, 326], [
        30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
        class_threshold = 0.6
        boxes = list()
        for i in range(len(yhat)):
            # decode the output of the network
            boxes += decode_netout(yhat[i][0], anchors[i],
                class_threshold, input_h, input_w)
        correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
        # import pdb; pdb.set_trace()
        # do_nms(boxes, 0.5)
        labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
                  "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                  "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                  "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                  "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                  "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
                  "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
                  "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
                  "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                  "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
        v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)
        for i in range(len(v_boxes)):
            print(v_labels[i], v_scores[i])

        for i in range(len(v_boxes)):
            box = v_boxes[i]
            # get coordinates
            y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
            # calculate width and height of the box
            width, height = x2 - x1, y2 - y1
            # create the shape
            cv2.rectangle(oimg, (x1, y1), (x2, y2), color=(0,0,255), thickness=1)
            cv2.putText(oimg, v_labels[i], (x1, y1), fontFace=1, fontScale=1, color=(255,0,0))

            # rect = Rectangle((x1, y1), width, height, fill=False, color='white')
            # # draw the box
            # ax.add_patch(rect)
            # # draw text and score in top left corner
            # label = "%s (%.3f)" % (v_labels[i], v_scores[i])
            # pyplot.text(x1, y1, label, color='white')

        cv2.imshow('frame', oimg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()
