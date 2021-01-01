import sys
sys.path.append('.')
import cv2
from model.utils import decode_netout, correct_yolo_boxes, do_nms, get_boxes, draw_boxes_cv2, timit
from keras.models import load_model, Model
from model.yolov3 import LABELS, ANCHORS


if __name__ == '__main__':
    # check args
    no_nms = "--no-nms" in sys.argv
    shallow = "--shallow" in sys.argv
    profile = "--profile" in sys.argv

    if no_nms:
        print("WARNING: Non-max suppression is disabled.\n")

    if shallow:
        print("WARNING: Using shallow predictions. This may impact detection performance.\n")

    # load yolov3 model
    model = load_model('modelweights/model.h5')
    model.summary()

    if shallow:
        model = Model(inputs=model.inputs, outputs=model.get_layer('conv_81').get_output_at(0))

    if profile:
        # allow profiling
        decode_netout = timit(decode_netout)
        do_nms = timit(do_nms)
        model.predict = timit(model.predict)
        draw_boxes_cv2 = timit(draw_boxes_cv2)

    # define the expected input shape for the model
    input_w, input_h = 416, 416
    # define our new photo
    vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
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

        yhat = model.predict(image / 255.)

        if shallow:
            yhat = [yhat]
        # import pdb; pdb.set_trace()
        # print(yhat)

        class_threshold = 0.6
        boxes = list()
        for i in range(len(yhat)):
            # decode the output of the network
            boxes += decode_netout(yhat[i][0], ANCHORS[i],
                                   class_threshold, input_h, input_w)
        correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
        # import pdb; pdb.set_trace()
        if not no_nms:
            do_nms(boxes, 0.5)

        v_boxes, v_labels, v_scores = get_boxes(boxes, LABELS, class_threshold)
        for i in range(len(v_boxes)):
            print(v_labels[i], v_scores[i])

        draw_boxes_cv2(oimg, v_boxes, v_labels, v_scores)

        cv2.imshow('frame', oimg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()
