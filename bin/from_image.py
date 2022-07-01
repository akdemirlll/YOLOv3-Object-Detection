import sys
sys.path.append('.')
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("--no-nms", nargs="?", const=True, default=False, dest="no_nms")
parser.add_argument("--shallow", nargs="?", const=True, default=False, dest="shallow")
parser.add_argument("--profile", nargs="?", const=True, default=False, dest="profile")
parser.add_argument("--image", nargs="+", dest="image_paths", required=True)
parser.add_argument("--outdir", dest="out_path", required=False, default=".")

if __name__ == '__main__':
    # check args
    args = parser.parse_args()
    no_nms = args.no_nms
    shallow = args.shallow
    profile = args.profile
    
    import cv2
    from model.utils import decode_netout, correct_yolo_boxes, do_nms, get_boxes, draw_boxes_cv2, timit
    from keras.models import load_model, Model
    from model.yolov3 import LABELS, ANCHORS

    if no_nms:
        print("WARNING: Non-max suppression is disabled.\n")

    if shallow:
        print("WARNING: Using shallow predictions. This may impact detection performance.\n")

    # load yolov3 model
    model = load_model('modelweights/model.h5')

    if shallow:
        model = Model(inputs=model.inputs, outputs=model.get_layer('conv_81').get_output_at(0))


    model.summary()

    if profile:
        # allow profiling
        decode_netout = timit(decode_netout)
        do_nms = timit(do_nms)
        model.predict = timit(model.predict)
        draw_boxes_cv2 = timit(draw_boxes_cv2)

    # define the expected input shape for the model
    input_w, input_h = 416, 416
    # define our new photo
    for filename in args.image_paths:
        image = cv2.imread(filename)
        oimg = image.copy()
        image_w, image_h = image.shape[:2][::-1]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

        draw_boxes_cv2(oimg, v_boxes, v_labels, v_scores, rectangle_kwargs=dict(thickness=3), text_kwargs=dict(fontScale=1, fontFace=1))

        outname = os.path.join(args.out_path, "boxed-" + os.path.split(filename)[-1])
        cv2.imwrite(outname, oimg)
