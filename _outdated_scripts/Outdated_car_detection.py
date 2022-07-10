"""
OUTDATED
Yolo v3 implementation ssed to detect Cars or Trucks in an image.
This Script is depricated and replaced byCar_DetectionV2 which is much faster and uses a model from tensorflow hub.
Used Code from: https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/
Saved the model via a Notebook and then just loaded it with:
#yolo_model = load_model('model.h5', compile=False)
"""
#Maybe Speed up through Python Multiprocessing? https://www.youtube.com/watch?v=Z_uPIUbGCkA

import tensorflow as tf
#import numpy as np
#import matplotlib.pyplot as plt
import tensorflow_hub as hub
#import tensorflow_datasets as tfds
#import cv2 as cv
import time

#import argparse
import os
import numpy as np
#from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
#from keras.layers.merge import add, concatenate
#from keras.models import Model
#import struct
#import cv2
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import expand_dims
#from matplotlib.patches import Rectangle

#print("Version: ", tf.__version__)
#print("Eager mode: ", tf.executing_eagerly())
#print("Hub version: ", hub.__version__)
#print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score
def _sigmoid(x):
	return 1. / (1. + np.exp(-x))
def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5

    boxes = []

    netout[..., :2] = _sigmoid(netout[..., :2])
    netout[..., 4:] = _sigmoid(netout[..., 4:])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h * grid_w):
        row = i / grid_w
        col = i % grid_w

        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            # objectness = netout[..., :4]

            if (objectness.all() <= obj_thresh): continue

            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]

            x = (col + x) / grid_w  # center position, unit: image width
            y = (row + y) / grid_h  # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w  # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h  # unit: image height

            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]

            box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes)
            # box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, None, classes)

            boxes.append(box)

    return boxes
def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    if (float(net_w) / image_w) < (float(net_h) / image_h):
        new_w = net_w
        new_h = (image_h * net_w) / image_w
    else:
        new_h = net_w
        new_w = (image_w * net_h) / image_h

    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
        y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h

        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)
def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3
def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union
def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return

    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0: continue

            for j in range(i + 1, len(sorted_indices)):
                index_j = sorted_indices[j]

                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0
def get_boxes(boxes, labels, thresh):
	v_boxes, v_labels, v_scores = list(), list(), list()
	# enumerate all boxes
	for box in boxes:
		# enumerate all possible labels
		for i in range(len(labels)):
			# check if the threshold for this label is high enough
			if box.classes[i] > thresh:
				v_boxes.append(box)
				v_labels.append(labels[i])
				v_scores.append(box.classes[i]*100)
				# don't break, many labels may trigger for one box
	return v_boxes, v_labels, v_scores

# load and prepare an image
def load_image_pixels(filename, shape):
    # load the image to get its shape
    image = load_img(filename)
    width, height = image.size
    # load the image with the required size
    image = load_img(filename, target_size=shape)
    # convert to numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image, width, height

print("Helloo")

def car_is_in_image(image, model):
    # load yolov3 model
    t0 = time.perf_counter()

    # define the expected input shape for the model
    input_w, input_h = 416, 416
    # define our new photo
    ## image = 'trucktest2.jpg'
    # load and prepare image
    image, image_w, image_h = load_image_pixels(image, (input_w, input_h))
    t1 = time.perf_counter()
    #print(f"Loading: {t1 - t0} sec")

    t2 = time.perf_counter()
    # make prediction
    yhat = model.predict(image)
    t3 = time.perf_counter()
    #print(f"Predicting: {t3 - t2} sec")

    # summarize the shape of the list of arrays
    #print([a.shape for a in yhat])

    t4 = time.perf_counter()
    # define the anchors
    anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
    # define the probability threshold for detected objects
    class_threshold = 0.70
    boxes = list()
    for i in range(len(yhat)):
        # print(yhat[1][0])
        # decode the output of the network
        boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)

    # correct the sizes of the bounding boxes for the shape of the image
    correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
    t5 = time.perf_counter()
    #print(f"correct box sizes: {t5 - t4} sec")

    t6 = time.perf_counter()
    # suppress non-maximal boxes
    #do_nms(boxes, 0.5)


    # define the labels
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

    # get the details of the detected objects
    v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)
    t7 = time.perf_counter()
    #print(f"NMS and labels: {t7 - t6} sec")

    t8 = time.perf_counter()
    # summarize what we found
    #for i in range(len(v_boxes)):
    #    print(v_labels[i], v_scores[i])
    #    box = v_boxes[i]
    #    y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
    #    print(f"1: {x1}|{y1}, 2: {x2}|{y2}")
    #    print("______________________")

    # TODO use this only use one box..x and if there is one good
    # Check if the list of v_scores is empty.
    #print("+#+#+#+#+#+#+#+#+#+#+#+")
    if len(v_scores) > 0:
        max_value = max(v_scores)
        max_index = v_scores.index(max_value)
    else:
        # no object detected => no car
        return False
    #print(max_index)
    #print(v_labels[max_index], v_scores[max_index])

    # TODO here changed, now only max box is looked at.
    n = 0
    if v_labels[max_index] == "car" or v_labels[max_index] == "truck":
        n = 1

    ## Check how many cars/(truck for g-class) are within the found objects
    #n = 0
    #for i in range(len(v_boxes)):
    #    if v_labels[i] == "car" or v_labels[i] == "truck":
    #        n += 1
    ## 1 first check if n > 2 -> Zu viele autos
    ## 2 check if n <= 0, keine Autos
    t9 = time.perf_counter()
    #print(f"Label zählen: {t9 - t8} sec")

    if n > 0:  # and n <= 6:
        return True
    else:
        return False


# Tests
#image = "C:/Users/Christian/PycharmProjects/_data\InnovationsProjekt/A205"
#image = image.replace(os.sep, "/")
#image = image+"/"+"id-1280-image1-A205.jpg"

#answer = car_is_in_image(image)
#print(f"Car in Image = {answer}")

