import argparse
import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from models.yolo import YoloV3
from utils import draw_outputs, transform_images
import matplotlib.pyplot as plt

CLASSES_PATH = 'coco.names'
WEIGHTS_PATH = 'yolov3.tf'
IMAGE_SIZE = 416

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default=None, type=str)

    yolo = YoloV3()
    yolo.load_weights(WEIGHTS_PATH)
    class_names = [c.strip() for c in open(CLASSES_PATH).readlines()]
    if parser.file is not None:
        img = tf.image.decode_image(open(parser.file, 'rb').read(), channels=3)
        plt.imshow(img)
        plt.show()

        input_img = tf.expand_dims(img, 0)
        input_img = transform_images(input_img, IMAGE_SIZE) 

        boxes, scores, classes, nums = yolo(input_img)


        logging.info('detections:')
        for i in range(nums[0]):
            print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                            np.array(scores[0][i]),
                                            np.array(boxes[0][i])))

        prediction_img = draw_outputs(img.numpy(), (boxes, scores, classes, nums), class_names)
        plt.figure(figsize=(20, 20))
        plt.imshow(prediction_img)
        plt.show()
    else:
        print("Add image file path!")