import os

# os.environ['CUDA_VISIBLE_DEVICES'] = ''

import glob
import time
import random
import sys

import cv2
import keras
import numpy as np
import tensorflow as tf

from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet.utils.tracker_old import iou

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    # import keras_retinanet.bin
    #
    # __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
import models

random.seed(2018)
save_path = '/home/yul-j/Desktop/Smoke/Detection/logs/0824_v3/test_result_30'

# snapshots_fld = "/home/yul-j/Desktop/Smoke/Detection/logs/0824_v3/snapshots/"
snapshots_fld = '/home/yul-j/Desktop/Project/Safetyhat/safetyhat-retinanet/snapshots/r50_InvColor_Rotation_tot/'
# model_subpath = "mobilenet224_1.0_NewDataSet_Base_NewAchor_BS4/mobilenet224_1.0_pascal_100_inference.h5"
# model_subpath = 'resnet50_pascal_30_inference.h5'
model_subpath = 'resnet50_pascal_100_inference.h5'
# model_subpath = 'resnet50_NewDataSet_Gray_InvColor_Rotation_NewAchor_BS4/resnet50_pascal_50_inference.h5'
# backbone = 'mobilenet224_1.0'
backbone = 'resnet50'
# model_subpath = "resnet50_NewDataSet_Base_NewAnchor_BS4/resnet50_pascal_25_inference.h5"
model_path = os.path.join(snapshots_fld, model_subpath)

im_source1 = '/home/yul-j/Desktop/Test/0830/*'
# im_source1 = '/home/yul-j/Desktop/Safetyhat/data/backup/difficault/*'
# im_source1 = '/home/yul-j/Desktop/Data/safetyhat/data/backup/unused'
# im_source2 = '/home/yul-j/Desktop/Safetyhat/data/test/test.txt'
# source2_path = '/home/yul-j/Desktop/Safetyhat/data/VOC2007/JPEGImages'
# im_source1 = '/home/yul-j/Desktop/*'

im_list = glob.glob(im_source1)

if 'im_source2' in locals().keys():
    f = open(im_source2)
    line = f.readline()
    while line:
        im_name = line.split('\n')[0] + '.jpg'
        im_path = os.path.join(source2_path, im_name)
        im_list.append(im_path)
        line = f.readline()


# random.shuffle(im_list)
im_list.sort()
# im_list = im_list[::-1]


def drop_large_overlaped_boxes(detect_result, drop_iou=0.5):
    drop_index = []  # record which boxes to drop

    scores = detect_result[:, 4]
    labels = detect_result[:, 5]
    boxes = detect_result[:, :4]

    num_boxes = len(boxes)
    for i in range(num_boxes - 1):
        for j in range(i + 1, num_boxes):
            if labels[i] == labels[j]:
                continue
            else:
                iou_ij = iou(boxes[i], boxes[j])
                if iou_ij >= drop_iou:

                    idx = np.argmin((scores[i], scores[j]))
                    if idx:
                        drop_index.append(j)
                    else:
                        drop_index.append(i)

    if not len(drop_index):
        return detect_result
    else:
        detect_result = np.delete(detect_result, drop_index, axis=0)

    return detect_result


def vis_detections(im, results):
    """Draw detected bounding boxes."""
    hat_thresh = 0.5
    nohat_thresh = 0.5
    rois = []
    results = drop_large_overlaped_boxes(results)
    num_head = len(results)
    if num_head == 0:
        return im
    else:
        img = im.copy()
        for i in range(num_head):
            bbox = results[i, :4]
            score = results[i, 4]
            label = results[i, 5]
            roi = im[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            rois.append(roi)

            if label:
                print("class: hat", "score: ", score, "left-top: ", int(bbox[0]), ", ", int(bbox[1]))
                if score < hat_thresh:
                    continue
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            else:
                print("class: nohat", "score: ", score, "left-top: ", int(bbox[0]), ", ", int(bbox[1]))
                if score < nohat_thresh:
                    continue
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

            img = cv2.putText(img, str(score)[:6], (bbox[0], bbox[1]), font, 0.5, (0, 0, 0), 2)
        return img


def vis_detections_track(im, tracker, frame):
    """Draw detected bounding boxes."""
    if type(tracker[0]) == int:
        tracker = tracker[1:]

    if len(tracker) == 0:
        return im
    else:
        img = im.copy()
        img = cv2.putText(img, str(frame), (10, 30), font, 2, (255, 255, 255), 2)
        for obj in tracker:
            bbox = obj["position"]
            score = np.round(obj["track_score"], 4)
            if not obj["show"]:
                continue
            else:

                if obj["class"] == -1:  # 无安全帽
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                    img = cv2.putText(img, str(obj["contains"]),
                                      (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)),
                                      font, 0.5, (255, 255, 255), 2)
                    img = cv2.putText(img, str(score)[:6], (int(bbox[0]), int(bbox[1])), font, 0.5, (255, 255, 255), 2)
                else:  # 有安全帽
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    img = cv2.putText(img, str(obj["contains"]),
                                      (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)),
                                      font, 0.5, (255, 255, 255), 2)
                    img = cv2.putText(img, str(score)[:6], (int(bbox[0]), int(bbox[1])), font, 0.5, (255, 255, 255), 2)

        print("--------------------------------------------------------------------------")
        return img


def iou_for_debug(tracker, idx1, idx2):
    box1 = tracker[idx1 + 1]["pred_position"]
    box2 = tracker[idx2 + 1]["pred_position"]
    return iou(box1, box2)


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# load retinanet model
model = models.load_model(model_path, backbone=backbone, convert=False)
# print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'nohathead', 1: 'hathead'}
labels_to_colors = {0: (0, 255, 0), 1: (0, 255, 0)}

time_log = []
orignal_image_size_log = []
resized_image_size_log = []
scale_log = []
scores_log = []
boxes_log = []
labels_log = []
img_idx = 0
length = len(im_list)

while img_idx < length:
    im_path = im_list[img_idx]
    if not im_path.split('.')[-1] in ['jpg', 'jpeg', 'JPG']:
        del (im_list[img_idx])
        length -= 1
        continue
    print(im_path)
    image = cv2.imread(im_path)

    if image is None:
        del (im_list[img_idx])
        length -= 1
        continue

    img_save = image.copy()
    im_name = im_path.split('/')[-1]
    im_save_path = os.path.join(save_path, im_name)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # draw = image.copy()
    # image = preprocess_image(image)
    print("original image size:", image.shape)
    orignal_image_size_log.append(image.shape)
    image, scale = resize_image(image)
    print("resized image size:", image.shape)
    resized_image_size_log.append(image.shape)
    scale_log.append(scale)

    print("scale:", scale)

    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    end = time.time()
    boxes_log.append(boxes)
    scores_log.append(scores)
    labels_log.append(labels)
    print("processing time: ", end - start)
    time_log.append(end - start)
    print("mean time:", np.mean(time_log))
    # img_idx += 1
    # boxes /= scale

    hat_idx = np.where(labels[0] == 1)[0]
    nohat_idx = np.where(labels[0] == 0)[0]
    det_hat = np.hstack((boxes[0][hat_idx],
                         scores[0][hat_idx][:, np.newaxis],
                         labels[0][hat_idx][:, np.newaxis])).astype(np.float32)
    det_nohat = np.hstack((boxes[0][nohat_idx],
                           scores[0][nohat_idx][:, np.newaxis],
                           labels[0][nohat_idx][:, np.newaxis])).astype(np.float32)
    dets = np.vstack((det_hat, det_nohat))

    frame_no_track = vis_detections(image, dets)

    cv2.namedWindow(im_name)  # Create a named window
    cv2.moveWindow(im_name, 1000, 30)  # Move it to (40,30)
    cv2.imshow(im_name, frame_no_track)

    op = cv2.waitKey(0)
    cv2.destroyAllWindows()

    if op == ord('s'):
        cv2.imwrite(im_save_path, img_save)
        print('image saved as:', save_path)
    elif op == ord('a'):
        img_idx -= 1
        img_idx = max(0, img_idx)
    elif op == ord('d'):
        img_idx += 1
    elif op == ord('r'):
        cv2.imwrite(im_save_path, frame_no_track)
        print('image saved as:', save_path)
    elif op == ord('q'):
        break

    # cv2.imshow("img", frame_no_track)
    # cv2.waitKey(0)

print('done')