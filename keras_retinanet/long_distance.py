import os

import cv2
import keras
import numpy as np
import tensorflow as tf

from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.image import preprocess_image, resize_image

model_path = "/home/yul-j/Desktop/Safetyhat/safetyhat-retinanet/inference/0608/resnet50_pascal_21.h5"

parent = "/home/yul-j/Desktop/安全巡检/VideoTranscode/"
video_name = "时代星胜3号楼围墙_2018_05_24_10_51_11_201805302020139.avi"
out_name = video_name.split("_")[0] + "_out.avi"
out_path = os.path.join(parent, out_name)

font = cv2.FONT_HERSHEY_SIMPLEX


def vis_detections(im, dets):
    """Draw detected bounding boxes."""
    rois = []
    inds_hathead = np.where(dets[0][:, -1] >= 0.2)[0]
    inds_nohathead = np.where(dets[1][:, -1] >= 0.2)[0]
    num_head = len(inds_hathead) + len(inds_nohathead)
    if num_head == 0:
        return im
    else:
        img = im.copy()
        for i in inds_nohathead:
            bbox = dets[1][i, :4]
            score = dets[1][i, -1]
            roi = im[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            rois.append(roi)

            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            img = cv2.putText(img, str(score)[:6], (bbox[0], bbox[1]), font, 0.5, (255, 255, 255), 1)
            # print("class: nohat", "score: ", score, "left-top: ", bbox[0], ", ", bbox[1])

        for i in inds_hathead:
            bbox = dets[0][i, :4]
            score = dets[0][i, -1]
            roi = im[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            rois.append(roi)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            img = cv2.putText(img, str(score)[:6], (bbox[0], bbox[1]), font, 0.5, (255, 255, 255), 1)
            # print("class: hat", "score: ", score, "left-top: ", bbox[0], ", ", bbox[1])

        return img


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# load retinanet model
model = keras.models.load_model(model_path, custom_objects=custom_objects)
# print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'nohathead', 1: 'hathead'}
labels_to_colors = {0: (0, 0, 255), 1: (0, 255, 0)}

video_path = os.path.join(parent, video_name)
video = cv2.VideoCapture(video_path)

fps = video.get(cv2.CAP_PROP_FPS)

size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
writer = cv2.VideoWriter(out_path, cv2.CAP_ANY, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps=fps, frameSize=size)

ret, image = video.read()
i = 0
while ret:
    print(i)
    i += 1
    draw = image.copy()

    image = preprocess_image(image)
    image, scale = resize_image(image)

    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

    boxes /= scale

    hat_idx = np.where(labels[0] == 1)[0]
    nohat_idx = np.where(labels[0] == 0)[0]
    det_hat = np.hstack((boxes[0][hat_idx],
                         scores[0][hat_idx][:, np.newaxis])).astype(np.float32)
    det_nohat = np.hstack((boxes[0][nohat_idx],
                           scores[0][nohat_idx][:, np.newaxis])).astype(np.float32)
    dets = [det_hat, det_nohat]

    frame_no_track = vis_detections(draw, dets)
    # frame_no_track = cv2.resize(frame_no_track, (int(size[0] * 2 / 3), int(size[1] * 2 / 3)))

    # cv2.imshow("a", frame_no_track)
    # cv2.waitKey(0)

    writer.write(frame_no_track)

    ret, image = video.read()
