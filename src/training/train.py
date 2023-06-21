#!/usr/bin/env python

import numpy as np
import sys, getopt
import os
from glob import glob
import piexif
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from TrackNet import TrackNet
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.activations import *
import tensorflow as tf
import cv2
import math
import json
import smdebug.tensorflow as smd
import argparse


BATCH_SIZE=3
HEIGHT=288
WIDTH=512

def find_ball_center(h):
    '''
    Locate the center of the ball for the given input
    Args:
        h: heatmap representation of an image

    Returns:
      x, y coordinates that identifies the center of the ball.
    '''
    (cnts, _) = cv2.findContours(h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in cnts]
    max_area_idx = 0
    max_area = 0
    for j in range(len(rects)):
       area = rects[j][2] * rects[j][3]
       if area > max_area:
          max_area_idx = j
          max_area = area
    target = rects[max_area_idx]
    return (int(target[0] + target[2] / 2), int(target[1] + target[3] / 2))

def outcome(y_pred, y_true, tol):
   '''
   Calculates true positive, true negative, false positive and false negative for the given input

   Args:
       y_pred: predicted values
       y_true: ground truth values
       tol: tolerance

   Returns:
       true positive, true negative, false positive and false negative scores.
   '''
   n = y_pred.shape[0]
   i = 0
   TP = TN = FP1 = FP2 = FN = 0
   while i < n:
      if np.amax(y_pred[i]) == 0 and np.amax(y_true[i]) == 0:
         TN += 1
      elif np.amax(y_pred[i]) > 0 and np.amax(y_true[i]) == 0:
         FP2 += 1
      elif np.amax(y_pred[i]) == 0 and np.amax(y_true[i]) > 0:
         FN += 1
      elif np.amax(y_pred[i]) > 0 and np.amax(y_true[i]) > 0:
         # transforms the ground truth and prediction outputs into RGB values that correspond to color intensity of a heatmap
         h_pred = y_pred[i] * 255
         h_true = y_true[i] * 255
         h_pred = h_pred.astype('uint8')
         h_true = h_true.astype('uint8')

         (cx_pred, cy_pred) = find_ball_center(h_pred.copy())
         (cx_true, cy_true) = find_ball_center(h_true.copy())

         # calculates the eucledian distance between the prediction and ground truth.
         dist = math.sqrt(pow(cx_pred-cx_true, 2)+pow(cy_pred-cy_true, 2))
         if dist > tol:
            FP1 += 1
         else:
            TP += 1
      i += 1
   return (TP, TN, FP1, FP2, FN)

def calc_metrics(TP, TN, FP1, FP2, FN):
   '''
    Calculates precision and recall scores for the prediction
    Args:
        TP: true positive score
        TN: true negative score
        FP1: false positive score
        FP2: secondary false positive score
        FN:  false negative score

    Returns:
      precision, recall and f1 score in a tuple
   '''
   try:
      accuracy = (TP + TN) / (TP + TN + FP1 + FP2 + FN)
   except:
      accuracy = 0
   try:
      precision = TP / (TP + FP1 + FP2)
   except:
      precision = 0
   try:
      recall = TP / (TP + FN)
   except:
      recall = 0
   try:
      f1 = 2 * (precision * recall) / (precision + recall)
   except:
      f1 = 0.0
   return (accuracy, precision, recall, f1)

def eval(model, dataDir, idx):
    '''
    Evaluates model performance
    Args:
        idx: index or the input numpy array
    Returns:
        evaluation results in a tuple
    '''
    TP = TN = FP1 = FP2 = FN = 0
    for j in idx:
        x_train = np.load(os.path.abspath(os.path.join(dataDir, 'x_data_' + str(j) + '.npy')))
        y_train = np.load(os.path.abspath(os.path.join(dataDir, 'y_data_' + str(j) + '.npy')))
        y_pred = model.predict(x_train, batch_size=BATCH_SIZE)
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype('float32')
        (tp, tn, fp1, fp2, fn) = outcome(y_pred, y_train, tol)
        TP += tp
        TN += tn
        FP1 += fp1
        FP2 += fp2
        FN += fn
        del x_train
        del y_train
        del y_pred
    print("Number of true positive:", TP)
    print("Number of true negative:", TN)
    print("Number of false positive FP1:", FP1)
    print("Number of false positive FP2:", FP2)
    print("Number of false negative:", FN)

    accuracy, precision, recall, f1 = calc_metrics(TP, TN, FP1, FP2, FN)
    print(f"accuracy: {accuracy}, precision: {precision} recall: {recall}, f1: {f1}")
    return

#Loss function
def custom_loss(y_true, y_pred):
   loss = (-1)*(K.square(1 - y_pred) * y_true * K.log(K.clip(y_pred, K.epsilon(), 1)) + K.square(y_pred) * (1 - y_true) * K.log(K.clip(1 - y_pred, K.epsilon(), 1)))
   return K.mean(loss)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("--load_weights", type=str)
  parser.add_argument("--save_weights", type=str, required=True)
  parser.add_argument("--dataDir", type=str, required=True)
  parser.add_argument("--epochs", type=int, required=True)
  parser.add_argument("--tol", type=int, default=10)

  args, unknown = parser.parse_known_args()
  load_weights = args.load_weights
  save_weights = args.save_weights
  dataDir = args.dataDir
  epochs = args.epochs
  tol = args.tol
  print("save_weight is: {}".format(save_weights))

  if not load_weights:
     model=TrackNet(HEIGHT, WIDTH)
     ADADELTA = optimizers.Adadelta(lr=1.0)
     model.compile(loss=custom_loss, optimizer=ADADELTA)
  else:
     model = load_model(load_weights, custom_objects={'custom_loss':custom_loss})

  r = os.path.abspath(os.path.join(dataDir))
  path = glob(os.path.join(r, '*.npy'))
  num = len(path) / 2
  idx = np.arange(num, dtype='int')
  hook = None
  if os.path.exists("/opt/ml/input/config/debughookconfig.json"):
      hook = smd.KerasHook.create_from_json_file()

  print('Beginning training......')
  for i in range(epochs):
     print('============epoch', i+1, '================')
     np.random.shuffle(idx)
     for j in idx:
        x_train = np.load(os.path.abspath(os.path.join(dataDir, 'x_data_' + str(j) + '.npy')))
        y_train = np.load(os.path.abspath(os.path.join(dataDir, 'y_data_' + str(j) + '.npy')))
        if hook:
           model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1, callbacks=[hook])
        else:
           model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1)
        del x_train
        del y_train

     eval(model, dataDir, idx)
     #Save intermediate weights during training
     if (i + 1) % 3 == 0:
        model.save(save_weights + '_' + str(i + 1))

  print('Saving weights......')
  model.save(save_weights)
  print('Done......')
