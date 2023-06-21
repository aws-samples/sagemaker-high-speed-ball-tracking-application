import warnings
import sys
import numpy as np
import os
from glob import glob
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import pandas as pd
import cv2
import time
import logging 
import argparse

import sagemaker
from sagemaker.local import LocalSession
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
import json
from sagemaker.predictor import Predictor

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
warnings.filterwarnings("ignore", category=FutureWarning)

HEIGHT=288
WIDTH=512
mag=1

def get_sm_predictor(endpoint_name):
    '''
    Creates a Sagemaker predictor with an endpoint specified.
    Args:
        endpoint_name name of the Sagemaker inference endpoint to create a predictor with

    Returns:
        a sagemaker Predictor object.
    '''
    session = None
    if endpoint_name.lower() == "local":
        session = LocalSession()
        session.config = {'local': {'local_code': True}}
    else:
        session = sagemaker.Session()
        
    return Predictor(
          endpoint_name=endpoint_name,
          sagemaker_session=session,
          serializer=JSONSerializer(),
          deserializer=JSONDeserializer()
    )

#time: in milliseconds
def custom_time(time):
    '''
    Reformats time for prediction output file.
    Args:
       input time in milliseconds
    Returns:
       formatted time for the given input
    '''
    
    remain = int(time / 1000)
    ms = (time / 1000) - remain
    s = remain % 60
    s += ms
    remain = int(remain / 60)
    m = remain % 60
    remain = int(remain / 60)
    h = remain
    #Generate custom time string
    cts = ''
    if len(str(h)) >= 2:
        cts += str(h)
    else:
        for i in range(2 - len(str(h))):
            cts += '0'
        cts += str(h)
    
    cts += ':'

    if len(str(m)) >= 2:
        cts += str(m)
    else:
        for i in range(2 - len(str(m))):
            cts += '0'
        cts += str(m)

    cts += ':'

    if len(str(int(s))) == 1:
        cts += '0'
    cts += str(s)

    return cts
        
def process(args):
    '''
    Creates ball trajectory tracking video for the input video file.
    
    Args:
        args: Arguments passed by the client invoking the script
    
    Returns: None
    '''
    
    input_video_name = args.input_video_file_path
    output_video_name = args.output_video_file_path
    endpoint_name = args.endpoint_name
    
    predictor = get_sm_predictor(endpoint_name)
    logging.info('Beginning predicting......')

    start = time.time()

    f = open(output_video_name[:-4]+'_predict.csv', 'w')
    f.write('Frame,Visibility,X,Y,Time\n')

    cap = cv2.VideoCapture(input_video_name)

    success, image1 = cap.read()
    success, image2 = cap.read()
    success, image3 = cap.read()

    ratio = image1.shape[0] / HEIGHT

    size = (int(WIDTH*ratio), int(HEIGHT*ratio))
    fps = 30

    if input_video_name[-3:] == 'avi':
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    elif input_video_name[-3:] == 'mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        logging.error('usage: video type can only be .avi or .mp4')
        exit(1)

    out = cv2.VideoWriter(output_video_name[:-4]+'_predict'+output_video_name[-4:], fourcc, fps, size)

    out.write(image1)
    out.write(image2)

    count = 2

    while success:
        unit = []
        #Adjust BGR format (cv2) to RGB format (PIL)
        x1 = image1[...,::-1]
        x2 = image2[...,::-1]
        x3 = image3[...,::-1]
        #Convert np arrays to PIL images
        x1 = array_to_img(x1)
        x2 = array_to_img(x2)
        x3 = array_to_img(x3)
        #Resize the images
        x1 = x1.resize(size = (WIDTH, HEIGHT))
        x2 = x2.resize(size = (WIDTH, HEIGHT))
        x3 = x3.resize(size = (WIDTH, HEIGHT))
        #Convert images to np arrays and adjust to channels first
        x1 = np.moveaxis(img_to_array(x1, dtype=np.uint8), -1, 0)        
        x2 = np.moveaxis(img_to_array(x2, dtype=np.uint8), -1, 0)        
        x3 = np.moveaxis(img_to_array(x3, dtype=np.uint8), -1, 0)
        #Create data
        unit.append(x1[0])
        unit.append(x1[1])
        unit.append(x1[2])
        unit.append(x2[0])
        unit.append(x2[1])
        unit.append(x2[2])
        unit.append(x3[0])
        unit.append(x3[1])
        unit.append(x3[2])
        unit=np.asarray(unit)    
        unit = unit.reshape((1, 9, HEIGHT, WIDTH)).astype('float32')
        unit /= 255
        lists = unit.tolist()
        y_pred_remote = predictor.predict(data=lists)
        y_pred = np.array(y_pred_remote["predictions"], dtype="float32").copy()
        y_pred = y_pred > 0.5
        h_pred = y_pred[0]*255
        h_pred = h_pred.astype('uint8')
        frame_time = custom_time(cap.get(cv2.CAP_PROP_POS_MSEC))
        if np.amax(h_pred) <= 0:
            f.write(str(count)+',0,0,0,'+frame_time+'\n')
            out.write(image3)
        else:    
            #h_pred
            (cnts, _) = cv2.findContours(h_pred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rects = [cv2.boundingRect(ctr) for ctr in cnts]
            max_area_idx = 0
            max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
            for i in range(len(rects)):
                area = rects[i][2] * rects[i][3]
                if area > max_area:
                    max_area_idx = i
                    max_area = area
            target = rects[max_area_idx]
            (cx_pred, cy_pred) = (int(ratio*(target[0] + target[2] / 2)), int(ratio*(target[1] + target[3] / 2)))

            f.write(str(count)+',1,'+str(cx_pred)+','+str(cy_pred)+','+frame_time+'\n')
            image3_cp = np.copy(image3)
            cv2.circle(image3_cp, (cx_pred, cy_pred), 5, (0,0,255), -1)
            out.write(image3_cp)
        image1 = image2
        image2 = image3
        success, image3 = cap.read()
        count += 1

    f.close()
    out.release()
    end = time.time()
    logging.info('Prediction time: {} secs'.format(end-start))
    logging.info('Count of frames: {}'.format(count))
    logging.info('Done......')


def main():
    parser = argparse.ArgumentParser( prog = 'sm_training.py',
                    description = 'A script that trains a deep learning model based on TrackNet on SageMaker Training job.')
    
    parser.add_argument('--input_video_file_path', required=True, help="input location of a video file for ball trajectory tracking creation")
    parser.add_argument('--output_video_file_path', help="output location of the video file with ball trajectory tracking. Default is in the same directory as input")
    parser.add_argument('--endpoint_name', required=True, help="Sagemaker endpoint name. if referecing a local endpoint, pass 'local' ")
    args = parser.parse_args()
    process(args)

if __name__ == "__main__":
    main()

