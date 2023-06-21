import numpy as np
import os
from glob import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
import tensorflow as tf
import random
import shutil
import boto3
import argparse
import json
from urllib.parse import urlparse
from PIL import Image as im

video_frames_destination_base_path = "/opt/ml/processing/input/data"
local_output_path = "/opt/ml/processing/output"

HEIGHT=288
WIDTH=512
mag = 1
sigma = 2.5

s3_client = boto3.client('s3')

def genHeatMap(w, h, cx, cy, r, mag):
    '''
    Generates a heatmap for the given X, Y coordinates.
    Args:
       w  width of the image
       h  height of the image
       cx x axis for the center of the ball position
       cy y axis for the center of the ball position
       r  sigma value to calculate the heatmap
       mag heatmap scaling factor
    
    Return:
       x y coordinates with a heatmap corresponds to the ball position.
       
    '''
    if cx < 0 or cy < 0:
        return np.zeros((h, w))
    x, y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
    heatmap = ((y - (cy + 1))**2) + ((x - (cx + 1))**2)
    heatmap[heatmap <= r**2] = 1
    heatmap[heatmap > r**2] = 0
    return heatmap*mag

def process_frame_seqs(video_frame_seq):
    '''
    Creates a dictionary from the given video frame sequence data.
    Args:
        video_frame_seq  video_frame_seq data
    Returns:
        dictionary of video frame sequence data for downstream processing.
     '''
    
    s3r = boto3.resource('s3')
    frame_location_prefix = video_frame_seq['prefix']
    parsed_url = urlparse(frame_location_prefix)
    s3_bucket = parsed_url.hostname
    s3_path = parsed_url.path[1:]
    bucket = s3r.Bucket(s3_bucket)
    dir_created = False
    video_frames_dir = None
    frame_seq_dict = {}
    download = True
    for obj in bucket.objects.filter(Prefix=s3_path):
        if not dir_created:
            video_frames_dir = os.path.join(video_frames_destination_base_path, os.path.dirname(obj.key))
            os.makedirs(video_frames_dir, exist_ok=True)
            dir_created = True
        if download:
            bucket.download_file(obj.key, os.path.join(video_frames_dir, os.path.basename(obj.key)))

    frame_seq_dict['number-of-frames'] = video_frame_seq['number-of-frames']
    frame_seq_dict['frames'] = video_frame_seq['frames']
    frame_seq_dict['local_frame_seq_location'] = video_frames_dir
    return frame_seq_dict

def get_job_name(manifest_output_line_item):
    '''
    Retrieves job name from the given manifest data
    Args:
        manifest_output_line_item  output manifest line item correspond to a labeling job
    Returns:
        ground truth labeling job name
    '''

    metadata = [x for x in manifest_output_line_item.keys() if x.endswith("-metadata") ]
    job_name = manifest_output_line_item[metadata[0]]['job-name']
    return job_name.split("/")[-1]

def get_video_frame_seq(manifest_output_line_item):
    '''
    Retrieves the video frame sequence dictionary from the given output manifest.
    Args:
        manifest_output_line_item - output manifest line item correspond to the labeling job
        
    Returns:
        a dictionary with sequence of video frames information for the given manifest. 
    '''

    parsed_url = urlparse(manifest_output_line_item['source-ref'])
    s3_bucket = parsed_url.hostname
    s3_path = parsed_url.path
    result = s3_client.get_object(Bucket=s3_bucket, Key=s3_path[1:])
    text = result["Body"].read().decode()
    video_frame_seq = json.loads(text)
    return video_frame_seq

def get_label_seq_dict(manifest_output_line_item):
    '''
    Retrieves the label sequence dictionary from the given output manifest.
    Args:
        manifest_output_line_item - output manifest line item correspond to the labeling job
        
    Returns:
        a dictionary with sequence of video frame labels for the given manifest. 
    '''
    
    job_name = get_job_name(manifest_output_line_item)
    manifest_ref_json_file = manifest_output_line_item[f"{job_name}-ref"]
    parsed_url = urlparse(manifest_ref_json_file)
    s3_bucket = parsed_url.hostname
    seq_label_file_path = os.path.join(parsed_url.path[1:])
    result = s3_client.get_object(Bucket=s3_bucket, Key=seq_label_file_path)
    text = result["Body"].read().decode()
    label_seq_dict = json.loads(text)
    return label_seq_dict


def main():
    '''
    Entrypoint function for the proceprocessing job. 
    
    '''
    parser = argparse.ArgumentParser(description='Preprocess the annotated images labeled using Sagemaker Ground Truth labeling job')
    parser.add_argument('--ground_truth_output_dataset_location', help='S3 location of the annotation output', required=True)
    args = parser.parse_args()

    ground_truth_output_dataset_location = args.ground_truth_output_dataset_location
    parsed_url = urlparse(ground_truth_output_dataset_location)
    s3_bucket = parsed_url.hostname
    annotation_manifest_file_path = os.path.join(parsed_url.path[1:], "manifests", "output", "output.manifest")

    result = s3_client.get_object(Bucket=s3_bucket, Key=annotation_manifest_file_path)
    text = result["Body"].read().decode()
    for label_index, line in enumerate(text.split("\n")):
        try:
            manifest_output_line_item = json.loads(line)
            video_frame_seq = get_video_frame_seq(manifest_output_line_item)
            video_seq_label = get_label_seq_dict(manifest_output_line_item)
            process(video_frame_seq, video_seq_label, label_index)
        except json.decoder.JSONDecodeError as e:
            print(f"skipping this line due to error in parsing the content into json format.")

def define_ratio_for_image_resize(frame_loc):
    a = img_to_array(load_img(frame_loc))
    ratio = a.shape[0] / HEIGHT
    return ratio

def build_ground_truth_label_dataframe(frames, annotations):
    '''
    Creates a pandas dataframe for the given video frames and annotations.
    Args:
        frames - video frames dictionary
        annotations - annotations correspond to the video frame number.
    
    Returns:
        dataframe with informaiton about visibitilty of a ball in each video frame.
    '''
    
    label_dicts = []
    for frame in frames:
        label_dict = {}
        label_dict['Visibility'] = 0
        label_dict['X'] = 0
        label_dict['Y'] = 0
        # Frame,Visibility,X,Y
        fr_no = str(frame['frame-no'])
        matching_annotation = [x for x in annotations if x['frame-no'] == fr_no]
        label_dict['Frame'] = fr_no
        keypoints = [] if len(matching_annotation) == 0 else matching_annotation[0]['keypoints']
        if len(keypoints) > 0: # In our use case, there should only be 1 keypoint within a single frame.
            keypoint = keypoints[0]
            label_dict['Visibility'] = 1
            label_dict['X'] = keypoint['x']
            label_dict['Y'] = keypoint['y']
        label_dicts.append(label_dict)
    df = pd.DataFrame(label_dicts)
    return df

def process(video_frame_seq, video_seq_label, label_index):
    '''
    Generates labels from video frames dataset. This function creates a heatmap for each frame
    with a positive label of a ball.
    Args:
        video_frame_seq - video frame sequence in dictionary
        video_seq_label - corresponding video frame label
        label_index - index used for identifying the label files.
    '''
    
    frame_seq_dict = process_frame_seqs(video_frame_seq)
    num_frames = frame_seq_dict['number-of-frames']
    frames = frame_seq_dict['frames']
    frames_location = frame_seq_dict['local_frame_seq_location']
    annotations = video_seq_label['tracking-annotations']
    first_frame_file_path = os.path.join(frames_location, frames[0]['frame'])
    ratio = define_ratio_for_image_resize(first_frame_file_path)
    data = build_ground_truth_label_dataframe(frames, annotations)
    v = data['Visibility'].values
    x = data['X'].values
    y = data['Y'].values
    x_data_tmp = []
    y_data_tmp = []
    for i, frame in enumerate(frames):
        if i + 2 == num_frames:
            break
        unit = []
        for j in range(3):
            target = frames[i+j]['frame']
            img_path = os.path.join(frames_location, target)
            a = load_img(img_path)
            resized_img = a.resize(size=(WIDTH, HEIGHT))
            img_array = img_to_array(resized_img)
            a = np.moveaxis(img_array, -1, 0)
            unit.append(a[0])
            unit.append(a[1])
            unit.append(a[2])
            del a
        x_data_tmp.append(unit)
        del unit
        if v[i+2] == 0:
            heatmap = genHeatMap(WIDTH, HEIGHT, -1, -1, sigma, mag)
            y_data_tmp.append(heatmap)
        else:
            cx = int(x[i+2]/ratio)
            cy = int(y[i+2]/ratio)
            heatmap = genHeatMap(WIDTH, HEIGHT, cx, cy, sigma, mag)
            y_data_tmp.append(heatmap)

    x_data_tmp2 = np.asarray(x_data_tmp)
    del x_data_tmp
    x_data = x_data_tmp2.astype('float32')
    del x_data_tmp2
    x_data=(x_data/255)

    y_data=np.asarray(y_data_tmp)
    del y_data_tmp
    np.save(os.path.join(local_output_path, 'x_data_' + str(label_index) + '.npy'), x_data)
    np.save(os.path.join(local_output_path, 'y_data_' + str(label_index) + '.npy'), y_data)

if __name__ == "__main__":
    main()
