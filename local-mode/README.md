# High-speed Ball Tracking Application in Sports Broadcast Videos

## Introduction

In ball Sports, ball tracking data is considered one of the fundamental and useful information in evaluating players’ performance and game strategies. Until recently, it was a challenge to come up with a reliable technique that would accurately recognize and position balls in sports that involve tiny balls moving at high-speed. For instance, tennis, badminton, baseball or golf. 

In this project, we are going to build an end to end machine learning workflow that prepares video files, performs model training and standing up a realtime endpoint in SageMaker. A sample application is also included that takes a sports broadcast video and produce a separate video file that contains ball trajectory thats overlays the original video. 

### Starting Point
To get started, you need to clone the project into your SageMaker Studio environment:

```
> git clone https://github.com/wei-m-teh/sagemaker-tracknet-v2
> cd sagemaker-tracknet-v2
> pip install -r requirements.txt
```

## Ground Truth Labeling Job

Assuming the original broadcast video files are uploaded to S3 bucket, we are going to apply appropriate labels to the video frames. SageMaker [Ground Truth](https://aws.amazon.com/sagemaker/data-labeling) is a data labeling service that makes it easy to label data in various formats and gives you the option to use human annotators through Amazon Mechanical Turk, third-party vendors, or your own private workforce. In our example, we are going to create a ground truth labeling job with private workforce to annotate the uploaded videos.

Instructions on how to create a SageMaker Ground Truth labeling job can be found [here](../Part1_Labeling.ipynb).

## Feature Engineering 

Once the video files are labeled, we will create the features to train a model. Given large volume of labeled data (labels are applied to every video frame in video files), we will leverage a SageMaker Processing job to help us featurize the dataset required for training a model. Instructions on how to create a SageMaker Processing Job locally can be found [Part2_Processing_Local Notebook](Part2_Processing_Local.ipynb).


## Model Training

Once feature engineering step is complete, all the input data required for training a model should be available in the speicfied S3 bucket location. We will trigger a SageMaker Training job to train a model, as followed [Part3_Training_Local Notebook](Part3_Training_Local.ipynb).


## Deploy Model Endpoint and Generate Ball Tracking Trajectory Video Files
After the model is trained successfully, we can now deploy a SageMaker endpoint to serve inference for the video files. 
With a realtime endpoint deployed in SageMaker, you can then create ball tracking videos by integrating the deployed endpoint with your own videos. A sample application is included in this repository to demonstrate the capability in action [Part4_Inference_Local Notebook](Part4_Inference_Local.ipynb).