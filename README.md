# Drone_Detection_Patched.
Drone_Detection_Patched.

Experiment that tries to convert a resnet model obtained using as input dataset https://universe.roboflow.com/drone-detection-pexej/drone-detection-data-set-yolov7/dataset/1 into an object detection: drones

Installation:

Download all the project files into a folder on disk and unzip the Test1.zip and Test2.zip files

Download the files from https://universe.roboflow.com/drone-detection-pexej/drone-detection-data-set-yolov7/dataset/1 (you need to have a Roboflow key, which is free ) The download folders appear in the Drone-Detection-data-set(yolov7)-1 folder

Create the folder layout that resnet requires by running:

python CreateDirResnetPatches.py

Train the resnet model:

python TrainDronePatcheds_Resnet_Pytorch.py

Evaluation with the images from the test folder of the Roboflow download:

python EvaluateDroneByPatches.py

the detected pieces are marked in blue and the envelope in red.

Conclusions:

The project aims to create an image detector from an image classification model (resnet).

The results are poor (they will be improved in later editions) and are mainly attributed to:

The size of the patch: 128 is excessively large, the ideal would be 16, but this generates an excessive number of small files to train.

The grouping of pieces is very simple, an envelope of all the pieces

References:

 https://universe.roboflow.com/drone-detection-pexej/drone-detection-data-set-yolov7/dataset/1 into an object detection: drones

 The resnet cnn is the same as in:
 https://github.com/ablanco1950/CarsBrands_Resnet_Pytorch

 https://github.com/ablanco1950/Drone-Detection_Yolov10
 
 
