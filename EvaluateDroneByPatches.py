

DimPatch=128
ConfLimit=2.5

model_path= "checkpoint10epoch2Classes.pth"

import torch
from torch import nn

import cv2
import os
import re

imgpath_from= "test\\images"
#imgpath_from= "C:\\Drone-Detection_Yolov10\\Drone-Detection-data-set(yolov7)-1\\test\\images"
#imgpath_from= "Test1"
#imgpath_from= "Test2"

def process_image(image):
    
    # Process a PIL image for use in a PyTorch model
  
    # Converting image to PIL image using image file path
    pil_im = Image.open(f'{image}')

  
    transform = transforms.Compose([transforms.Resize((DimPatch,DimPatch)),
                                            transforms.CenterCrop(DimPatch),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    # Transforming image for use with network
    pil_tfd = transform(pil_im)
    
    # Converting to Numpy array 
    array_im_tfd = np.array(pil_tfd)
    
    return array_im_tfd

def load_checkpoint(filepath):

    checkpoint = torch.load(filepath)
    
    model.load_state_dict(checkpoint['state_dict'], strict=False)
   
    return model

def predict(image_path, model, topk=2):
    # Implement the code to predict the class from an image file   
    
    # Loading model - using .cpu() for working with CPUs
    loaded_model = load_checkpoint(model).cpu()
    # Pre-processing image
    img = process_image(image_path)
    # Converting to torch tensor from Numpy array
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    # Adding dimension to image to comply with (B x C x W x H) input of model
    img_add_dim = img_tensor.unsqueeze_(0)

    # Setting model to evaluation mode and turning off gradients
    loaded_model.eval()
    with torch.no_grad():
        # Running image through network
        output = loaded_model.forward(img_add_dim)
        
    #conf, predicted = torch.max(output.data, 1)   
    probs_top = output.topk(topk)[0]
    predicted_top = output.topk(topk)[1]
    
    # Converting probabilities and outputs to lists
    conf = np.array(probs_top)[0]
    predicted = np.array(predicted_top)[0]
        
    #return probs_top_list, index_top_list
    return conf, predicted                

#################################################################
# MAIN
#################################################################
import numpy as np

import time
inicio=time.time()

from torchvision import datasets, transforms, models
import torchvision.models as models
from PIL import Image

# https://stackoverflow.com/questions/53612835/size-mismatch-for-fc-bias-and-fc-weight-in-pytorch
model = models.resnet50(pretrained=True)    
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # make the change


checkpoint=torch.load(model_path)
model.load_state_dict(checkpoint, strict=False)  # load

# https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-state-dict-recommended
# https://stackoverflow.com/questions/55900754/why-am-i-getting-different-results-after-saving-and-loading-model-weights-in-pyt
model.eval()


Num_img=0
for root, dirnames, filenames in os.walk(imgpath_from):
    
 for filename in filenames:
             
   if re.search("\.(jpg|jpeg|JPEG|png|bmp|tiff)$", filename):
                
                 
               
                Num_img=Num_img+1
                #if Num_img > 10: break
                
                imgpath= imgpath_from+ "\\" + filename 
               
                img=cv2.imread(imgpath)

                # https://medium.com/@saairaamprasad/opencv-in-python-image-processing-part-2-10-10cc9ab91a95
                # cv2.INTER_LANCZOS4 (best quality, slowest)
                #img = cv2.resize(img, (640,640), interpolation = cv2.INTER_AREA)
                if img.shape[0] !=640 or img.shape[1] !=640:
                   img = cv2.resize(img, (640,640), interpolation = cv2.INTER_LANCZOS4)
                imgOriginal=img.copy()

                #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #cv2.imshow("Prueba",img)
                #cv2.waitKey(0)
               
                N_Rows=int(img.shape[0] / DimPatch)
                N_Colums=int(img.shape[1] / DimPatch)

                #print( "Rows = " + str(N_Rows) + " Colums = " + str(N_Colums))
                #print("***")
                
                Xmax=0.0
                Xmin=999.0
                Ymax=0.0
                Ymin=999.0
                SwHay=0
                for i in range (N_Colums-1):
                  for j in range(N_Rows-1):
                    #crop_img=img[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
                    start_point=(j*DimPatch,i*DimPatch)
                    end_point=((j+1)*DimPatch,(i+1)*DimPatch)
                    end_point_left=(j*DimPatch,(i+1)*DimPatch)
                    start_point_right=((j+1)*DimPatch,i*DimPatch)
                    crop_img=imgOriginal[start_point[1]:end_point[1],start_point[0]:end_point[0]]
                    cv2.imwrite("pp.jpg", crop_img)
                    
                  
                    conf, predicted1=predict("pp.jpg", model_path, topk=1)
                    #print(conf)
                    #print(predicted1)
                    #if predicted1[0]==0 and conf[0] > 1.5:
                    if predicted1[0]==0 and conf[0] > ConfLimit:   
                        SwHay=1
                        #print("===")
                        #print(conf)
                        img = cv2.rectangle(img, start_point, end_point,(255,0,0), 2)
                        if Xmin > start_point[0]: Xmin=start_point[0]
                        if Xmax < end_point[0]: Xmax=end_point[0]
                        if Ymin > start_point[1]: Ymin=start_point[1]
                        if Ymax < end_point[1]: Ymax=end_point[1]
                if SwHay == 1:        
                   img = cv2.rectangle(img, (Xmin, Ymin), (Xmax, Ymax),(0,0,255), 2)
                else:
                    print(filename + " NON DETECTED")
                cv2.imshow(filename,img)
                cv2.waitKey(0)
