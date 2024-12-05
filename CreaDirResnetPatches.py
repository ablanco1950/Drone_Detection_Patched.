
imgpath_from= "Drone-Detection-data-set(yolov7)-1\\train\\images"
labelpath_from="Drone-Detection-data-set(yolov7)-1\\train\\labels"

DimPatch=128
Num_clases=2

import cv2
import os
import shutil
import re


imgpath_to_train= "train"
imgpath_to_valid= "valid"
imgpath_to_test= "test"

if  os.path.exists("train"):shutil.rmtree("train")
os.mkdir("train")
if  os.path.exists("valid"):shutil.rmtree("valid")
os.mkdir("valid")
if  os.path.exists("test"):shutil.rmtree("test")
os.mkdir("test")


for i in range (Num_clases):
    j=i+1
    NameDir=str(j)
    if len(NameDir) < 2: NameDir= "0" + NameDir
    os.mkdir("train\\"+NameDir)
    os.mkdir("valid\\"+NameDir)
    os.mkdir("test\\"+NameDir)



def unconvert(width, height, x, y, w, h):

    xmax = int((x*width) + (w * width)/2.0)
    xmin = int((x*width) - (w * width)/2.0)
    ymax = int((y*height) + (h * height)/2.0)
    ymin = int((y*height) - (h * height)/2.0)

    return xmin, ymin, xmax, ymax


   
           
          

#################################################################
# MAIN
#################################################################

Num_img=0
for root, dirnames, filenames in os.walk(imgpath_from):
    
 for filename in filenames:
             
   if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                 
                 
                # filepath = os.path.join(root, filename)
                
                 
                #image = cv2.imread(filepath)

                
                Num_img=Num_img+1
                #if Num_img > 10: break
                if Num_img > 2920:
                    imgpath_to="test"
                    
                elif Num_img > 2600:
                    imgpath_to="valid"
                    
                else:
                    imgpath_to="train"
                  
                    
                                
                imgpath= imgpath_from+ "\\" + filename
                labelpath=labelpath_from+"\\" +  filename[0:len(filename)-4] + ".txt"
                

                img=cv2.imread(imgpath)

                imgOriginal=img.copy()

                #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #cv2.imshow("Prueba",img)
                #cv2.waitKey(0)
                
                f=open(labelpath,"r")

                lxywh=""          
                for linea in f:         
                    lxywh=linea[2:]
                    break
                xywh=lxywh.split(" ")
                width=float(img.shape[0])
                height=float(img.shape[1])
                x=float(xywh[0])
                y=float(xywh[1])
                w=float(xywh[2])
                h=float(xywh[3])
                xTrue,yTrue,xMaxTrue,yMaxTrue=unconvert(width, height, x, y, w, h)
                                
                start_pointTrue=(int(xTrue),int(yTrue)) 
                end_pointTrue=(int(xMaxTrue),int( yMaxTrue))
                                
                #colorTrue=(0,0,255)
                                 
                # Using cv2.rectangle() method
                # Draw a rectangle with green line borders of thickness of 2 px
                #img = cv2.rectangle(img, start_pointTrue, end_pointTrue,(0,255,0), 2)
                #cv2.imshow("Prueba",img)
                #cv2.waitKey(0)
                #print(img.shape)
                N_Rows=int(img.shape[0] / DimPatch)
                N_Colums=int(img.shape[1] / DimPatch)

                #print( "Rows = " + str(N_Rows) + " Colums = " + str(N_Colums))

                for i in range (N_Colums-1):
                    for j in range(N_Rows-1):
                        #crop_img=img[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
                        start_point=(j*DimPatch,i*DimPatch)
                        end_point=((j+1)*DimPatch,(i+1)*DimPatch)
                        end_point_left=(j*DimPatch,(i+1)*DimPatch)
                        start_point_right=((j+1)*DimPatch,i*DimPatch)

                        

                        if (start_point[0] > start_pointTrue[0] and start_point[0] < end_pointTrue[0] \
                                 and  start_point[1] > start_pointTrue[1] and start_point[1] < end_pointTrue[1]):
                            
                           img = cv2.rectangle(img, start_point, end_point,(255,0,0), 2)
                           filenamew=filename+ "_" +str((i+1)*100 + j)
                           filenamewImage=imgpath_to + "\\01\\" +filenamew + ".jpg"
                           crop_img=imgOriginal[start_point[1]:end_point[1],start_point[0]:end_point[0]]
                           cv2.imwrite(filenamewImage,crop_img)
                                      
                        
                        elif (end_point_left[0] > start_pointTrue[0] and end_point_left[0] < end_pointTrue[0] \
                                 and  end_point_left[1] > start_pointTrue[1] and end_point_left[1] < end_pointTrue[1]): 
                           img = cv2.rectangle(img, start_point, end_point,(255,0,0), 2)
                            
                           filenamew=filename+ "_" +str((i+1)*100 + j)
                           filenamewImage=imgpath_to + "\\01\\" +filenamew + ".jpg"
                           crop_img=imgOriginal[start_point[1]:end_point[1],start_point[0]:end_point[0]]
                           cv2.imwrite(filenamewImage,crop_img)
                                      
                        
                        elif    (end_point[0] < end_pointTrue[0] and end_point[0] > start_pointTrue[0] \
                                 and  end_point[1] < end_pointTrue[1] and end_point[1] > start_pointTrue[1]):
                           img = cv2.rectangle(img, start_point, end_point,(255,0,0), 2)
                            
                           filenamew=filename+ "_" +str((i+1)*100 + j)
                           filenamewImage=imgpath_to + "\\01\\" +filenamew + ".jpg"
                           crop_img=imgOriginal[start_point[1]:end_point[1],start_point[0]:end_point[0]]
                           cv2.imwrite(filenamewImage,crop_img)
                                      
                            
                        elif    (start_point_right[0] < end_pointTrue[0] and start_point_right[0] > start_pointTrue[0] \
                                 and  start_point_right[1] < end_pointTrue[1] and start_point_right[1] > start_pointTrue[1]):
                           img = cv2.rectangle(img, start_point, end_point,(255,0,0), 2)
                            
                           filenamew=filename+ "_" +str((i+1)*100 + j)
                           filenamewImage=imgpath_to + "\\01\\" +filenamew + ".jpg"
                           crop_img=imgOriginal[start_point[1]:end_point[1],start_point[0]:end_point[0]]
                           cv2.imwrite(filenamewImage,crop_img)
                                      

                        elif    ( end_pointTrue[0] < end_point[0] and start_point[0]< end_pointTrue[0] \
                                 and  start_point_right[1] < end_pointTrue[1] and end_point[1] > end_pointTrue[1]):
                           img = cv2.rectangle(img, start_point, end_point,(255,0,0), 2)
                            
                           filenamew=filename+ "_" +str((i+1)*100 + j)
                           filenamewImage=imgpath_to + "\\01\\" +filenamew + ".jpg"
                           crop_img=imgOriginal[start_point[1]:end_point[1],start_point[0]:end_point[0]]
                           cv2.imwrite(filenamewImage,crop_img)
                                      

                        elif    ( start_pointTrue[0] > start_point[0] and start_pointTrue[0]< end_point[0] \
                                 and  start_pointTrue[1] > start_point[1] and start_pointTrue[1] < end_point[1]):
                           img = cv2.rectangle(img, start_point, end_point,(255,0,0), 2)
                            
                           filenamew=filename+ "_" +str((i+1)*100 + j)
                           filenamewImage=imgpath_to + "\\01\\" +filenamew + ".jpg"
                           crop_img=imgOriginal[start_point[1]:end_point[1],start_point[0]:end_point[0]]
                           cv2.imwrite(filenamewImage,crop_img)
                                      
                            
                            
                        else:    
                           img = cv2.rectangle(img, start_point, end_point,(0,255,0), 2)
                            
                           filenamew=filename+ "_" +str((i+1)*100 + j)
                           filenamewImage=imgpath_to + "\\02\\" +filenamew + ".jpg"
                           crop_img=imgOriginal[start_point[1]:end_point[1],start_point[0]:end_point[0]]
                           cv2.imwrite(filenamewImage,crop_img)
                        
                        #crop_img=img[i*DimPatch:(i+1)*DimPatch,j*DimPatch:(j+1)*DimPatch]
                        #cv2.imshow("Patch",crop_img)
                        #cv2.imshow("Patch",img)
                        #cv2.waitKey(0)
