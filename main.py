# import cv2
import numpy as np
from PIL import Image
import os

# Import the ImgEnhance class from your module
from img_enhance import Enhance
from trash_det import Detect
from mass_estimate import Estimate
names={0: 'bio', 1: 'rov', 2: 'trash_etc', 3: 'trash_fabric', 4: 'trash_fishing_gear', 5: 'trash_metal', 6: 'trash_paper', 7: 'trash_plastic', 8: 'trash_rubber', 9: 'trash_wood'}
image_paths=[]
bbox={}


image_dir="sample_img"
for img in os.listdir(image_dir):
    if img.lower().endswith('.jpg') or img.lower().endswith('.jpeg'):
        image_paths.append(image_dir+"/"+img)
for i in image_paths:
    enhancer = Enhance()
    enhanced_img=enhancer.UIE(i)
    detector= Detect()
    bbox[i]=detector.find_trash(enhanced_img)
    if(bbox[i]!=[]):
        if int(bbox[i][0][-1])>1:
            estimator=Estimate()
            

    # print(list(bbox[i]))
    # if(int(bbox[i][-1][-1])>1):
    #     print("trash",int(bbox[i][-1][-1]))
    #     estimator=Estimate()

# img="sample_img/vid_000027_frame0000056.jpg"
# fin="./final"
# Create an instance of the ImgEnhance class
# print(bbox)

