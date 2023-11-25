import os
from os import listdir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sewar
from niqe import niqe
import torch
from PIL import Image
from skimage import img_as_float, restoration
import cv2

import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')

class Enhance():
  # def __init__(self):
  #    #constcuctorcode
  #    pass
  def UIE(self,sample_img_path):
    img_1 = cv2.imread(sample_img_path)
    filtered_image = cv2.bilateralFilter(img_1, 9, 75, 75)
    sample_img = Image.fromarray(np.uint8(filtered_image))
    sample_img.save('img_stage_1.jpg')

    ##Contrast correction
    #Rayleigh stretching
    img= cv2.imread("img_stage_1.jpg")
    img1=img.copy()
    img2=img.copy()
    b,g,r=cv2.split(img2)
    b2,g2,r2= cv2.split(img)
    b1,g1,r1=cv2.split(img1)
    imdb = ((b2.max()-b2.min())/2) + b2.min()

    alpha=0.9
    b[b<imdb]=imdb
    for index,value in np.ndenumerate( b ):
      new_value=((255* (value-imdb))/ ((b2.max()-b2.min())/(alpha**2)))
      b[index]= new_value

    imdg = ((g2.max()-g2.min())/2) + g2.min()
    g[g<imdg]=imdg
    for index,value in np.ndenumerate(g):
        new_value=((255* (value-imdg))/((g2.max()-g2.min())*(alpha**2)))
        g[index]= new_value

    imdr = ((r2.max()-r2.min())/2) + r2.min()
    r[r<imdr]=imdr
    for index,value in np.ndenumerate(r):
        new_value=((255* (value-imdr))/ ((r2.max()-r2.min())/(alpha**2)))
        r[index]= new_value

    b1[b1>imdb]=imdb
    for index,value in np.ndenumerate( b1 ):
        new_value=((255* (value-b2.min()))/ ((b2.max()-b2.min())*(alpha**2)))
        b1[index]= new_value

    g1[g1>imdg]=imdg
    for index,value in np.ndenumerate(g1):
      new_value=((255* (value-g2.min()))/((g2.max()-g2.min())*(alpha**2)))
      g1[index]= new_value

    imdr = ((r2.max()-r2.min())/2) + r.min()
    r1[r1>imdr]=imdr
    for index,value in np.ndenumerate(r1):
        new_value=((255* (value-r2.min()))/ ((r2.max()-r2.min())*(alpha**2)))
        r1[index]=new_value

    res= cv2.merge((b,g,r))
    res1= cv2.merge((b1,g1,r1))
    res2= cv2.addWeighted(res,.5,res1,.5,0)



    fin= self.adjust_gamma(res2,1.2)
    cv2.imwrite("img_stage_2_1.jpg", fin)

    #CLAHE


    clahe_image_input = cv2.imread('img_stage_2_1.jpg')
    clahe_image = self.apply_clahe_color(clahe_image_input)
    clahe_image= Image.fromarray(np.uint8(clahe_image))
    clahe_image.save('img_stage_2_2.jpg')

    ##Color correction
    gwa_input= cv2.imread("img_stage_2_2.jpg")
    img_array = np.asarray(gwa_input)
    avg_r = np.mean(img_array[:,:,0])
    avg_g = np.mean(img_array[:,:,1])
    avg_b = np.mean(img_array[:,:,2])
    avg_all = (avg_r + avg_g + avg_b) / 3
    # Scaling factor for each channel
    max_scale = 1.5
    scale_r = min(avg_all / avg_r, max_scale)
    scale_g = avg_all / avg_g
    scale_b = avg_all / avg_b
    # Then apply scaling factor to each respective channel
    # To prevent red channel over-correction, we can limit the amount of scaling for it
    img_array[:,:,0] = np.clip(img_array[:,:,0] * scale_r, 0, 255)
    img_array[:,:,1] = np.clip(img_array[:,:,1] * scale_g, 0, 255)
    img_array[:,:,2] = np.clip(img_array[:,:,2] * scale_b, 0, 255)
    img_gray_world = Image.fromarray(np.uint8(img_array))
    img_gray_world.save('img_stage_3.jpg')

    output_folder = 'enhanced_images'
    os.makedirs(output_folder, exist_ok=True)

    path= sample_img_path.rsplit(".", 1)[0] + '.jpg'
    path= path.rsplit("/", 1)[1]
    # os.path.join(output_folder, 'enhanced_image.jpg')
    # final_path= output_folder+path
    enhanced_image_path = os.path.join(output_folder, path)

    # final_path= data_new+path
    img_fin = cv2.imread('img_stage_3.jpg')
    # cv2.imwrite(final_path, img_fin)
    cv2.imwrite(enhanced_image_path, img_fin)
    return enhanced_image_path


  def adjust_gamma(self,image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

  def apply_clahe_color(self,image, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_l_channel = clahe.apply(l_channel)
    enhanced_lab_image = cv2.merge((clahe_l_channel, a_channel, b_channel))
    enhanced_bgr_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)
    return enhanced_bgr_image