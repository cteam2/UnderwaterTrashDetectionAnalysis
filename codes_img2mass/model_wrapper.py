import keras
import numpy as np
import os

import util
import features
import cv2
import numba

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image

def visualize_original_bbox(bbox_coords, img):
    fig, axs = plt.subplots(1, figsize=(8, 6))
    axs.imshow(img)
    rect = convert_to_rotated_bbox(bbox_coords)

    rx, ry, rw, rh, rtheta = rect
    print("the width and height is ", rw, rh)
    rect_patch = patches.Rectangle(
        (rx - rw/2 , ry - rh/2  ), rw, rh, 
        linewidth=1, edgecolor='r', facecolor='none', 
        angle=rtheta
    )

    axs.add_patch(rect_patch)
    # axs[2].set_xlim(0, img.shape[1])
    # axs[2].set_ylim(img.shape[0], 0)
    # Plot the rectangle on the mask
    axs.set_title('Rectangle on img')
    axs.set_axis_off()
    plt.show()


def visualize_image_mask_rect(mask, rect, img):
    
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot the original image
    axs[0].imshow(img)
    axs[0].set_title('Original Image')
    
    # Plot the mask
    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title('Mask')

    # Draw the rectangle on the mask
#    rect_image = np.copy(mask)
    rx, ry, rw, rh, rtheta = rect

    rect_patch = patches.Rectangle(
        (rx - rw/2 , ry - rh/2 ), rw, rh, 
        linewidth=1, edgecolor='r', facecolor='none', 
        angle=rtheta
    )
    axs[2].imshow(mask, cmap='gray')
    axs[2].add_patch(rect_patch)
    axs[2].set_xlim(0, mask.shape[1])
    axs[2].set_ylim(mask.shape[0], 0)
    # Plot the rectangle on the mask
    axs[2].set_title('Rectangle on Mask')

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Call the function with the model and image
# visualize_image_mask_rect(thickness_model, img)

def convert_to_rotated_bbox(bounding_box):
    """
    Convert axis-aligned bounding box coordinates to a rotated bounding box format.
    Parameters:
        x1, y1: The top-left corner coordinates of the bounding box.
        x2, y2: The bottom-right corner coordinates of the bounding box.
    Returns:
        A list containing the center coordinates (rx, ry), width (rw), height (rh),
        and rotation angle (rtheta) of the bounding box.
    """
    x1,y1,x2,y2 = bounding_box
    rx = (x1 + x2) / 2.0  # Center x-coordinate
    ry = (y1 + y2) / 2.0  # Center y-coordinate
    rw = x2 - x1           # Width of the bounding box
    rh = y2 - y1           # Height of the bounding box
    rtheta = 0             # Rotation angle (0 for axis-aligned boxes)
    return [rx, ry, rw, rh, rtheta]

def get_mask_only(thickness_model,img):
    small_img = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
    small_img = np.asarray(small_img)
    mask = predict(thickness_model,small_img)
    binary_mask = get_filter_mask(small_img)
    coarse_binary_mask = cv2.resize(binary_mask.astype(np.uint8),(32,32), interpolation=cv2.INTER_AREA) > .001
        
    mask = mask * coarse_binary_mask
    mask = cv2.resize(mask, (19,19), interpolation=cv2.INTER_AREA)
    return mask


def give_scaled_rectangle(bbox_coords,size):
    original_height,original_width,channels = size
    new_width,new_height = 299,299
    rx,ry,rw,rh,rtheta = convert_to_rotated_bbox(bbox_coords)

    scale_x = new_width / original_width
    scale_y = new_height / original_height

    rx_scaled = rx * scale_x
    ry_scaled = ry * scale_y
    rw_scaled = rw * scale_x
    rh_scaled = rh * scale_y

    rect = [rx_scaled,ry_scaled,rw_scaled,rh_scaled,rtheta]
    return rect

def clip_and_makebg(rect,img):
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # axs[0].imshow(img)
    

    mask = np.ones_like(img) * 255
    rx,ry,rw,rh,rtheta = rect
    x1_scaled = int(rx - rw / 2)
    y1_scaled = int(ry- rh / 2)
    x2_scaled = int(rx + rw / 2)
    y2_scaled = int(ry+ rh/ 2)

    mask[y1_scaled:y2_scaled, x1_scaled:x2_scaled] = img[y1_scaled:y2_scaled, x1_scaled:x2_scaled]
    # axs[1].imshow(mask)
    # plt.show()
    return mask

def get_model(model_contains):
    newest = max(map(lambda x: 'models/'+x, [x for x in os.listdir('models') if '.h5' in x and model_contains in x]), key=os.path.getctime)
#    print(newest)
    model = keras.models.load_model(newest)
    return model, newest[7:]

def get_points_from_mask(mask):

    cntr = []
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            t = mask[x][y]
            if t > .001:
                cntr.append([[y,x]])
    return cntr

def get_rect_from_mask(mask):
    cntr=get_points_from_mask(mask)
    if len(cntr)<3:
        return [(16,16),[0,0],0]
    try:
        rect = cv2.minAreaRect(np.array(cntr))
    except:
        return [(16,16),[0,0],0]
    return rect

def predict(model, image):
    cannonical_transformations = util.get_cannonical_transformations()
    
    xes = []
    for func, inverse in cannonical_transformations:
        xes.append(func(image))
    
    ys = model.predict(np.array(xes))
    y = np.zeros_like(ys[0])
    for i,(func, inverse) in enumerate(cannonical_transformations):
        better = inverse(ys[i])
        y+=better
        
    y /= len(cannonical_transformations)
    return y

@numba.jit
def get_filter_mask(img,threshold=250):
    mask = (np.sum(img,axis=2) < threshold*3).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)).astype(np.uint8)
    
    ret = mask
    ret = cv2.morphologyEx(ret, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)).astype(np.uint8)
    ret = cv2.erode(ret,kernel,iterations = 1)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4)).astype(np.uint8)
    ret = cv2.morphologyEx(ret, cv2.MORPH_OPEN, kernel)
    
    ret = ret.astype(np.bool)
    return ret

def get_mask_and_rect(thickness_model,img):
    small_img = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
    small_img = np.asarray(small_img)
    mask = predict(thickness_model,small_img)
    binary_mask = get_filter_mask(small_img)
    coarse_binary_mask = cv2.resize(binary_mask.astype(np.uint8),(32,32), interpolation=cv2.INTER_AREA) > .001
        
    mask = mask * coarse_binary_mask
    rect = get_rect_from_mask(mask)
    ((rx,ry),(rw,rh),rtheta) = rect
    rect = np.array([rx,ry,rw,rh,rtheta])
    mask = cv2.resize(mask, (19,19), interpolation=cv2.INTER_AREA)
    return mask,rect

class ComplexModel:
    def __init__(self,simple=False):
        if simple == 'no_geometry':
            self.model, self.model_name = get_model('no_geometry')
        elif not simple:
            self.model, self.model_name = get_model('shape_aware')
        else:
            self.model, self.model_name = get_model('cnn_w_alde')
        self.input_shape=(None,299,299,3)

        self.thickenss_model = None
        
    def predict(self, x):
        if len(x) ==2:
            img, dims = x
            # dims is like [99.72371673583984, 165.0560302734375, 226.5860595703125, 247.29486083984375]
            if self.thickenss_model is None:
                self.thickenss_model = keras.models.load_model('thickness_model.h5')
                        
            # mask, rect = get_mask_and_rect(self.thickenss_model,img)
            img = clip_and_makebg(convert_to_rotated_bbox(dims),img)
            mask = get_mask_only(self.thickenss_model,img)
            
            # this section contains code to convert the dimensions of the 
            # bounding box of image to the dimensions of the bounding box 
            # of the mask 
            dscale_for_mask = 0.0635 # for the 256x256 then 19x19 resize
            #bbox is like [99.72371673583984, 165.0560302734375, 226.5860595703125, 247.29486083984375]
            bbox_coords = dims 
            scaled_bbox_coords = list(map(lambda x:x*dscale_for_mask,bbox_coords))
            rect = give_scaled_rectangle(scaled_bbox_coords,img.shape)
            # print(rect)
            # end of the section

            #this sections deals with converting the dimensions(which are in pixel value) 
            # scale them to a real world value (in inches)  
            mask_scale = 15.73
            real_scale = 0.0296
            # clip only the part of the image with the object and find np.max()
            # clipped_mask = clip_from_rectangle(rect,mask)
            dims = [abs(dims[0]-dims[2])*real_scale,abs(dims[1]-dims[3])*real_scale,np.max(mask)*32*mask_scale*real_scale]
            print("the dimensions of the image should be : ",dims) #this is the size of real image 
            # dims = [dims[0],dims[1],np.max(mask)*32*mask_scale*real_scale]
            # end of the section


            if img.shape != (299,299,3):
                img=cv2.resize(img, (299,299), interpolation=cv2.INTER_AREA)
        else:
            img,mask,dims,rect = x

        visualize_image_mask_rect(mask,rect, img)
        dummy=0
        item = (img,mask,dummy,dims,rect)
        
        img,mask,aux_input = features.prepare_features(item)

                    
        cannonical_transformations = util.get_cannonical_transformations()

        imgs = []
        masks = []
        auxes = []
        for func, _ in cannonical_transformations:
            imgs.append(func(img))
            masks.append(func(mask))
            auxes.append(np.copy(np.array(aux_input)))
        predictions = self.model.predict([np.array(imgs),np.array(masks),np.array(auxes)])
            
        return np.median(predictions)
    
class SimpleModel:
    def __init__(self):
        self.model, self.model_name = get_model('pure_cnn')
        self.input_shape=(None,299,299,3)
    def predict(self, x):
        image,mask = x
        print(mask)
        img,mask,dims,rect = x

        cannonical_transformations = util.get_cannonical_transformations()
        
        imgs = []
        for func, _ in cannonical_transformations:
            imgs.append(func(img))
        predictions = self.model.predict(np.array(imgs))
            
        return util.percentile_to_score(np.median(predictions))*np.prod(dims)/27.6799
