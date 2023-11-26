import sys
from PIL import Image
import numpy as np

#import image_util
from model_wrapper import ComplexModel


shape_aware_model = ComplexModel()

def predict_mass(filename,dims):
    global shape_aware_model
    im = Image.open(filename)
    #im = image_util.resize_and_pad_image(im,(299,299))
    im = np.array(im)

    output = shape_aware_model.predict((im,dims))
    return output

def main():
    try:
        # filename,l,w = sys.argv[1:]	#taking the bounding box width and height  
#        filename, l,w,h = sys.argv[1:]
        filename, x1,y1,x2,y2 = sys.argv[1:]

    except ValueError:
        # print("Usage: $ python3 predict_mass.py filename length width height")
        # print("Length, width, and height must be floating point numbers in inches")
        # print("Example: $ python3 predict_mass.py filename length width height")
        print("Usage $ python3 predict_mass.py filename x1 y1 x2 y2")
        sys.exit(1)


#    print("Got: filename=",filename,'dimensions= (', l, 'inches by', w, 'inches by', h, 'inches.)')
#    print("Got: filename=",filename,'dimensions= (', l, 'inches by', w, 'inches )')    
#    dims = (float(l),float(w),float(h))
    # dims = (float(l),float(w))
    print("Got: filename=",filename,'dimensions= (', x1,y1, x2, y2,')')
    dims = (float(x1),float(y1),float(x2),float(y2))
    output = predict_mass(filename,dims)
    #print(filename, 'probably weighs about', output, 'pounds.')
    print(filename, 'probably weighs about', round(output*453.592,2), 'grams.')

if __name__ == "__main__":
    main()
