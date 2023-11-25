import os
HOME = os.getcwd()
print(HOME)
from IPython import display
import ultralytics
from ultralytics import YOLO

from IPython.display import display, Image
from PIL import Image


class Detect():
    def find_trash(self,img):
        # Load a model
        model = YOLO('best.pt')  # load a custom model

        # Predict with the model
        results = model(img)  # predict on an image
        # print("hi")
        bbox=results[0].boxes.data
        # print(bbox)
        bbox=bbox.tolist()
        # print(bbox)
        # print(results)
        # print(results[0].boxes.data)
        # print(results[0])
        # Image.plot_bboxes(image, results[0].boxes.data, score=False)\

        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            im.show()  # show image
            im.save('results.jpg')  # save image
            # print(r)

        return bbox

# d= Detect()
# d.find_trash("sample_img/vid_000555_frame0000040.jpg")
    # yolo task=detect mode=predict model={HOME}/runs/detect/train/weights/best.pt conf=0.25 source={dataset.location}/test/images save=True