"""
This Script iterates over the trainingdata and filters all images in which no car or truck was detected
with the Car_DetectionV2 Script. It uses a pretrained CenterNet with resnet50v1_fpn_512x512 backbone
for object detection. https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1
Trained on COCO 2017
"""


from Car_detectionV2 import car_is_in_imageV2
import time
import os

filter_to_path = "C:/Users/Christian/PycharmProjects/_data/filteredImages"
filter_from_path = "C:/Users/Christian/PycharmProjects/_data/InnovationsProjekt"


# to define folders that are already searched, so they are not searched again
#when enabling the first line of code
SEARCHED = []

for root, dirs, files in os.walk(filter_from_path, topdown=True):
    #dirs[:] = [d for d in dirs if d not in SEARCHED]  # exlude finished folders

    # Wenn in der files liste des aktuellen subdirectories mindest 1 bild ist
    if len(files) > 0:
        category = root.split(os.sep)[-1]
        print(f"Folder started {category}")
        for file in files:

            t1 = time.perf_counter()
            # check if it is actually is an image
            filter_from_path = root.replace(os.sep, "/")
            ### img = PIL.Image.open(filter_from_path+'/'+file)
            ### img.verify()

            # check if a car is in the image, if not remove it
            # "from-path" is the root
            #if not car_is_in_image(filter_from_path + '/' + file, yolo_model):
            if not car_is_in_imageV2(filter_from_path + '/' + file):
                print(f"replace {filter_from_path + '/' + file} to {filter_to_path + '/' + file}")
                os.replace(filter_from_path+"/"+file, filter_to_path+"/"+file)

            t2 = time.perf_counter()
            print(f"Time 1 file check: {t2 - t1} sec: {file}")


#for image_path in os.listdir()
#to = "C:/Users/Christian/PycharmProjects/test"
#fromP = "C:/Users/Christian/PycharmProjects"
#filen = "image.jpg"
#os.replace(fromP+"/"+filen, to+"/"+filen)