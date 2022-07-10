"""
Used to iterate over the folders containing the training images.
Classes that are underrepresented are augmentated by random rotations or horizontal flips to the image.
The newly generated Images are saved into a copy of the same folder structure, to not change the original data.

Also used to split the images from 1 folder into 2 filter
A given percentage (default 20%) of the folder with the data
will be moved to a folder with same name but "_TEST_SET" at the end.
"""
from PIL.ExifTags import *
from PIL import Image
import numpy as np
import random
import sys
import os
import math

# path_to_augmentated_images = "C:/Users/Christian/PycharmProjects/_data/InnovationsProjektAugmentated"
path_to_augmentated_images = "C:/Users/Christian/PycharmProjects/_data/InnovationsProjekt"
path_to_images = "C:/Users/Christian/PycharmProjects/_data/InnovationsProjekt"

_class_names = ['A205', 'A207', 'A208', 'A209', 'A217', 'A238', 'C117', 'C118', 'C167', 'C204', 'C205', 'C207', 'C208',
                'C209', 'C215', 'C216', 'C217', 'C218', 'C219', 'C238', 'C253', 'C257', 'C292', 'CL203', 'H247', 'N293',
                'S202', 'S203', 'S204', 'S205', 'S210', 'S211', 'S212', 'S213', 'T245', 'T246', 'V167', 'W163', 'W164',
                'W166', 'W202', 'W203', 'W204', 'W205', 'W210', 'W211', 'W212', 'W213', 'W247', 'W461-463', 'WV140',
                'WV220', 'WV221', 'WV222', 'WV223', 'X117', 'X118', 'X156', 'X164', 'X166', 'X167', 'X204', 'X218',
                'X247', 'X253']


def get_augment_image_params():
    """
    Randomly picks an angle between -20 and 20 degree in stepsize 5
    Also returns flip or noflip string with a 50% chance for each.
    returns the angle and the string
    used as random parameters to augment an image
    :return: angle (int) and flipped (string)
    :rtype:
    """
    pick_array = np.arange(-20, 25, 5, dtype=int)
    np.delete(pick_array, [0])  # we dont want 0
    ## Augment
    angle = random.choice(pick_array)
    flip = random.getrandbits(1)
    flipped = "noflip"
    if flip == 1:
        flipped = "flip"
    return angle, flipped

targetsize = 1900

# loop over every dir check how many files are in there
# if its not enough augment until target is reached
# only augment every image once and dont augment augmented images
# if its not enough repeat but dont augment the same image the way it already is augmented
def augment_images():
    print("Start Augmenting")
    for root, dirs, files in os.walk(path_to_augmentated_images, topdown=True):
        files_to_augment = targetsize - len(files)
        aug_counter = 0
        if len(files) > 0 and len(files) < targetsize:
            augmentation_variants = []
            already_augmented_files = []

            available_files = [x for x in files if x.split("-")[-1] != "aug" and x not in already_augmented_files]
            # Augment until enough files generated
            while aug_counter < files_to_augment:

                # check if new (unused) files to pick from exist
                # else reset of available_files (files will be used twice or more)
                if len(available_files) > 0:
                    file = random.choice(available_files)
                    ###################
                    ## Augment Image ##
                    aug_success = False
                    # create augmentation until one is unique
                    while not aug_success:
                        angle, flipped = get_augment_image_params()
                        if flipped == "flip":
                            # file.split(".")[0] = filename without .jpg
                            new_file_name = file.split(".")[0] + "-" + str(angle) + "-" + flipped + "-" + "aug.jpg"
                        else:
                            new_file_name = file.split(".")[0] + "-" + str(angle) + "-" + "aug.jpg"
                        # dont augment files the exact same way
                        if new_file_name not in augmentation_variants:
                            aug_success = True
                            im = Image.open(path_to_augmentated_images + "/" + root.split(os.sep)[-1] + "/" + file)
                            # rotate image
                            im = im.rotate(angle)
                            # flip image
                            if flipped == "flip":
                                im = im.transpose(Image.FLIP_LEFT_RIGHT)
                            ## crop to remove the black outlines
                            crop_factor = 75  # tested value that works vor -20 to 20 rotation
                            width, height = im.size
                            im = im.crop((crop_factor, crop_factor, width - crop_factor, height - crop_factor))
                            # example path: C:/Users/Christian/PycharmProjects/_data/InnovationsProjekt\S204
                            # os.walk uses os.sep for subfolders thats why the last one has "\" and not "/"
                            im.save(path_to_augmentated_images + "/" + root.split(os.sep)[-1] + "/" + new_file_name)
                    ###################

                    augmentation_variants.append(new_file_name)
                    already_augmented_files.append(file)
                    aug_counter += 1
                else:
                    # reset available files again
                    already_augmented_files = []
                    available_files = [x for x in files if x.split("-")[-1] != "aug" and x not in already_augmented_files]

def delete_augmented_Images():
    for root, dirs, files in os.walk(path_to_augmentated_images, topdown=True):
        for file in files:
            print(root)
            if file.split("-")[-1] == "aug.jpg":
                # bsp root: C:/Users/Christian/PycharmProjects/_data/InnovationsProjektAugmentated\W205
                # python can handle different seperator in the path, no need to change the backslash before classname
                os.remove(root + "/" + file)

def split_images_in_two_folder(trainsize=0.8, testsize=0.2):
    """
    splits a directory of images (with names like id-1280-image1-A205.jpg) in form of
    - imgdir
        - class1_dir
            - img1
        - class2_dir
            - img2
    into two directories with the given split size. Default 80% train 20% Test.
    Images with same ID are always put together into train or test.
    :param trainsize:
    :type trainsize:
    :param testsize:
    :type testsize:
    :return:
    :rtype:
    """
    print("-splitting-")
    for root, dirs, files in os.walk(path_to_images, topdown=True):
        if len(files) > 0:
            # we are in a folder with images:
            # This iteration is done for each class folder
            unique_ids = []
            id_count = {}
            ## CALCULATE HOW MANY IDs TO MOVE
            for file in files:
                # example name: id-1280-image1-A205.jpg
                id = file.split("-")[1]
                if id not in unique_ids:
                    unique_ids.append(id)  # add unique id
                    id_count[id] = 0  # add new unique id as key to dict
                id_count[id] += 1  # count dict for this id
            avg_id_count = sum(id_count.values()) / len(unique_ids)
            avg_id_count = int(math.ceil(avg_id_count))  # round up to full int
            ids_to_move_count = int(len(files) * testsize)  # amount of images to move to test set
            ids_to_move_count = int(math.ceil(ids_to_move_count / avg_id_count))  # calculate how many id sets have to be moved

            ## MOVE IMAGES
            random_ids_to_move = random.sample(unique_ids, ids_to_move_count)

            ## Go back 1 directory in the path
            # Takes the path goes one back and adds the same folder with "_TEST_SET" at the end
            # i.e. from path "../data/myimages" to "../data/myimages_TEST_SET"
            path_to_images_root = ""
            path_to_images_root_list = path_to_images.split("/")[:-1]
            for piece in path_to_images_root_list:
                path_to_images_root += piece + "/"
            path_to_images_test = path_to_images_root + path_to_images.split("/")[-1] + "_TEST_SET"
            ##
            #
            current_class_str = root.split(os.sep)[-1]
            for file in files:
                id = file.split("-")[1]
                if id in random_ids_to_move:
                    ## create Folders if needed
                    if not os.path.exists(path_to_images_test):
                        os.mkdir(path_to_images_test)
                    if not os.path.exists(path_to_images_test + "/" + current_class_str):
                        os.mkdir(path_to_images_test + "/" + current_class_str)
                    ##
                    ## Move Files
                    os.replace(path_to_images + "/" + current_class_str + "/" + file,
                               path_to_images_test + "/" + current_class_str + "/" + file)



# SET HERE WHICH FUNTION TO USE

#split_images_in_two_folder()
#augment_images()
#delete_augmented_Images()





