"""This script is modified version of data preparation script for Deep3DFaceRecon_pytorch
"""
import os
import numpy as np
import argparse
from util.detect_lm68 import detect_68p, load_lm_graph
from util.skin_mask import get_skin_mask
from util.generate_list import check_list, write_list

import glob

import cv2
import os
# from util.detect_lm5 import generate5keypoints

import warnings

# datapath = 'data\CASIA-WebFace'
# landmark_path_root = 'data\CASIA-WebFace'

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='/mnt/sata/data/VGG-Face2/data',
                    help='root directory for training data')
parser.add_argument('--img_folder', nargs="+",  # required=True,
                    default=['vggface2_train_mod'], help='folders of training images')  # , 'lfw'
parser.add_argument('--mode', type=str, default='train', help='train or val')
opt = parser.parse_args()


def checkFileCounts(dataPath):
    imagePath = os.path.join(dataPath, '')


def generateImageList(dataPath):
    img_path = os.path.join(dataPath, 'Images')
    if os.path.exists(os.path.join(dataPath, 'images.txt')):
        os.remove(os.path.join(dataPath, 'images.txt'))
    with open(os.path.join(dataPath, 'images.txt'), 'w') as f1:
        for root, dirs, files in os.walk(img_path):
            if len(files) > 0:
                imgList = glob.glob(root + '/' + '*.jpg')
                for img in imgList:
                    detect = img.replace('Images', 'Detections').replace('jpg', 'txt')
                    if os.path.exists(img) and os.path.exists(detect):
                        print(img)
                        f1.writelines(img + '\n')  # +','+detect


def generate5keypointdata(img_path):
    for root, dirs, files in os.walk(img_path):
        if len(files) > 0:
            imgList = glob.glob(root + '/' + '*.jpg')
            detectionRoot = root.replace('Images', 'Detections')
            if not os.path.exists(detectionRoot):
                os.makedirs(detectionRoot, exist_ok=True)

            for img in imgList:
                image = cv2.imread(img)
                ext = img.split('.')[-1]
                detectionPath = img.replace('Images', 'Detections').replace(ext, 'txt')
                if os.path.exists(detectionPath):
                    os.remove(detectionPath)
                lndmrks = generate5keypoints(image)
                if lndmrks is not None:
                    s = True
                    with open(detectionPath, "a") as f:  # img_addr.split('.')[0] + ".txt"
                        for i in lndmrks:
                            print(str(i[0]) + ' ' + str(i[1]), file=f)
                else:
                    print('Issue : ', img)
                    os.remove(img)


def data_prepare(folder_list, mode):
    lm_sess, input_op, output_op = load_lm_graph(
        './checkpoints/lm_model/68lm_detector.pb')  # load a tensorflow version 68-landmark detector

    for img_folder in folder_list:
        detect_68p(img_folder, lm_sess, input_op, output_op)  # detect landmarks for images
        get_skin_mask(img_folder)  # generate skin attention mask for images

    # create files that record path to all training data
    msks_list = []

    for img_folder in folder_list:
        msks_path = os.path.join(img_folder, 'mask')
        for root, dirs, files in os.walk(msks_path):
            if len(files) > 0:
                msks_list += glob.glob(root + '/' + '*.jpg')

    # for img_folder in folder_list:
    #     path = os.path.join(img_folder, 'mask')
    #     msks_list += ['/'.join([img_folder, 'mask', i]) for i in sorted(os.listdir(path)) if 'jpg' in i or
    #                   'png' in i or 'jpeg' in i or 'PNG' in i]

    imgs_list = [i.replace('mask', 'images') for i in msks_list]
    lms_list = [i.replace('mask', 'landmarks') for i in msks_list]
    lms_list = ['.'.join(i.split('.')[:-1]) + '.txt' for i in lms_list]

    lms_list_final, imgs_list_final, msks_list_final = check_list(lms_list, imgs_list,
                                                                  msks_list)  # check if the path is valid
    write_list(lms_list_final, imgs_list_final, msks_list_final, mode=mode)  # save files


if __name__ == '__main__':
    # generate 5 key points
    # generate5keypoints(image_path)

    # generate image list file
    # generateImageList(datapath)

    print('Datasets:', opt.img_folder)
    data_prepare([os.path.join(opt.data_root, folder) for folder in opt.img_folder], opt.mode)
