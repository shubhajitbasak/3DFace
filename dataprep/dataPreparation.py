import glob

import cv2
import os
from util.detect_lm5 import generate5keypoints

image_path = r'C:\Users\sbasak\Downloads\CASIA-WebFace\Images'
landmark_path_root = r'C:\Users\sbasak\Downloads\CASIA-WebFace\Detections'


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


