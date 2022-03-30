"""This script is the test script for Deep3DFaceRecon_pytorch
"""

import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import MyVisualizer
from util.preprocess import align_img
from PIL import Image
import numpy as np
from util.load_mats import load_lm3d
from util import util
import torch
from data.flist_dataset import default_flist_reader
from scipy.io import loadmat, savemat
import glob


def get_data_path(root='examples', eval = None):
    # validation
    if eval:
        valList = '/mnt/sata/data/NowDataset/NoW_Dataset/Validation/imagepathsvalidation.txt'
        with open(valList,mode='r') as f:
            t = f.readlines()
        im_path = [os.path.join(root, i.strip()) for i in t]
        lm_path = [i.replace('jpg', 'txt') for i in im_path]
    else:
        lm_path = sorted(glob.glob(root + '/**/IMG*.txt', recursive=True))
        im_path = [i.replace('txt', 'jpg') for i in lm_path]

    for im,lm in zip(im_path, lm_path):
        if os.path.exists(im) and os.path.exists(lm):
            # print('True')
            pass
        else:
            print(im)

    return im_path, lm_path


def read_data(im_path, lm_path, lm3d_std, to_tensor=True):
    # to RGB
    im = Image.open(im_path).convert('RGB')
    W, H = im.size
    lm = np.loadtxt(lm_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    _, im, lm, _ = align_img(im, lm, lm3d_std)
    if to_tensor:
        im = torch.tensor(np.array(im) / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm).unsqueeze(0)
    return im, lm


def main(rank, opt, root='examples', eval=None):
    device = torch.device(rank)
    torch.cuda.set_device(device)
    model = create_model(opt)
    model.setup(opt)
    model.device = device
    model.parallelize()
    model.eval()
    visualizer = MyVisualizer(opt)

    im_path, lm_path = get_data_path(root, eval)
    lm3d_std = load_lm3d(opt.bfm_folder)

    for i in range(len(im_path)):

        print(im_path)
        if not os.path.isfile(lm_path[i]):
            continue
        im_tensor, lm_tensor = read_data(im_path[i], lm_path[i], lm3d_std)
        data = {
            'imgs': im_tensor,
            'lms': lm_tensor
        }
        pred_obj_path = im_path[i].replace('final_release_version/iphone_pictures',
                                           'ModelOutput/SwinBaseNeutral').replace('jpg', 'obj')
        pred_viz_path = pred_obj_path.replace('obj', 'png')
        if not os.path.exists(os.path.dirname(pred_obj_path)):
            os.makedirs(os.path.dirname(pred_obj_path), exist_ok=True)

        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        model.save_mesh_neutral(pred_obj_path)
        model.save_visuals(pred_viz_path)
        # shp = model.get_shape()
        # lmp = model.get_lm()

        # print(np.array(face_shapes).shape)
        # recon_shape = np.mean(face_shapes, axis=0)
        # tri = model.facemodel.face_buf + 1
        # objFile = os.path.join(root, sub, scn+'.obj')
        # print(objFile)
        # util.write_obj(objFile, recon_shape, tri)


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    main(0, opt, root=opt.img_folder, eval=True)
