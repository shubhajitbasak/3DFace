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


def get_data_path(evalData, root='examples'):
    if evalData == 'MICC':
        imagepath = root  # '/mnt/sata/code/myGit/3DFaceEvaluation/Data/MICCFlorence/Images'
        im_path = sorted(glob.glob(imagepath + '/**/*.jpg', recursive=True))
        lm_path = [i.replace('jpg', 'txt') for i in im_path]
    else:
        im_path = [os.path.join(root, i) for i in sorted(os.listdir(root)) if i.endswith('png') or i.endswith('jpg')]
        lm_path = [i.replace('png', 'txt').replace('jpg', 'txt') for i in im_path]
        lm_path = [os.path.join(i.replace(i.split(os.path.sep)[-1], ''), 'detections',
                                i.split(os.path.sep)[-1]) for i in lm_path]
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


def main(rank, opt, root='examples', evalData=None):
    device = torch.device(rank)
    torch.cuda.set_device(device)
    model = create_model(opt)
    model.setup(opt)
    model.device = device
    model.parallelize()
    model.eval()
    visualizer = MyVisualizer(opt)

    scene = ['Indoor-Cooperative', 'PTZ-Indoor', 'PTZ-Outdoor']
    for sub in next(os.walk(root))[1]:

        for scn in scene:

            if os.path.exists(os.path.join(root, sub, scn + '.obj')):
                os.remove(os.path.join(root, sub, scn + '.obj'))
                # continue

            name = os.path.join(root, sub, scn)
            print(name, ' : ', len(glob.glob1(name, '*.jpg')))
            im_path, lm_path = get_data_path(evalData, name)
            lm3d_std = load_lm3d(opt.bfm_folder)

            face_shapes = []

            for i in range(len(im_path)):
                # print(i, im_path[i])
                img_name = im_path[i].split(os.path.sep)[-1].replace('.png', '').replace('.jpg', '')
                if not os.path.isfile(lm_path[i]):
                    continue
                im_tensor, lm_tensor = read_data(im_path[i], lm_path[i], lm3d_std)
                data = {
                    'imgs': im_tensor,
                    'lms': lm_tensor
                }
                model.set_input(data)  # unpack data from data loader
                model.test()  # run inference

                face_shapes.append(model.get_shape())

                # print(im_path[i], ': ', test.shape)

                # visuals = model.get_current_visuals()  # get image results
                #
                # if evalData == 'MICC':
                #     visual_save_path = os.path.join(visualizer.img_dir, '/'.join(im_path[i].split('/')[-5:-1]),
                #                                     'epoch_%s' % opt.epoch)
                # else:
                #     visual_save_path = None
                #
                # visualizer.display_current_results(visuals, 0, opt.epoch, dataset=name.split(os.path.sep)[-1],
                #                                    save_results=True, count=i, name=img_name,
                #                                    save_path=visual_save_path, add_image=False)
                #
                # if evalData == 'MICC':
                #
                #     save_obj_path = os.path.join(visualizer.img_dir, '/'.join(im_path[i].split('/')[-5:-1]),
                #                                  'epoch_%s' % opt.epoch, img_name + '.obj')
                #     save_coeff_path = os.path.join(visualizer.img_dir, '/'.join(im_path[i].split('/')[-5:-1]),
                #                                    'epoch_%s' % opt.epoch, img_name + '.mat')
                # else:
                #     save_obj_path = os.path.join(visualizer.img_dir, name.split(os.path.sep)[-1],
                #                                  'epoch_%s_%06d' % (opt.epoch, 0), img_name + '.obj')
                #
                #     save_coeff_path = os.path.join(visualizer.img_dir, name.split(os.path.sep)[-1],
                #                                    'epoch_%s_%06d' % (opt.epoch, 0), img_name + '.mat')
                #
                # model.save_mesh(save_obj_path)  # save reconstruction meshes
                # model.save_coeff(save_coeff_path)  # save predicted coefficients

            print(np.array(face_shapes).shape)
            recon_shape = np.mean(face_shapes, axis=0)
            tri = model.facemodel.face_buf + 1
            objFile = os.path.join(root, sub, scn+'.obj')
            print(objFile)
            util.write_obj(objFile, recon_shape, tri)


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    main(0, opt, root=opt.img_folder, evalData=opt.test_data)
