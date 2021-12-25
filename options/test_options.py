"""This script contains the test options for Deep3DFaceRecon_pytorch
"""

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--name', type=str, default='swin3dface_base', help='name of the experiment. It decides where '
                                                                           'to store samples and models')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--dataset_mode', type=str, default=None, help='chooses how datasets are loaded. [None | '
                                                                           'flist]')
        parser.add_argument('--img_folder', type=str,
                            default='/mnt/sata/data/NowDataset/NoW_Dataset/final_release_version/iphone_pictures',
                            help='folder for test images.')
        parser.add_argument('--test_data', type=str, default='MICC',
                            help='test dataset.')

        # Dropout and Batchnorm has different behavior during training and test.
        self.isTrain = False
        return parser
