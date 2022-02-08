import logging
import os
import time

import torch
from tensorboardX import SummaryWriter
from torchvision.utils import save_image

__all__ = ['Recorder']


class Recorder:
    def __init__(self, base_path, exp_name, logger_name, code_path, save_code=True, level=logging.INFO):
        '''
        :param base_path: base path for exp sets
        :param exp_name: description of current exp
        :param logger_name: __name__
        :param code_path: __file__
        :param save_code: backup the __main__ file or not
        :param level: logging level
        '''
        # folder init
        suffix = time.strftime("-%Y-%m-%d-%H-%M-%S", time.localtime())
        self.exp_path = os.path.join(base_path, exp_name + suffix)
        os.makedirs(self.exp_path)

        # set up logger
        self.logger = logging.getLogger(logger_name)
        formatter = logging.Formatter('%(asctime)s %(levelname)s : %(message)s')
        fileHandler = logging.FileHandler(os.path.join(self.exp_path, 'output.log'), mode='w')
        fileHandler.setFormatter(formatter)
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)

        self.logger.setLevel(level)
        self.logger.addHandler(fileHandler)
        self.logger.addHandler(streamHandler)

        # set up tensorboard writer
        self.writer = SummaryWriter(os.path.join(self.exp_path, 'tfboard'))

        # save image path
        self.save_image_path = os.path.join(self.exp_path, 'imgs')
        os.mkdir(self.save_image_path)

        # backup code
        self.code_path = code_path
        if save_code:
            self.cp_code(code_path)

        # save model path
        self.save_model_path = os.path.join(self.exp_path, 'ckps')
        os.mkdir(self.save_model_path)

    def save_img(self, tensor, filename, **kwargs):
        save_image(tensor, os.path.join(self.save_image_path, filename), **kwargs)

    def cp_code(self, code_path):
        from shutil import copyfile
        copyfile(code_path, os.path.join(self.exp_path, 'main.py'))

    def add_scalars_from_dict(self, scalars_dict: dict, global_step):
        for key in scalars_dict.keys():
            self.writer.add_scalar(key, scalars_dict[key], global_step)

    def save_model(self, state_dict, filename):
        torch.save(state_dict, os.path.join(self.save_model_path, filename))

    def close(self):
        self.writer.close()
