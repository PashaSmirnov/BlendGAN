import argparse
import os

import cv2
import numpy as np
import torch
from torchvision.transforms import ToPILImage

from BlendGAN.model import Generator
from BlendGAN.psp_encoder.psp_encoders import PSPEncoder
from BlendGAN.utils import ten2cv, cv2ten
import glob
import random

seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class StyleTransferer:
    MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrained_models')

    def __init__(self, device='cuda', size=1024, add_weight_index=6, channel_multiplier=2, latent=512, n_mlp=8):
        self._size = size
        self._add_weight_index = add_weight_index
        self._channel_multiplier = channel_multiplier
        self._latent = latent
        self._n_mlp = n_mlp

        self._device = device

        ckpt_path = os.path.join(self.MODELS_DIR, 'blendgan.pt')
        self._checkpoint = torch.load(ckpt_path)
        self._model_dict = self._checkpoint['g_ema']

        self._g_ema = Generator(
            self._size,
            self._latent,
            self._n_mlp,
            channel_multiplier=self._channel_multiplier,
            load_pretrained_vgg=False
        ).to(self._device)
        self._g_ema.load_state_dict(self._model_dict)
        self._g_ema.eval()

        psp_encoder_path = os.path.join(self.MODELS_DIR, 'psp_encoder.pt')
        self._psp_encoder = PSPEncoder(psp_encoder_path, output_size=self._size).to(self._device)
        self._psp_encoder.eval()

    def transfer_style(self, sketch_img, style_img):
        img_in = np.array(sketch_img)
        img_in_ten = cv2ten(img_in, self._device)
        img_in = cv2.resize(img_in, (self._size, self._size))

        img_style = np.array(style_img)
        img_style_ten = cv2ten(img_style, self._device)
        img_style = cv2.resize(img_style, (self._size, self._size))

        with torch.no_grad():
            sample_style = self._g_ema.get_z_embed(img_style_ten)
            sample_in = self._psp_encoder(img_in_ten)
            img_out_ten, _ = self._g_ema([sample_in], z_embed=sample_style, add_weight_index=self._add_weight_index,
                                         input_is_latent=True, return_latents=False, randomize_noise=False)
            img_out = ten2cv(img_out_ten)
        return ToPILImage()(img_out)
