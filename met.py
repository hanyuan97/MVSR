import torch
import numpy as np
from models.SRX264 import SRX264v1
import matplotlib.pyplot as plt
import cv2
from utils.jpeg import JPEG
from utils.metrics import convert_cal_ssim, convert_cal_ms_ssim, psnr
import os
import argparse

if __name__ == "__main__":
    hr_path = "../dataset/vimeo90k/hr/00002/0001"
    lr_path = "../dataset/vimeo90k/lr/00002/0001"
    hr1 = cv2.imread(f"{hr_path}/im1.png")
    hr2 = cv2.imread(f"{hr_path}/im2.png")
    hr3 = cv2.imread(f"{hr_path}/im3.png")
    lr1 = cv2.imread(f"{lr_path}/im1.png")
    lr2 = cv2.imread(f"{lr_path}/im2.png")
    lr3 = cv2.imread(f"{lr_path}/im3.png")
    lr1 = cv2.resize(lr1, (448, 256))
    lr2 = cv2.resize(lr2, (448, 256))
    lr3 = cv2.resize(lr3, (448, 256))
    print(np.mean(psnr(hr1, lr1)))
    print(np.mean(psnr(hr2, lr2)))
    print(np.mean(psnr(hr3, lr3)))
    # sr1 = cv2.imread("test1_model_96_80.png")
    # sr2 = cv2.imread("test2_model_96_80.png")
    # sr3 = cv2.imread("test3_model_96_80.png")
    # print(np.mean(psnr(hr1, sr1)))
    # print(np.mean(psnr(hr2, sr2)))
    # print(np.mean(psnr(hr3, sr3)))
    sr1 = cv2.imread("test1_model_128_13.png")
    sr2 = cv2.imread("test2_model_128_13.png")
    sr3 = cv2.imread("test3_model_128_13.png")
    print(np.mean(psnr(hr1, sr1)))
    print(np.mean(psnr(hr2, sr2)))
    print(np.mean(psnr(hr3, sr3)))
    # sr1 = cv2.imread("test1_model_192_80.png")
    # sr2 = cv2.imread("test2_model_192_80.png")
    # sr3 = cv2.imread("test3_model_192_80.png")
    # print(np.mean(psnr(hr1, sr1)))
    # print(np.mean(psnr(hr2, sr2)))
    # print(np.mean(psnr(hr3, sr3)))