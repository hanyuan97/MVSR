import numpy as np
import cv2
from utils.metrics import cal_ssim, cal_ms_ssim, psnr, to_CHW_cuda
import os
import argparse
from tqdm import tqdm
from piqa import LPIPS, PSNR, HaarPSI
import torch

# loss_fn_alex = lpips.LPIPS(net='alex')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--less", action="store_true")
    args = parser.parse_args()
    lpips_vgg = LPIPS(network='vgg').cuda()
    lpips_alex = LPIPS(network='alex').cuda()
    haar = HaarPSI().cuda()
    path = "../dataset/vimeo90k"
    test_txt = open(f"{path}/sep_testlist.txt", "r")
    test_list = [i[:-1] for i in test_txt.readlines()]
    if args.less:
        test_list = ["00002/0025"]
    lr_dir = "../dataset/vimeo90k/lr"
    hr_dir = "../dataset/vimeo90k/hr"
    log_file = open(f"experiments/bicubic.csv", "w")
    
    log_file.write("target,frame,lr_psnr,lr_ssim,lr_ms_ssim\n")
    for i in tqdm(test_list):
        target_frames = [1,2,3,4,5,6,7]
        lr_path = f"../dataset/vimeo90k/lr/{i}"
        hr_path = f"../dataset/vimeo90k/hr/{i}"
        log_file.write(f"{i}")
        for target_frame in target_frames:
            lr1 = cv2.imread(f"{lr_path}/im{target_frame}.png")[:,:,::-1]
            lr1 = cv2.resize(lr1, (448, 256))
            hr1 = cv2.imread(f"{hr_path}/im{target_frame}.png")[:,:,::-1]
            lr1_cuda = to_CHW_cuda(lr1)
            hr1_cuda = to_CHW_cuda(hr1)
            lr_psnr1 = np.mean(psnr(lr1, hr1))
            lr_ssim1 = cal_ssim(lr1_cuda, hr1_cuda)
            lr_ms_ssim1 = cal_ms_ssim(lr1_cuda, hr1_cuda)
            lr_haar1 = haar(lr1_cuda, hr1_cuda).item()
            lr_lpips_vgg1 = lpips_vgg(lr1_cuda, hr1_cuda).item()
            lr_lpips_alex1 = lpips_alex(lr1_cuda, hr1_cuda).item()        
            log_file.write(f",im{target_frame},{lr_psnr1},{lr_ssim1},{lr_ms_ssim1},{lr_haar1},{lr_lpips_alex1},{lr_lpips_vgg1}\n")
            # log_file.write(f",im{target_frame},{lr_psnr1},{lr_ssim1},{lr_ms_ssim1}\n")

    log_file.close()