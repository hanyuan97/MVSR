import numpy as np
import cv2
from utils.metrics import convert_cal_ssim, convert_cal_ms_ssim, psnr
import os
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--less", action="store_true")
    args = parser.parse_args()
    path = "../dataset/vimeo90k"
    test_txt = open(f"{path}/sep_testlist.txt", "r")
    test_list = [i[:-1] for i in test_txt.readlines()]
    if args.less:
        test_list = test_list[:5]
    lr_dir = "../dataset/vimeo90k/lr"
    hr_dir = "../dataset/vimeo90k/hr"
    log_file = open(f"result/bicubic.csv", "w")
    # test_set = load_file("data", filename)
    # print(len(dctDataset))
    log_file.write("target,frame,lr_psnr,lr_ssim,lr_ms_ssim\n")
    for i in tqdm(test_list):
        target_frames = [1,2,3,4,5,6,7]
        lr_path = f"../dataset/vimeo90k/lr/{i}"
        hr_path = f"../dataset/vimeo90k/hr/{i}"
        log_file.write(f"{i}")
        for target_frame in target_frames:
            lr1 = cv2.imread(f"{lr_path}/im{target_frame}.png")
            lr1 = cv2.resize(lr1, (448, 256))
            hr1 = cv2.imread(f"{hr_path}/im{target_frame}.png")
            lr_psnr1 = np.mean(psnr(hr1, lr1))
            lr_ssim1 = convert_cal_ssim(hr1, lr1)
            lr_ms_ssim1 = convert_cal_ms_ssim(hr1, lr1)            
            log_file.write(f",im{target_frame},{lr_psnr1},{lr_ssim1},{lr_ms_ssim1}\n")

    log_file.close()
            