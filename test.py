import torch
import numpy as np
from model import SRX264
import matplotlib.pyplot as plt
import cv2
from utils.jpeg import JPEG
from utils.metrics import cal_ssim, cal_ms_ssim, psnr
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch", type=int, default=8)
    parser.add_argument("-w", "--weight", type=str, default="model")
    parser.add_argument("-m", "--maps", type=int, default=96)
    parser.add_argument("-l", "--lr", action="store_true")
    args = parser.parse_args()
    
    weight = args.weight
    
    # save_path = f"./jpeg_result/color_q{q_str}_s{args.sample}_dec"
    # crop_size = 8 if args.sample == "444" else 16
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # log_file = open(f"{save_path}/color_q{q_str}_s{args.sample}_dec_log.csv", "w")
    train_on_gpu = torch.cuda.is_available()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # dctDataset = DCTDataset(filename=filename)
    # test_loader = DataLoader(dataset=dctDataset,
    #                         batch_size=3,
    #                         shuffle=True,
    #                         num_workers=8,
    #                         pin_memory=True)

    model = SRX264(maps=args.maps)

    model.to(device)
    model.load_state_dict(torch.load(f"./weights/{weight}.pth"))
    model.eval()
    # test_set = load_file("data", filename)
    # print(len(dctDataset))
    target_frame = 1
    lr_path = "../dataset/vimeo90k/lr/00002/0001"
    f1_lr = (cv2.imread(f"{lr_path}/im{target_frame}.png").astype('float')/255).transpose(2, 0, 1)
    f2_mv = (np.load(f"{lr_path}/mv{target_frame+1}.npy")).astype('float').transpose(2, 0, 1)
    f2_lr = (cv2.imread(f"{lr_path}/im{target_frame+1}.png").astype('float')/255).transpose(2, 0, 1)
    f3_mv = (np.load(f"{lr_path}/mv{target_frame+2}.npy")).astype('float').transpose(2, 0, 1)
    f3_lr = (cv2.imread(f"{lr_path}/im{target_frame+2}.png").astype('float')/255).transpose(2, 0, 1)
    lr = np.concatenate([f1_lr, f2_mv, f2_lr, f3_mv, f3_lr])
    lr = torch.from_numpy(lr).to(device, dtype=torch.float)
    lr = torch.unsqueeze(lr, 0)
    opt = model(lr)
    opt = opt[0].cpu().detach().numpy()*255
    opt = opt.transpose(1, 2, 0)
    cv2.imwrite("test1.png", opt[:, :, :3])
    cv2.imwrite("test2.png", opt[:, :, 3:6])
    cv2.imwrite("test3.png", opt[:, :, 6:9])