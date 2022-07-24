import torch
import numpy as np
from model import SRX264
import matplotlib.pyplot as plt
import cv2
from utils.jpeg import JPEG
from utils.metrics import cal_ssim, cal_ms_ssim, psnr
import os
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch", type=int, default=8)
    parser.add_argument("-w", "--weight", type=str, default="model")
    parser.add_argument("-m", "--maps", type=int, default=96)
    parser.add_argument("-l", "--less", action="store_true")
    args = parser.parse_args()
    
    weight = args.weight
    
    train_on_gpu = torch.cuda.is_available()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    
    path = "../dataset/vimeo90k"
    test_txt = open(f"{path}/sep_testlist.txt", "r")
    test_list = [i[:-1] for i in test_txt.readlines()]
    if args.less:
        test_list = ["00002/0025"]
    lr_dir = "../dataset/vimeo90k/lr"
    hr_dir = "../dataset/vimeo90k/hr"
    
    model = SRX264(maps=args.maps)

    model.to(device)
    model.load_state_dict(torch.load(f"./weights/{weight}.pth"))
    model.eval()

    for i in tqdm(test_list):
        save_paths = i.split('/')
        save_path = f"show/{save_paths[0]}"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if not os.path.exists(f"{save_path}/{save_paths[1]}"):
            os.mkdir(f"{save_path}/{save_paths[1]}")
        if not os.path.exists(f"{save_path}/{save_paths[1]}/{args.maps}"):
            os.mkdir(f"{save_path}/{save_paths[1]}/{args.maps}")
        log_file = open(f"show/{i}/{args.maps}/{args.batch}_{args.maps}.csv", "w")
        log_file.write("target,frame,sr_psnr,sr_ssim,sr_ms_ssim\n")
        target_frames = [1, 4, 5]
        lr_path = f"../dataset/vimeo90k/lr/{i}"
        hr_path = f"../dataset/vimeo90k/hr/{i}"
        log_file.write(f"{i}")
        for target_frame in target_frames:
            lr1 = cv2.imread(f"{lr_path}/im{target_frame}.png")
            f1_lr = (lr1.astype('float')/255).transpose(2, 0, 1)
            f2_mv = (np.load(f"{lr_path}/mv{target_frame+1}.npy")).astype('float').transpose(2, 0, 1)
            lr2 = cv2.imread(f"{lr_path}/im{target_frame+1}.png")
            f2_lr = (lr2.astype('float')/255).transpose(2, 0, 1)
            f3_mv = (np.load(f"{lr_path}/mv{target_frame+2}.npy")).astype('float').transpose(2, 0, 1)
            lr3 = cv2.imread(f"{lr_path}/im{target_frame+2}.png")
            f3_lr = (lr3.astype('float')/255).transpose(2, 0, 1)
            lr = np.concatenate([f1_lr, f2_mv, f2_lr, f3_mv, f3_lr])
            lr = torch.from_numpy(lr).to(device, dtype=torch.float)
            lr = torch.unsqueeze(lr, 0)
            opt = model(lr)
            opt = opt[0].cpu().detach().numpy()*255
            opt = opt.transpose(1, 2, 0)
            
            # BGR
            if target_frame != 5:
                hr1 = cv2.imread(f"{hr_path}/im{target_frame}.png")
                hr2 = cv2.imread(f"{hr_path}/im{target_frame+1}.png")
                sr1 = np.round(opt[:, :, :3]).astype('uint8')
                sr2 = np.round(opt[:, :, 3:6]).astype('uint8')
                sr_psnr1 = np.mean(psnr(hr1, sr1))
                sr_psnr2 = np.mean(psnr(hr2, sr2))
                sr_ssim1 = cal_ssim(hr1, sr1)
                sr_ssim2 = cal_ssim(hr2, sr2)
                sr_ms_ssim1 = cal_ms_ssim(hr1, sr1)
                sr_ms_ssim2 = cal_ms_ssim(hr2, sr2)
                log_file.write(f",im{target_frame+0},{sr_psnr1},{sr_ssim1},{sr_ms_ssim1}\n")
                log_file.write(f",im{target_frame+1},{sr_psnr2},{sr_ssim2},{sr_ms_ssim2}\n")
                cv2.imwrite(f"show/{i}/{args.maps}/im{target_frame+0}.png", sr1)
                cv2.imwrite(f"show/{i}/{args.maps}/im{target_frame+1}.png", sr2)
                
            hr3 = cv2.imread(f"{hr_path}/im{target_frame+2}.png")
            sr3 = np.round(opt[:, :, 6:9]).astype('uint8')
            sr_psnr3 = np.mean(psnr(hr3, sr3))
            sr_ssim3 = cal_ssim(hr3, sr3)
            sr_ms_ssim3 = cal_ms_ssim(hr3, sr3)
            log_file.write(f",im{target_frame+2},{sr_psnr3},{sr_ssim3},{sr_ms_ssim3}\n")
            cv2.imwrite(f"show/{i}/{args.maps}/im{target_frame+2}.png", sr3)
        log_file.close()