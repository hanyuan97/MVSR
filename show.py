import torch
import numpy as np
from models.SRX264 import SRX264v1
import cv2
from utils.metrics import psnr, cal_ssim, cal_ms_ssim
import os
import argparse
import yaml
from tqdm import tqdm
from piqa import PSNR, SSIM, MS_SSIM, LPIPS, HaarPSI

def to_CHW_cuda(img):
    im = img.copy()
    return torch.from_numpy(im.transpose(2, 0, 1)).cuda().unsqueeze(0)/255

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch", type=int, default=8)
    parser.add_argument("-c", "--config", type=str, default="SRx264_48_l1_v1.yml")
    parser.add_argument("-old", action="store_true")
    parser.add_argument("-l", "--less", action="store_true")
    args = parser.parse_args()
    with open(f"configs/{args.config}", "r") as f:
        opt = yaml.safe_load(f)
    
    lpips_vgg = LPIPS(network='vgg').cuda()
    lpips_alex = LPIPS(network='alex').cuda()
    haar = HaarPSI().cuda()
    # pyqa_psnr = PSNR()
    # pyqa_ssim = SSIM().cuda()
    model_path = f"./experiments/{opt['name']}/weights"
    
    train_on_gpu = torch.cuda.is_available()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    path = opt['datasets']['test']['path']
    test_txt = open(opt['datasets']['test']['txt_path'], "r")
    test_list = [i[:-1] for i in test_txt.readlines()]
    if args.less:
        test_list = ["00002/0025"]
    
    lr_dir = opt['datasets']['train']['lr_dir']
    hr_dir = opt['datasets']['train']['hr_dir']
    
    if opt['network_G']['net'] == 'SRX264':
        if opt['network_G']['version'] == 'v1':
            from models.SRX264 import SRX264v1
            net = SRX264v1(scale_factor=opt['scale_factor'], maps=opt['network_G']['feature_maps'], in_nc=opt['network_G']['in_nc'])
        else:
            from models.SRX264 import SRX264v2
            net = SRX264v2(scale_factor=opt['scale_factor'], maps=opt['network_G']['feature_maps'], in_nc=opt['network_G']['in_nc'])
    net.to(device)
    if opt['gan']:
        weight_path = f"{model_path}/model_g_{args.batch}.pth"
    else:
        weight_path = f"{model_path}/model_{args.batch}.pth"
    if args.old:
        print(weight_path)
        net.load_state_dict(torch.load(weight_path))
    else:
        checkpoint = torch.load(weight_path)
        net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    for i in tqdm(test_list):
        save_paths = i.split('/')
        save_path = f"experiments/{opt['name']}/show/{save_paths[0]}"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if not os.path.exists(f"{save_path}/{save_paths[1]}"):
            os.mkdir(f"{save_path}/{save_paths[1]}")
        if not os.path.exists(f"{save_path}/{save_paths[1]}/{args.batch}"):
            os.mkdir(f"{save_path}/{save_paths[1]}/{args.batch}")
        save_path = f"{save_path}/{save_paths[1]}/{args.batch}"
        log_file = open(f"{save_path}/{opt['name']}_{args.batch}.csv", "w")
        log_file.write("target,frame,sr_psnr,sr_ssim,sr_ms_ssim,sr_lpips_alex,sr_lpips_vgg\n")
        target_frames = [1, 4, 5]
        lr_path = f"../dataset/vimeo90k/lr/{i}"
        hr_path = f"../dataset/vimeo90k/hr/{i}"
        log_file.write(f"{i}")
        for target_frame in target_frames:
            lr1 = cv2.imread(f"{lr_path}/im{target_frame}.png")[...,::-1]
            f1_lr = (lr1.astype('float')/255).transpose(2, 0, 1)
            lr2 = cv2.imread(f"{lr_path}/im{target_frame+1}.png")[...,::-1]
            f2_lr = (lr2.astype('float')/255).transpose(2, 0, 1)
            lr3 = cv2.imread(f"{lr_path}/im{target_frame+2}.png")[...,::-1]
            f3_lr = (lr3.astype('float')/255).transpose(2, 0, 1)
            if opt['mv']:
                f2_mv = (np.load(f"{lr_path}/mv{target_frame+1}.npy")).astype('float').transpose(2, 0, 1)
                f3_mv = (np.load(f"{lr_path}/mv{target_frame+2}.npy")).astype('float').transpose(2, 0, 1)
                lr = np.concatenate([f1_lr, f2_mv, f2_lr, f3_mv, f3_lr])
            else:
                lr = np.concatenate([f1_lr, f2_lr, f3_lr])
            lr = torch.from_numpy(lr).to(device, dtype=torch.float)
            lr = torch.unsqueeze(lr, 0)
            output = net(lr)
            output = output[0].cpu().detach().numpy()*255
            output = output.transpose(1, 2, 0)
            output[np.where(output > 255)] = 255
            output[np.where(output < 0)] = 0
            
            # BGR
            if target_frame != 5:
                hr1 = cv2.imread(f"{hr_path}/im{target_frame}.png")[...,::-1]
                hr1_cuda = to_CHW_cuda(hr1)
                hr2 = cv2.imread(f"{hr_path}/im{target_frame+1}.png")[...,::-1]
                hr2_cuda = to_CHW_cuda(hr2)
                sr1 = np.round(output[:, :, :3]).astype('uint8')
                sr1_cuda = to_CHW_cuda(sr1)
                sr2 = np.round(output[:, :, 3:6]).astype('uint8')
                sr2_cuda = to_CHW_cuda(sr2)
                sr_psnr1 = np.mean(psnr(sr1, hr1))
                sr_psnr2 = np.mean(psnr(sr2, hr2))
                sr_ssim1 = cal_ssim(sr1_cuda, hr1_cuda).item()
                sr_ssim2 = cal_ssim(sr2_cuda, hr2_cuda).item()
                sr_ms_ssim1 = cal_ms_ssim(sr1_cuda, hr1_cuda).item()
                sr_ms_ssim2 = cal_ms_ssim(sr2_cuda, hr2_cuda).item()
                sr_haar1 = haar(sr1_cuda, hr1_cuda).item()
                sr_haar2 = haar(sr2_cuda, hr2_cuda).item()
                sr_lpips_vgg1 = lpips_vgg(sr1_cuda, hr1_cuda).item()
                sr_lpips_vgg2 = lpips_vgg(sr2_cuda, hr2_cuda).item()
                sr_lpips_alex1 = lpips_alex(sr1_cuda, hr1_cuda).item()
                sr_lpips_alex2 = lpips_alex(sr2_cuda, hr2_cuda).item()
                log_file.write(f",im{target_frame+0},{sr_psnr1},{sr_ssim1},{sr_ms_ssim1},{sr_haar1},{sr_lpips_alex1},{sr_lpips_vgg1}\n")
                log_file.write(f",im{target_frame+1},{sr_psnr2},{sr_ssim2},{sr_ms_ssim2},{sr_haar2},{sr_lpips_alex2},{sr_lpips_vgg2}\n")
                cv2.imwrite(f"{save_path}/im{target_frame+0}.png", sr1[...,::-1])
                cv2.imwrite(f"{save_path}/im{target_frame+1}.png", sr2[...,::-1])
                
            hr3 = cv2.imread(f"{hr_path}/im{target_frame+2}.png")[...,::-1]
            hr3_cuda = to_CHW_cuda(hr3)
            sr3 = np.round(output[:, :, 6:9]).astype('uint8')
            sr3_cuda = to_CHW_cuda(sr3)
            sr_psnr3 = np.mean(psnr(sr3, hr3))
            sr_ssim3 = cal_ssim(sr3_cuda, hr3_cuda).item()
            sr_ms_ssim3 = cal_ms_ssim(sr3_cuda, hr3_cuda).item()
            sr_haar3 = haar(sr3_cuda, hr3_cuda).item()
            sr_lpips_alex3 = lpips_alex(sr3_cuda, hr3_cuda).item()
            sr_lpips_vgg3 = lpips_vgg(sr3_cuda, hr3_cuda).item()
            log_file.write(f",im{target_frame+2},{sr_psnr3},{sr_ssim3},{sr_ms_ssim3},{sr_haar3},{sr_lpips_alex3},{sr_lpips_vgg3}\n")
            cv2.imwrite(f"{save_path}/im{target_frame+2}.png", sr3[...,::-1])
        log_file.close()