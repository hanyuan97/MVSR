import torch
import numpy as np
from models.SRX264 import SRX264v1
import cv2
from utils.metrics import psnr, cal_ssim, cal_ms_ssim, to_CHW_cuda
import os
import time
import argparse
import yaml
from tqdm import tqdm
from piqa import LPIPS, PSNR, HaarPSI

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch", type=int, default=10)
    parser.add_argument("-c", "--config", type=str, default="SRx264_48_l1_v1.yml")
    parser.add_argument("-old", action="store_true")
    parser.add_argument("-l", "--less", action="store_true")
    args = parser.parse_args()
    with open(f"configs/{args.config}", "r") as f:
        opt = yaml.safe_load(f)
    
    lpips_vgg = LPIPS(network='vgg').cuda()
    lpips_alex = LPIPS(network='alex').cuda()
    haar = HaarPSI().cuda()
    # psnr = PSNR()
    # ssim = SSIM().cuda()
    # ms_psnr = MS_SSIM().cuda()
    # pyqa_psnr = PSNR().cuda()
    # pyqa_ssim = SSIM().cuda()
    model_path = f"./experiments/{opt['name']}/weights"
    
    train_on_gpu = torch.cuda.is_available()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    path = opt['datasets']['test']['path']
    test_txt = open(opt['datasets']['test']['txt_path'], "r")
    test_list = [i[:-1] for i in test_txt.readlines()]
    
    if args.less:
        # test_list = test_list[:5]
        test_list = ["00002/0025"]
    
    lr_dir = opt['datasets']['train']['lr_dir']
    hr_dir = opt['datasets']['train']['hr_dir']
    
    log_file = open(f"experiments/{opt['name']}/test/{opt['name']}_{args.batch}.csv", "w")
    
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
    net(torch.rand((1, opt['network_G']['in_nc'], 128, 224)).cuda())
    total_psnr = 0.0
    total_ssim = 0.0
    total_ms_ssim = 0.0
    total_haar = 0.0
    total_lpips_alex = 0.0
    total_lpips_vgg = 0.0
    total_exec_time = 0.0
    # test_set = load_file("data", filename)
    # print(len(dctDataset))
    log_file.write("target,frame,sr_psnr,sr_ssim,sr_ms_ssim,sr_haar,sr_lpips_alex,sr_lpips_vgg,exec_time\n")
    for i in tqdm(test_list):
        # save_paths = i.split('/')
        # save_path = f"experiments/{opt['name']}/{save_paths[0]}"
        # if not os.path.exists(save_path):
        #     os.mkdir(save_path)
        # if not os.path.exists(f"{save_path}/{save_paths[1]}"):
        #     os.mkdir(f"{save_path}/{save_paths[1]}")
        # if not os.path.exists(f"{save_path}/{save_paths[1]}/{args.batch}"):
        #     os.mkdir(f"{save_path}/{save_paths[1]}/{args.batch}")
        # save_path = f"{save_path}/{save_paths[1]}/{args.batch}"

        target_frames = [1, 4, 5]
        lr_path = f"{lr_dir}/{i}"
        hr_path = f"{hr_dir}/{i}"
        log_file.write(f"{i}")
        for target_frame in target_frames:
            t0 = time.time()
            lr1 = cv2.imread(f"{lr_path}/im{target_frame}.png")[...,::-1]
            lr2 = cv2.imread(f"{lr_path}/im{target_frame+1}.png")[...,::-1]
            lr3 = cv2.imread(f"{lr_path}/im{target_frame+2}.png")[...,::-1]
            f1_lr = (lr1.astype('float')/255).transpose(2, 0, 1)
            f2_lr = (lr2.astype('float')/255).transpose(2, 0, 1)
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
            t1 = time.time()
            exec_time = t1-t0
            # BGR
            if target_frame != 5:
                hr1 = cv2.imread(f"{hr_path}/im{target_frame}.png")[...,::-1]
                hr2 = cv2.imread(f"{hr_path}/im{target_frame+1}.png")[...,::-1]
                hr1_cuda = to_CHW_cuda(hr1)
                hr2_cuda = to_CHW_cuda(hr2)
                sr1 = np.round(output[:, :, :3]).astype('uint8')
                sr2 = np.round(output[:, :, 3:6]).astype('uint8')
                sr1_cuda = to_CHW_cuda(sr1)
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
                total_psnr += sr_psnr1+sr_psnr2
                total_ssim += sr_ssim1+sr_ssim2
                total_ms_ssim += sr_ms_ssim1+sr_ms_ssim2
                total_haar += sr_haar1+sr_haar2
                total_lpips_alex += sr_lpips_alex1+sr_lpips_alex2
                total_lpips_vgg += sr_lpips_vgg1+sr_lpips_vgg2
                log_file.write(f",im{target_frame+0},{sr_psnr1},{sr_ssim1},{sr_ms_ssim1},{sr_haar1},{sr_lpips_alex1},{sr_lpips_vgg1}\n")
                log_file.write(f",im{target_frame+1},{sr_psnr2},{sr_ssim2},{sr_ms_ssim2},{sr_haar2},{sr_lpips_alex2},{sr_lpips_vgg2}\n")
                
            
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
            total_psnr += sr_psnr3
            total_ssim += sr_ssim3
            total_ms_ssim += sr_ms_ssim3
            total_haar += sr_haar3
            total_lpips_alex += sr_lpips_alex3
            total_lpips_vgg += sr_lpips_vgg3
            log_file.write(f",im{target_frame+2},{sr_psnr3},{sr_ssim3},{sr_ms_ssim3},{sr_haar3},{sr_lpips_alex3},{sr_lpips_vgg3},{exec_time}\n")
    count = len(test_list)*7
    log_file.write(f",,{total_psnr/count},{total_ssim/count},{total_ms_ssim/count},{total_haar/count},{total_lpips_alex/count},{total_lpips_vgg/count},{total_exec_time/count}\n")
    log_file.close()