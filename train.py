import argparse
from calendar import EPOCH
import os
import cv2
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split, Subset
from models.loss import GANLoss, VGGFeatureExtractor, MVLoss, MVLoss2
from dataset import SRDataset
from tqdm import tqdm
import yaml

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def init(args, opt):
    train_batch = opt['datasets']['train']['batch_size']
    val_batch = opt['datasets']['train']['val_batch_size']
    train_ratio = opt['datasets']['train']['train_ratio']
    use_shuffle = opt['datasets']['train']['use_shuffle']
    scale_factor = opt['scale_factor']
    has_mv = opt['mv']
    workers = args.workers
    
    train_dataset = SRDataset(scale_factor=scale_factor, datasets=opt['datasets'], has_mv=has_mv)
    # valid_dataset = SRDataset(filename="sep_testlist.txt")
    train_num = int(len(train_dataset) * train_ratio)
    val_num = len(train_dataset) - train_num
    
    # train_set, val_set = random_split(train_dataset, [train_num, val_num])
    training_loader = DataLoader(dataset=Subset(train_dataset, range(train_num)),
                            batch_size=train_batch,
                            shuffle=use_shuffle,
                            num_workers=workers,
                            pin_memory=True)

    validation_loader = DataLoader(dataset=Subset(train_dataset, range(train_num, len(train_dataset))),
                            batch_size=val_batch,
                            shuffle=use_shuffle,
                            num_workers=workers,
                            pin_memory=True)

    if opt['network_G']['net'] == 'SRX264':
        if opt['network_G']['version'] == 'v1':
            from models.SRX264 import SRX264v1
            net = SRX264v1(scale_factor=opt['scale_factor'], maps=opt['network_G']['feature_maps'], in_nc=opt['network_G']['in_nc'])
        else:
            from models.SRX264 import SRX264v2
            net = SRX264v2(scale_factor=opt['scale_factor'], maps=opt['network_G']['feature_maps'], in_nc=opt['network_G']['in_nc'])
    elif opt['network_G']['net'] == 'RRDBNet':
        from models.RRDBNet_arch import RRDBNet
        net = RRDBNet(in_nc=opt['network_G']['in_nc'], out_nc=opt['network_G']['out_nc'], nf=opt['network_G']['nf'], nb=opt['network_G']['nb'])
    
    net_f = VGGFeatureExtractor(feature_layer=34, use_bn=False,
                                use_input_norm=True, device=device)
    net_f.eval()
    if not os.path.exists(f"./experiments/{opt['name']}"):
        os.mkdir(f"./experiments/{opt['name']}")
        os.mkdir(f"./experiments/{opt['name']}/weights")
        os.mkdir(f"./experiments/{opt['name']}/show")
        os.mkdir(f"./experiments/{opt['name']}/test")
    
    log_file = open(f"./experiments/{opt['name']}/loss.log", "w")
    model_path = f"./experiments/{opt['name']}/weights"
    net.to(device)
    net_f.to(device)
    optimizer_g = optim.Adam(net.parameters(), lr=opt['train']['lr_G'], betas=(0.9, 0.999))
    if args.resume != 0:
        if opt['gan']:
            weight_path = f"{model_path}/model_g_{args.resume}.pth"
        else:
            weight_path = f"{model_path}/model_{args.resume}.pth"
        if args.old:
            print(weight_path)
            net.load_state_dict(torch.load(weight_path))
        else:
            checkpoint = torch.load(weight_path)
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer_g.load_state_dict(checkpoint['optimizer_state_dict'])
            
            
    pixel_loss_fn = nn.L1Loss().to(device) if opt['train']['pixel_criterion'] == 'l1' else nn.MSELoss().to(device)
    feature_loss_fn = nn.L1Loss().to(device) if opt['train']['feature_criterion'] == 'l1' else nn.MSELoss().to(device)
    
    return net, net_f, pixel_loss_fn, feature_loss_fn, optimizer_g, training_loader, validation_loader, log_file

def init_d(args, opt):
    from models.Discriminator import NLayerDiscriminator
    net_d = NLayerDiscriminator(3)
    net_d.to(device)
    optimizer_d = optim.Adam(net_d.parameters(), lr=opt['train']['lr_D'], betas=(0.9, 0.999))
    if args.resume != 0:
        if args.old:
            net_d.load_state_dict(torch.load(f"{model_path}/model_d_{args.resume}.pth"))
        else:
            checkpoint = torch.load(f"{model_path}/model_d_{args.resume}.pth")
            net_d.load_state_dict(checkpoint['model_state_dict'])
            optimizer_d.load_state_dict(checkpoint['optimizer_state_dict'])
        
    cri_gan = GANLoss('ragan', 1.0, 0.0).to(device)
    return net_d, optimizer_d, cri_gan
    
def train(training_loader, validation_loader, log_file, args, opt):
    EPOCH = opt['train']['epoch']
    multiple = opt['train']['multiple']
    model_path = f"./experiments/{opt['name']}/weights"
    gan = opt['gan']
    use_mv_loss = opt['train']['mv_loss']
    for epoch in range(args.resume + 1, EPOCH+1):
        print(f"Epoch: {epoch}/{EPOCH}")
        net.train()
        if gan:
            net_d.train()
            
        train_loss_g, train_loss_pix, train_loss_fea, train_loss_mv, train_loss_gan, train_loss_d = train_one_epoch(epoch, gan, use_mv_loss, multiple)
        net.eval()
        if gan:
            net_d.eval()
            train_loss_gan = train_loss_gan / len(training_loader.dataset)
        val_loss_g, val_loss_d = val_one_epoch(epoch, gan, multiple)
        train_loss_g = train_loss_g / len(training_loader.dataset)
        train_loss_pix = train_loss_pix / len(training_loader.dataset)
        train_loss_fea = train_loss_fea / len(training_loader.dataset)
        train_loss_mv = train_loss_mv / len(training_loader.dataset)
        train_loss_d = train_loss_d / len(training_loader.dataset)
        val_loss_g = val_loss_g / len(validation_loader.dataset)
        val_loss_d = val_loss_d / len(validation_loader.dataset)
        save_model_path = f"./{model_path}/model_{epoch}.pth"
        if gan:
            save_model_path = f"./{model_path}/model_g_{epoch}.pth"
            torch.save({'epoch': epoch, 
                        'model_state_dict': net_d.state_dict(),
                        'optimizer_state_dict': optimizer_d.state_dict(),
                        'train_loss': train_loss_d,
                        'val_loss': val_loss_d
                        }, f"./{model_path}/model_d_{epoch}.pth")
            print(f"Training loss: pix: {train_loss_pix:.4f}, mv: {train_loss_mv:.4f}, fea: {train_loss_fea:.4f}, gan: {train_loss_gan:.4f}, total: {train_loss_g:.4f} \tValidation Loss: {val_loss_g:.4f}, {val_loss_d:.6f}")
            
        else:
            print(f"Training loss: pix: {train_loss_pix:.4f}, mv: {train_loss_mv:.4f}, fea: {train_loss_fea:.4f}, total: {train_loss_g:.4f} \tValidation Loss: {val_loss_g:.4f}")
        
        torch.save({'epoch': epoch, 
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer_g.state_dict(),
            'train_loss': train_loss_g,
            'pix_loss': train_loss_pix,
            'fea_loss': train_loss_fea,
            'mv_loss': train_loss_mv,
            'gan_loss': train_loss_gan,
            'val_loss': val_loss_g
            }, save_model_path)
        log_file.write(f"{train_loss_g},{train_loss_pix},{train_loss_fea},{train_loss_mv},{train_loss_gan},{train_loss_d},{val_loss_g},{val_loss_d}\n")
    log_file.close()
    return net
    
def train_one_epoch(epoch_index, is_gan=False, use_mv_loss=False, mix_loss=False, multiple=1):
    running_loss_g = 0.
    running_loss_d = 0.
    running_loss_pix = 0.
    running_loss_fea = 0.
    running_loss_gan = 0.
    running_loss_mv = 0.
    last_loss = 0.
    for data, real_H in tqdm(training_loader):
        data, real_H = data.to(device, dtype=torch.float), real_H.to(device, dtype=torch.float)
        optimizer_g.zero_grad()
        if is_gan:
            optimizer_d.zero_grad()
            for p in net_d.parameters():
                p.requires_grad = False
        fake_H = net(data)
        # if epoch_index > d_init_iter:
        if mix_loss:
            loss_pix, loss_mv = mix_loss_fn(data, fake_H*multiple, real_H*multiple)
            loss_g = mv_w * (loss_pix + loss_mv)
            running_loss_mv += loss_mv.item()
        else:
            loss_pix = pixel_loss_fn(fake_H*multiple, real_H*multiple)
            loss_g = pix_w * loss_pix
            if use_mv_loss:
                loss_mv = mv_loss_fn(data, fake_H*multiple, real_H*multiple)
                loss_g += mv_w * loss_mv
                running_loss_mv += loss_mv.item()
        
        real_fea = net_f(real_H.view(-1, 3, 256, 448)).detach()
        fake_fea = net_f(fake_H.view(-1, 3, 256, 448))        
        loss_fea = feature_loss_fn(fake_fea, real_fea)
        loss_g += loss_fea
        if is_gan:
            pred_g_fake = net_d(fake_H.view(-1, 3, 256, 448))
            pred_d_real = net_d(real_H.view(-1, 3, 256, 448)).detach()
            loss_gan = (cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                        cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
            running_loss_gan += loss_gan.item()
            loss_g += gan_w * loss_gan
        
        loss_g.backward()
        optimizer_g.step()
        running_loss_pix += loss_pix.item()
        running_loss_fea += loss_fea.item()
        running_loss_g += loss_g.item()
        if is_gan:
            for p in net_d.parameters():
                p.requires_grad = True
            pred_d_real = net_d(real_H.view(-1, 3, 256, 448))
            pred_d_fake = net_d(fake_H.detach().view(-1, 3, 256, 448))
            l_d_real = cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
            l_d_fake = cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
            loss_d = (l_d_real + l_d_fake) / 2
            loss_d.backward()
            optimizer_d.step()
            running_loss_d += loss_d.item()
    
    return running_loss_g, running_loss_pix, running_loss_fea, running_loss_mv, running_loss_gan, running_loss_d
    
def val_one_epoch(epoch_index, is_gan=False, multiple=1):
    running_loss_g = 0.
    running_loss_d = 0.
    for data, real_H in tqdm(validation_loader):
        data, real_H = data.to(device, dtype=torch.float), real_H.to(device, dtype=torch.float)
        with torch.no_grad():
            fake_H = net(data)
            if is_gan:
                pred_d_real = net_d(real_H.view(-1, 3, 256, 448))
                pred_d_fake = net_d(fake_H.detach().view(-1, 3, 256, 448))
        if is_gan:    
            l_d_real = cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
            l_d_fake = cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
            loss_d = (l_d_real + l_d_fake) / 2
            running_loss_d += loss_d.item()
            
        loss = pixel_loss_fn(fake_H*multiple, real_H*multiple)
        running_loss_g += loss.item()
    return running_loss_g, running_loss_d

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str)
    parser.add_argument("-r", "--resume", type=int, default=0)
    parser.add_argument("-w", "--workers", type=int, default=1)
    parser.add_argument("-old", "--old", action="store_true")
    args = parser.parse_args()
    with open(f"configs/{args.config}", "r") as f:
        opt = yaml.safe_load(f)
    d_init_iter = 0
    model_path = f"./experiments/{opt['name']}/weights"
    pix_w = opt['train']['pixel_weight']
    gan_w = opt['train']['gan_weight']
    mv_w = opt['train']['mv_weight']
    mv_loss_fn = MVLoss().to(device)
    mix_loss_fn = MVLoss2().to(device)
    net, net_f, pixel_loss_fn, feature_loss_fn, optimizer_g, training_loader, validation_loader, log_file = init(args, opt)
    if opt['gan']:
        net_d, optimizer_d, cri_gan = init_d(args, opt)
    
    train(training_loader, validation_loader, log_file, args, opt)
    