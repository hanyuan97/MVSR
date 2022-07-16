import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split, Subset
from loss import GANLoss, VGGFeatureExtractor
from dataset import SRDataset
from model import SRX264, NLayerDiscriminator
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def init(args):
    EPOCH, train_batch, val_batch, train_ratio, workers = args.epoch, args.batch, args.val_batch, args.train_ratio, args.workers
    train_dataset = SRDataset()
    # valid_dataset = SRDataset(filename="sep_testlist.txt")
    train_num = int(len(train_dataset) * args.train_ratio)
    val_num = len(train_dataset) - train_num
    
    # train_set, val_set = random_split(train_dataset, [train_num, val_num])
    training_loader = DataLoader(dataset=Subset(train_dataset, range(train_num)),
                            batch_size=train_batch,
                            shuffle=True,
                            num_workers=workers,
                            pin_memory=True)

    validation_loader = DataLoader(dataset=Subset(train_dataset, range(train_num, len(train_dataset))),
                            batch_size=val_batch,
                            shuffle=True,
                            num_workers=workers,
                            pin_memory=True)

    net = SRX264(maps=args.maps)
    net_d = NLayerDiscriminator(3)
    net_f = VGGFeatureExtractor(feature_layer=34, use_bn=False,
                                use_input_norm=True, device=device)
    net_f.eval()
    if not os.path.exists("./loss_log"):
        os.mkdir("./loss_log")
    
    if not os.path.exists("./weights"):
        os.mkdir("./weights")
    
    
    log_file = open(f"./loss_log/x264SR_{args.maps}.log", "w")
    cri_gan = GANLoss('ragan', 1.0, 0.0).to(device)
    net.to(device)
    net_d.to(device)
    net_f.to(device)
    
    if args.resume != 0:
        model_path = f"./weights"
        if args.gan:
            model_path += "/gan"
        model_path += f"/{args.maps}"
        if args.gan:
            net.load_state_dict(torch.load(f"{model_path}/{args.weight}_g_{args.resume}.pth"))
            net_d.load_state_dict(torch.load(f"{model_path}/{args.weight}_d_{args.resume}.pth"))
        else:
            net.load_state_dict(torch.load(f"{model_path}/{args.weight}_{args.resume}.pth"))
            
    loss_fn_g = nn.L1Loss().to(device)
    loss_fn_d = nn.L1Loss().to(device)
    optimizer_g = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999))
    optimizer_d = optim.Adam(net_d.parameters(), lr=1e-4, betas=(0.9, 0.999))
    return net, net_d, net_f, cri_gan, loss_fn_g, loss_fn_d, optimizer_g, optimizer_d, training_loader, validation_loader, log_file
    
    
def train(EPOCH, net, loss_fn_g, loss_fn_d, optimizer_g, optimizer_d, training_loader, validation_loader, log_file, args):
    for epoch in range(args.resume + 1, EPOCH+1):
        print(f"Epoch: {epoch}/{EPOCH}")
        net.train()
        train_loss_g, train_loss_d = train_one_epoch(epoch)
        net.eval()
        val_loss = val_one_epoch(epoch)
        train_loss_g = train_loss_g/len(training_loader.dataset)
        train_loss_d = train_loss_d/len(training_loader.dataset)
        val_loss = val_loss/len(validation_loader.dataset)
        print(f"Training Loss: {train_loss_g:.6f}, {train_loss_d:.6f} \tValidation Loss: {val_loss:.6f}")
        log_file.write(f"{train_loss_g:.6f},{train_loss_d:.6f},{val_loss:.6f}\n")
        torch.save(net.state_dict(), f"./weights/model_g_{epoch}.pth")
        torch.save(net_d.state_dict(), f"./weights/model_d_{epoch}.pth")
        
    log_file.close()
    return net
    
def train_one_epoch(epoch_index):
    running_loss_g = 0.
    running_loss_d = 0.
    last_loss = 0.
    for data, real_H in tqdm(training_loader):
        data, real_H = data.to(device, dtype=torch.float), real_H.to(device, dtype=torch.float)
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        fake_H = net(data)
        # if epoch_index > d_init_iter:
        for p in net_d.parameters():
            p.requires_grad = False
        
        loss_g = l1_w * loss_fn_g(fake_H, real_H)
        real_fea = net_f(real_H.view(-1, 3, 256, 448)).detach()
        fake_fea = net_f(fake_H.view(-1, 3, 256, 448))
        loss_g += loss_fn_g(fake_fea, real_fea)
        
        pred_g_fake = net_d(fake_H.view(-1, 3, 256, 448))
        pred_d_real = net_d(real_H.view(-1, 3, 256, 448)).detach()
        loss_g += gan_w * (
                cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
        
        loss_g.backward()
        optimizer_g.step()
        running_loss_g += loss_g.item()
        
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
    return running_loss_g, running_loss_d

def val_one_epoch(epoch_index):
    running_loss = 0.
    
    for data, label in tqdm(validation_loader):
        data, label = data.to(device, dtype=torch.float), label.to(device, dtype=torch.float)
        with torch.no_grad():
            output = net(data)
        loss = loss_fn_d(output*255, label*255)
        running_loss += loss.item()
    return running_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epoch", type=int, default=20)
    parser.add_argument("-b", "--batch", type=int, default=64)
    parser.add_argument("-vb", "--val_batch", type=int, default=32)
    parser.add_argument("-tr", "--train_ratio", type=float, default=0.8)
    parser.add_argument("-o", "--output_filename", type=str, default="model.pth")
    parser.add_argument("-m", "--maps", type=int, default=96)
    parser.add_argument("-s", "--save", action="store_true")
    parser.add_argument("-g", "--gan", action="store_true")
    parser.add_argument("--weight", type=str, default="model")
    parser.add_argument("-r", "--resume", type=int, default=0)
    parser.add_argument("-w", "--workers", type=int, default=1)
    
    args = parser.parse_args()
    d_init_iter = 0
    l1_w = 1e-2
    gan_w = 5e-3
    net, net_d, net_f, cri_gan, loss_fn_g, loss_fn_d, optimizer_g, optimizer_d, training_loader, validation_loader, log_file = init(args)
    
    train(args.epoch, net, loss_fn_g, loss_fn_d, optimizer_g, optimizer_d, training_loader, validation_loader, log_file, args)
    
    #if args.save:
    #    torch.save(model.state_dict(), f"./weights/{args.output_filename}")
    #    print(f"model save to: ./weights/{args.output_filename}")
    # for data, label in tqdm(validation_loader):
    #     data, label = data.to(device), label.to(device)
    #     output = model(data)
    #     print(output[1])
    #     break
    