import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split, Subset
from dataset import SRDataset
from model import SRX264
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

    model = SRX264(maps=args.maps)
        
    if not os.path.exists("./loss_log"):
        os.mkdir("./loss_log")
    
    if not os.path.exists("./weights"):
        os.mkdir("./weights")
    
    
    log_file = open(f"./loss_log/x264SR_{args.maps}.log", "w")

    model.to(device)
    if args.last_epoch != 0:
        model.load_state_dict(torch.load(f"./weights/{args.maps}/{args.weight}_{args.last_epoch}.pth"))
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    return model, loss_fn, optimizer, training_loader, validation_loader, log_file
    
    
def train(EPOCH, model, loss_fn, optimizer, training_loader, validation_loader, log_file, args):
    for epoch in range(args.last_epoch + 1, EPOCH+1):
        print(f"Epoch: {epoch}/{EPOCH}")
        model.train()
        train_loss = train_one_epoch(epoch)
        model.eval()
        val_loss = val_one_epoch(epoch)
        train_loss = train_loss/len(training_loader.dataset)
        val_loss = val_loss/len(validation_loader.dataset)
        print(f"Training Loss: {train_loss:.6f} \tValidation Loss: {val_loss:.6f}")
        log_file.write(f"{train_loss:.6f},{val_loss:.6f}\n")
        torch.save(model.state_dict(), f"./weights/model_{epoch}.pth")
        
    log_file.close()
    return model
    
def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.
    for data, label in tqdm(training_loader):
        data, label = data.to(device, dtype=torch.float), label.to(device, dtype=torch.float)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output*255, label*255)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss

def val_one_epoch(epoch_index):
    running_loss = 0.
    for data, label in tqdm(validation_loader):
        data, label = data.to(device, dtype=torch.float), label.to(device, dtype=torch.float)
        output = model(data)
        loss = loss_fn(output*255, label*255)
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
    parser.add_argument("--weight", type=str, default="model")
    parser.add_argument("-l", "--last_epoch", type=int, default=0)
    parser.add_argument("-w", "--workers", type=int, default=1)
    
    args = parser.parse_args()
    
    model, loss_fn, optimizer, training_loader, validation_loader, log_file = init(args)
    
    train(args.epoch, model, loss_fn, optimizer, training_loader, validation_loader, log_file, args)
    
    #if args.save:
    #    torch.save(model.state_dict(), f"./weights/{args.output_filename}")
    #    print(f"model save to: ./weights/{args.output_filename}")
    # for data, label in tqdm(validation_loader):
    #     data, label = data.to(device), label.to(device)
    #     output = model(data)
    #     print(output[1])
    #     break
    