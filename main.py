import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

from datasets import MITDataSet
from engine import train_one_epoch, evaluate
from utils import save_model, train_record, load_model, create_logs, create_model, plot_history

def get_args_parser():

    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--lr", default=1e-2, type=float)

    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--save_ckpt_freq", type=int, default=10)

    parser.add_argument("--model", type=str, default="lstm")
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--classes_num", type=int, default=5)

    return parser


def main(args):

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    epochs = args.epochs

    model = create_model(args.model, args.hidden_size, args.classes_num).to(device)

    save_path, record_path = create_logs(args, model)

    train_dataset = MITDataSet(filepath="preprocessing/train.txt")
    val_dataset = MITDataSet(filepath="preprocessing/val.txt")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, min_lr=1e-6, factor=0.1)

    start_epoch = 1
    best_acc = 0

    if args.checkpoint:
        start_epoch, best_acc = load_model(args.checkpoint, optimizer, model, device, record_path)

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    for epoch in range(start_epoch, epochs + 1):

        start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, criterion, train_dataloader, optimizer, epoch,
                                                epochs, device)
        val_loss, val_acc = evaluate(model, criterion, val_dataloader, epoch, epochs, device)
        end_time = time.time()

        lr_scheduler.step(val_loss)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        train_record(epoch, epochs, train_loss, train_acc, val_loss, val_acc, end_time - start_time, record_path)

        if epoch % args.save_ckpt_freq == 0 or epoch == epochs:
            save_model(args, epoch, model, optimizer, val_acc, train_loss, best_acc, save_path, record_path)

        if val_acc > best_acc:
            save_model(args, epoch, model, optimizer, val_acc, train_loss, best_acc, save_path, record_path, best_model=True)
            best_acc = val_acc

    plot_history(train_loss_list, train_acc_list, val_loss_list, val_acc_list, epochs, record_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)