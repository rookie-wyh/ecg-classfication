import torch
from tqdm import tqdm
import sys


def train_one_epoch(model, criterion, dataloader, optimizer, lr_scheduler, epoch, epochs, device):
    model.train()
    accu_corrects = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    dataloader = tqdm(dataloader, file=sys.stdout)
    sample_num = 0
    for step, (seqs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        seqs = seqs.to(device)
        labels = labels.to(device)
        sample_num += seqs.shape[0]
        pred = model(seqs)
        pred_classes = torch.argmax(pred, dim=1)
        accu_corrects += torch.sum(pred_classes == labels)
        loss = criterion(pred, labels)
        loss.backward()
        accu_loss += loss.item()
        dataloader.desc = "[train epoch {} / {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch,
            epochs,
            accu_loss.item() / (step + 1),
            accu_corrects.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print("WARNING: non-finite loss, ending training ", loss)
            sys.exit(1)

        optimizer.step()
    lr_scheduler.step(accu_loss.item() / (step + 1))

    return accu_loss.item() / (step + 1), accu_corrects.item() / sample_num

@torch.no_grad()
def evaluate(model, criterion, dataloader, epoch, epochs, device):
    model.eval()
    accu_corrects = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    dataloader = tqdm(dataloader, file=sys.stdout)
    sample_num = 0
    for step, (seqs, labels) in enumerate(dataloader):
        seqs = seqs.to(device)
        labels = labels.to(device)
        sample_num += seqs.shape[0]
        pred = model(seqs)
        pred_classes = torch.argmax(pred, dim=1)
        accu_corrects += torch.sum(pred_classes == labels)
        loss = criterion(pred, labels)
        accu_loss += loss.item()

        dataloader.desc = "[valid epoch {} / {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            epochs,
            accu_loss.item() / (step + 1),
            accu_corrects.item() / sample_num,
        )

    return accu_loss.item() / (step + 1), accu_corrects.item() / sample_num