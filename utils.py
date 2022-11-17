import matplotlib.pyplot as plt
import torch
import time
import os

from models import RNN, CNN, GRU, LSTM, MLP

def save_model(args, epoch, model, optimizer, val_acc, train_loss, best_acc, save_path, record_path, best_model=False):

    if not best_model:
        filename = "checkpoint_epoch_{}_acc_{}.pth".format(epoch, val_acc)
        mode = "checkpoint"
    else:
        filename = "best_weights.pth"
        mode = "best_weights"

    state_dict = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": train_loss,
        "best_acc": best_acc,
        "args": args
    }

    torch.save(state_dict, os.path.join(save_path, filename))

    # save record
    with open(record_path, "a") as f:
        f.write("save {} file : {}, current_epoch : {}, loss : {:.5f}, val_acc : {:.5f}\n".format(
            mode, filename, epoch, train_loss, val_acc))


def train_record(epoch, epochs, train_loss, train_acc, val_loss, val_acc, time, record_path):

    with open(record_path, "a") as f:
        f.write("[epoch {} / {}] train_loss: {:.3f},train_acc:{:.3f},val_loss:{:.3f},val_acc:{:.3f},Spend_time:{:.3f}s \n"
                .format(epoch, epochs, train_loss, train_acc, val_loss, val_acc, time))


def load_model(checkpoint, optimizer, model, device, record_path):

    assert os.path.exists(checkpoint), "checkpoint file '{}' not exists.".format(checkpoint)

    checkpoint = torch.load(checkpoint, map_location=device)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    start_epoch = checkpoint["epoch"]
    best_acc = checkpoint["best_acc"]

    with open(record_path, "a") as f:
        f.write("load model file : {}, epoch : {}, best_acc : {:.5f}\n".format(
            checkpoint, start_epoch, best_acc))

    return start_epoch + 1, best_acc

def create_logs(args, model):

    save_path = "logs/runs_" + model.__class__.__name__ + time.strftime("_%Y-%m-%d_%H-%M-%S", time.localtime())
    os.makedirs(save_path)
    record_path = os.path.join(save_path, "record.txt")

    with open(record_path, 'w') as record:
        record.write("{}\n".format(args))

    return save_path, record_path

def create_model(model, hidden_size=64, classes_num=5):
    if model == "cnn":
        model_ = CNN(classes_num)
    elif model == "rnn":
        model_ = RNN(hidden_size=hidden_size, classes_num=classes_num)
    elif model == "lstm":
        model_ = LSTM(hidden_size=hidden_size, classes_num=classes_num)
    elif model == "gru":
        model_ = GRU(hidden_size=hidden_size, classes_num=classes_num)
    elif model == "mlp":
        model_ = MLP(classes_num=classes_num)
    else:
        assert "input model not exists, the name must in [cnn, rnn, lstm, gru, mlp]"
    return model_

def plot_history(train_loss_list, train_acc_list, val_loss_list, val_acc_list, epochs, record_path):
    plt.plot([str(i + 1) for i in range(epochs)], train_loss_list, color="blue", label="train_loss")
    plt.plot([str(i + 1) for i in range(epochs)], val_loss_list, color="orange", label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("loss")
    plt.legend()
    #注意顺序
    plt.savefig(os.path.join(record_path, "loss.png"))
    plt.show()

    plt.plot([str(i + 1) for i in range(epochs)], train_acc_list, color="blue", label="train_acc")
    plt.plot([str(i + 1) for i in range(epochs)], val_acc_list, color="orange", label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.title("accuary")
    plt.legend()
    plt.savefig(os.path.join(record_path, "acc.png"))
    plt.show()
