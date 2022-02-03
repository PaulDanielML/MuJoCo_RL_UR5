from grasping_dataset import Grasping_Dataset
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from Modules import MULTIDISCRETE_RESNET
import torch.nn.functional as F
from collections import deque
import numpy as np
import sys
import time
from tqdm import tqdm
import random

sys.path.append("../")
from Modules import simple_Transition
from torch.utils.tensorboard import SummaryWriter


BATCH_SIZE = 15
SAVE_WEIGHTS = True
LOAD_WEIGHTS = False
LEARNING_RATE = 0.001

EPOCHS = 20

SEED = 30

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

date = "_".join(
    [
        str(time.localtime()[1]),
        str(time.localtime()[2]),
        str(time.localtime()[0]),
        str(time.localtime()[3]),
        str(time.localtime()[4]),
    ]
)
DESCRIPTION = "_".join(
    [
        "LR",
        str(LEARNING_RATE),
        "OPTIM",
        "ADAM",
        "H",
        "EPOCHS",
        str(EPOCHS),
        "BATCH_SIZE",
        str(BATCH_SIZE),
        "SEED",
        str(SEED),
    ]
)
WEIGHT_PATH = DESCRIPTION + "_" + date + "_weights.pt"

LOAD_PATH = "LR_0.001_OPTIM_ADAM_H_EPOCHS_20_BATCH_SIZE_15_SEED_30_9_15_2020_23_3_weights.pt"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file = "Data/16_09_20_10_54_total_of_6081_transitions.pt"
# file = 'Data/14_09_20_13_56_total_of_16968_transitions.pt'

dataset_1 = Grasping_Dataset(file)

train_size = int(0.8 * len(dataset_1))
test_size = len(dataset_1) - train_size
trainset, testset = random_split(dataset_1, [train_size, test_size])

train_loader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=True)

policy_net = MULTIDISCRETE_RESNET(6).to(device)

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE, weight_decay=0.00002)

writer = SummaryWriter(comment=DESCRIPTION)


if LOAD_WEIGHTS:
    checkpoint = torch.load(LOAD_PATH)
    policy_net.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Loaded network weights and optimizer state from {LOAD_PATH}.")


def main():

    train_losses = AverageMeter("Training Loss", ":.4f")
    train_pos_accuracy = AverageMeter("Training positive Accuracy", ":.5f")
    train_neg_accuracy = AverageMeter("Training negative Accuracy", ":.5f")

    start_time = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        train_losses.reset()
        train_pos_accuracy.reset()
        train_neg_accuracy.reset()
        for data in tqdm(train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True):

            state_batch = data[0].to(device)
            action_batch = data[1].unsqueeze(1).to(device)
            reward_batch = data[2].unsqueeze(1).to(device)

            # Current Q prediction of our policy net, for the actions we took
            q_pred = policy_net(state_batch).view(len(data[0]), -1).gather(1, action_batch)

            q_expected = reward_batch.float()

            loss = F.binary_cross_entropy(q_pred, q_expected)
            # loss = F.binary_cross_entropy(q_pred, q_expected) / NUMBER_ACCUMULATIONS_BEFORE_UPDATE
            loss.backward()

            train_losses.update(loss.item(), len(data[0]))
            optimizer.step()

            optimizer.zero_grad()

            pos_acc, number_pos, neg_acc, number_neg = binary_accuracy(q_pred, reward_batch)
            train_pos_accuracy.update(pos_acc, number_pos)
            train_neg_accuracy.update(neg_acc, number_neg)

        print(train_losses)
        print(train_pos_accuracy)
        print(train_neg_accuracy)
        writer.add_scalar("Loss/Train", train_losses.avg, epoch)
        writer.add_scalar("Accuracy Positive/Train", train_pos_accuracy.avg, epoch)
        writer.add_scalar("Accuracy Negative/Train", train_neg_accuracy.avg, epoch)

        test_loss, test_pos_acc, test_neg_acc = eval_on_test_set()

        writer.add_scalar("Loss/Test", test_loss, epoch)
        writer.add_scalar("Accuracy Positive/Test", test_pos_acc, epoch)
        writer.add_scalar("Accuracy Negative/Test", test_neg_acc, epoch)

    end_time = time.perf_counter()
    run_time = end_time - start_time
    print(
        f"Training ({EPOCHS} epochs, dataset size: {len(dataset_1)}) took {run_time:.1f} secs ({(run_time/3600):.1f} h)."
    )

    if SAVE_WEIGHTS:
        if LOAD_WEIGHTS:
            torch.save(
                {
                    "model_state_dict": policy_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                LOAD_PATH,
            )
            print(f"Saved network weights and optimizer state to {LOAD_PATH}.")

        else:
            torch.save(
                {
                    "model_state_dict": policy_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                WEIGHT_PATH,
            )
            print(f"Saved network weights and optimizer state to {WEIGHT_PATH}.")


def eval_on_test_set():
    test_losses = AverageMeter("Test Loss", ":.4f")
    test_pos_accuracy = AverageMeter("Test Positive Accuracy", ":.5f")
    test_neg_accuracy = AverageMeter("Test Negative Accuracy", ":.5f")

    policy_net.eval()

    with torch.no_grad():
        for data in tqdm(test_loader, desc=f"Testing", dynamic_ncols=True):
            state_batch = data[0].to(device)
            action_batch = data[1].unsqueeze(1).to(device)
            reward_batch = data[2].unsqueeze(1).to(device)

            # Current Q prediction of our policy net, for the actions we took
            q_pred = policy_net(state_batch).view(len(data[0]), -1).gather(1, action_batch)

            q_expected = reward_batch.float()

            loss = F.binary_cross_entropy(q_pred, q_expected)
            test_losses.update(loss.item(), len(data[0]))
            pos_acc, number_pos, neg_acc, number_neg = binary_accuracy(q_pred, reward_batch)
            test_pos_accuracy.update(pos_acc, number_pos)
            test_neg_accuracy.update(neg_acc, number_neg)

        print(test_losses)
        print(test_pos_accuracy)
        print(test_neg_accuracy)

    return test_losses.avg, test_pos_accuracy.avg, test_neg_accuracy.avg


def binary_accuracy(predictions, targets, positive_threshold=0.5, negative_threshold=0.3):
    """Checks weather each predictions lies within a specified threshold of the ground truth"""

    # print(predictions)
    # print(targets)

    with torch.no_grad():
        positive_labels = targets.sum().item()
        positive_predection = (predictions > positive_threshold).int()
        correct_prediction = (positive_predection == targets).int()
        correct_positive_pred = (correct_prediction * targets).sum().item()
        if positive_labels != 0:
            positive_accuracy = correct_positive_pred / positive_labels
        else:
            positive_accuracy = 0

        negative_labels = targets.size()[0] - positive_labels
        negative_positions = (targets == 0).int()
        negative_predection = (predictions < negative_threshold).int()
        correct_prediction_2 = (negative_predection != targets).int()
        correct_negative_pred = (correct_prediction_2 * negative_positions).sum().item()
        if negative_labels != 0:
            negative_accuracy = correct_negative_pred / negative_labels
        else:
            negative_accuracy = 0

    return positive_accuracy, positive_labels, negative_accuracy, negative_labels


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = (
            "{name} {val" + self.fmt + "} (last) ({avg" + self.fmt + "} average, {count} samples)"
        )
        return fmtstr.format(**self.__dict__)


if __name__ == "__main__":
    main()
