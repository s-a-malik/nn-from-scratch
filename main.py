"""Machine Learning Assignment, AIMS CDT 2021
Author: Shreshth Malik
Neural Network for classifying flowers.
"""

import os
import random
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from data import load_iris_data
from net import NN
from utils import cross_entropy_loss, plot_confusion, plot_train_curves

# plt set up
plt.style.use('seaborn')
plt.rc('font', size=15)
plt.rc('xtick', labelsize='medium')
plt.rc('ytick', labelsize='medium')
plt.rc('xtick.major', size=5, width=1.5)
plt.rc('ytick.major', size=5, width=1.5)
plt.rc('axes', linewidth=2, labelsize='large', titlesize='large')
plt.rcParams["lines.markeredgewidth"] = 2


def parse_args():
    """Parse command line arguments
    """

    parser = argparse.ArgumentParser()

    # optimisation parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for training')
    parser.add_argument('--seed', type=int, default=123, help='Seed for random number generator')
    parser.add_argument('--evaluate', action='store_true', default=False, help='Evaluate the model on the test set, skipping training')

    # data parameters
    parser.add_argument('--val-size', type=float, default=0.2, help='Fraction of training data to use for validation')
    parser.add_argument('--data-dir', type=str, default='./', help='Directory with the data file')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to a checkpoint to load (for evaluation)')

    # architecture parameters
    parser.add_argument('--hidden-dim', type=int, default=32, help='Size of hidden layers')
    parser.add_argument('--num-layers', type=int, default=2, choices=[1, 2], help='Number of hidden layers')
    parser.add_argument('--initialisation', type=str, default='normal', choices=['xavier', 'uniform', 'normal'], help='Initialisation method for weights')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'sigmoid'], help='Activation function')

    args = parser.parse_args()

    return args


def train(args, model, train_loader, val_loader):
    """Train the model
    """

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        total_train_loss = 0
        total_train_correct = 0
        total = 0
        with tqdm.tqdm(total=len(train_loader)) as pbar:
            for (x,y) in train_loader:
                # zero gradients
                model.zero_grad()
                # forward pass
                out = model(x)
                # backward pass
                train_loss = cross_entropy_loss(out, y)
                train_loss.backward()
                # update weights using SGD
                for p in model.parameters():
                    p.data = p.data - args.lr * p.grad
                # predictions
                y_pred = torch.argmax(out, dim=1)

                total_train_loss += train_loss.item()*y.size(0)
                total += y.size(0)
                total_train_correct += torch.sum(y_pred == y).item()
                pbar.update()
        train_loss_avg = total_train_loss / total
        train_acc_avg = total_train_correct / total
        train_losses.append(train_loss_avg)
        train_accs.append(train_acc_avg)
        
        total_val_loss = 0
        total_val_correct = 0
        total = 0
        # eval on val set
        with torch.no_grad():
            for (x,y) in val_loader:
                # forward pass
                out = model(x)
                val_loss = cross_entropy_loss(out, y)
                total_val_loss += val_loss.item()*y.size(0)
                y_pred = torch.argmax(out, dim=1)
                total_val_correct += torch.sum(y_pred == y).item()
                total += y.size(0)
            val_loss_avg = total_val_loss / total
            val_acc_avg = total_val_correct / total
            val_losses.append(val_loss_avg)
            val_accs.append(val_acc_avg)

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg

        print(f'Epoch: {epoch + 1}/{args.epochs}, Train Loss: {train_loss_avg}, Val Loss: {val_loss_avg},\n'
                f'Train Acc: {train_acc_avg}, Val Acc: {val_acc_avg} \n'
                f'Best val loss: {best_val_loss}')
    
    return model, train_losses, val_losses, train_accs, val_accs



def test(model, test_loader):
    """Evaluate the model on the test set
    """

    total_correct = 0
    total = 0 
    preds = []
    targets = []

    # run through test set and compute accuracy
    for (x,y) in test_loader:
        # forward pass
        out = model(x)
        # compute accuracy
        y_pred = torch.argmax(out, dim=1)
        total_correct += torch.sum(y_pred == y).item()
        total += y.size(0)
        preds += y_pred.tolist()
        targets += y.tolist()

    conf_matrix = confusion_matrix(targets, preds, normalize='true')

    test_acc = total_correct / total
    print(f'\nTEST ACC: {test_acc}')
    return test_acc, conf_matrix


def main(args):
    """Main function
    """
    # load data
    scaler, train_dataset, val_dataset, test_dataset = load_iris_data(args)
    input_dim = train_dataset[0][0].shape[0]
    num_classes = 3

    # initialise model
    model = NN(input_dim, num_classes, args.num_layers, args.hidden_dim, args.initialisation, args.activation)

    if not args.evaluate:
        # train the model
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
        model, train_losses, val_losses, train_accs, val_accs = train(args, model, train_loader, val_loader)
    else:
        # load the model from checkpoint

        pass

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # test the model
    test_acc, conf_matrix = test(model, test_loader)

    # plot results
    plot_train_curves(train_losses, val_losses, train_accs, val_accs)
    plot_confusion(conf_matrix)


if __name__ == "__main__":
    
    # parse command line arguments
    args = parse_args()
    # random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    # run program
    main(args)
