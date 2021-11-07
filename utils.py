"""Utility functions
"""

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

LABEL_DICT = {0: 'Iris-setosa',
           1: 'Iris-virginica',
           2: 'Iris-versicolor'}


def cross_entropy_loss(y_pred, y_true):
    """Cross entropy loss function
    Params:
    - y (torch.tensor): predictions (shape: (batch_size, num_classes))
    - t (torch.tensor): targets (shape: (batch_size,))
    Returns:
    - loss (float): cross entropy loss
    """

    # expand the target to the same shape as the predictions
    y_true_ohe = F.one_hot(y_true, num_classes=y_pred.size(1)).float()

    # add a small number to avoid log(0)
    # print(y_true_ohe, y_pred)
    loss = torch.sum(-y_true_ohe * torch.log(y_pred + 1e-10), dim=1)
    # print(loss)
    # return average loss across all samples
    return torch.mean(loss)


def plot_train_curves(train_losses, val_losses, train_accs, val_accs):
    """Plot training curves
    Params:
    - train_losses (list): training losses
    - val_losses (list): validation losses
    - train_accs (list): training accuracies
    - val_accs (list): validation accuracies
    """

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

    axes[0].plot(train_losses, label='Training Loss')
    axes[0].plot(val_losses, label='Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Learning Curves (Loss)')
    axes[0].legend()
    axes[1].plot(train_accs, label='Training Acc')
    axes[1].plot(val_accs, label='Validation Acc')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Learning Curves (Accuracy)')
    axes[1].legend()
    
    plt.savefig('training_curves.png', dpi=300)


def plot_confusion(conf_matrix):
    """Plot confusion matrix
    Params:
    - conf_matrix (np.array): confusion matrix
    """

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(conf_matrix, cmap='Oranges')

    # set the labels
    ax.set_xticks(range(conf_matrix.shape[1]))
    ax.set_yticks(range(conf_matrix.shape[0]))
    ax.set_xticklabels([LABEL_DICT[i] for i in range(conf_matrix.shape[1])],
                       rotation=0)
    ax.set_yticklabels([LABEL_DICT[i] for i in range(conf_matrix.shape[1])], 
                       rotation=0)

    # set the label's font size
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, "{:.4f}".format(conf_matrix[i, j]), ha='center', va='center', fontsize=12)

    # set the title
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    # set the colorbar
    # fig.colorbar(im, ax=ax)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
