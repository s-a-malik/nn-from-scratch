"""Data loaders and transformations
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset


df = pd.read_csv(f"./IRIS.csv", sep=',')
# df.head()

# df[df.isnull().any(axis=1)]

# pd.get_dummies()
# df["species"]


def load_iris_data(args):
    """Loads and preprocesses data from a file.
    Params:
    - args (argparse.Namespace): command line arguments
    Returns:
    - train_dataset (Dataset): training data
    - val_dataset (Dataset): validation data
    - test_dataset (Dataset): test data
    """
    # load raw data
    df = pd.read_csv(f"{args.data_dir}/IRIS.csv", sep=',')
    df["species"] = df["species"].astype('category')
    # integer encode classes
    df["species"] = df["species"].cat.codes
    y = df["species"].to_numpy()
    df = df.drop(columns=["species"])
    X = df.to_numpy()

    # split into training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    scaler = StandardScaler()
    # fit to training data only
    scaler.fit(X_train)
    # scale training and test data
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # split train into validation and training
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=args.val_size, random_state=args.seed)
    
    # transform to torch tensor
    X_train, X_val, X_test = [torch.tensor(x, dtype=torch.float32) for x in [X_train, X_val, X_test]]
    y_train, y_val, y_test = [torch.tensor(x, dtype=torch.int64) for x in [y_train, y_val, y_test]]

    # put in dataset
    train_dataset = TensorDataset(X_train, y_train) 
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    return scaler, train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    # test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='Seed for random number generator')
    parser.add_argument('--val-size', type=float, default=0.2, help='Fraction of training data to use for validation')
    parser.add_argument('--data-dir', type=str, default='./', help='Directory with the data file')

    args = parser.parse_args()
    scaler, train_dataset, val_dataset, test_dataset = load_iris_data(args)
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    print(train_dataset[:5])
