import os
import time
import math
import torch
import argparse
import numpy as np
import networkx as nx
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
parser = argparse.ArgumentParser()
from collections import defaultdict
from sklearn.model_selection import KFold
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import StratifiedKFold
from torch_geometric.nn import GCNConv,GraphConv,GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import degree
from network import Net
parser.add_argument('--seed', type=int, default=44,
                    help='seed')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay')
parser.add_argument('--nhid', type=int, default=128,
                    help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=0.9,
                    help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.8,
                    help='dropout ratio')
parser.add_argument('--num_epochs', type=int, default=300,
                    help='maximum number of epochs')
parser.add_argument('--dataset', type=str, default='BZR',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')

args = parser.parse_args()
args.device = 'cpu'
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda'

def train(model, train_loader, optimizer):
    correct = 0
    train_loss = 0.0
    model.train()
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(args.device)
        out = model(data)
        out_batch_size = out.size(0)
        target = data.y[:out_batch_size]
        loss = F.nll_loss(out, target)
        loss.backward()
        optimizer.step()
        #running_loss += loss.item()
        pred = out.max(dim=1)[1]
        correct += pred.eq(target).sum().item()
        train_loss += F.nll_loss(out, target, reduction='sum').item()
    return train_loss/ len(train_loader.dataset), correct / len(train_loader.dataset)

# Define the validation function
def validate(model, val_loader):
    model.eval()
    test_loss = 0.0
    num_correct = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(args.device)
            out = model(data)
            out_batch_size = out.size(0)
            target = data.y[:out_batch_size]
            pred = out.max(dim=1)[1]
            num_correct += pred.eq(target).sum().item()
            test_loss += F.nll_loss(out,target, reduction='sum').item()
    return test_loss/len(val_loader.dataset) , num_correct / len(val_loader.dataset)

# Stratified k-fold cross-validation
def k_fold(dataset, folds, seed):
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    graph_labels = np.array([int(data.y) for data in dataset])
    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), graph_labels):
        test_indices.append(torch.from_numpy(idx).to(torch.long))
    val_indices = [test_indices[i - 1] for i in range(folds)]
    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))
    return train_indices, test_indices, val_indices
dataset = TUDataset(os.path.join('data',args.dataset),name=args.dataset)
print("Dataset",dataset)
args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

num_folds = 10
seed = 44
# K-Fold Cross-validation
train_indices, test_indices, val_indices = k_fold(dataset, num_folds, seed)
fold_accuracy = []
for fold, (train_idx, val_idx, test_idx) in enumerate(zip(train_indices, val_indices, test_indices)):
    print(f'Fold {fold+1}')
    train_loader = DataLoader([dataset[i] for i in train_idx], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader([dataset[i] for i in val_idx], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader([dataset[i] for i in test_idx], batch_size=args.batch_size, shuffle=False)
    # Model
    model = Net(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_acc = 0.0
    best_model = None
    total_training_time = 0.0  # Initialize total training time
    best_val_loss = float('inf')
    counter = 0
    patience = 50
    # Training
    for epoch in range(args.num_epochs):
        start_time = time.time()  # Start time for this epoch
        train_loss, train_acc = train(model, train_loader, optimizer)
        end_time = time.time()  # End time for this epoch
        epoch_training_time = end_time - start_time  # Compute training time for this epoch
        total_training_time += epoch_training_time  # Update total training time
        val_loss, val_acc = validate(model, val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0  # Reset counter when there is improvement
            best_model = model.state_dict()
        else:
            counter += 1
        if counter >= patience:
            print('Early stopping triggered.')
            break
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Epoch Time: {epoch_training_time:.2f} seconds')
    print(f'Total training time: {total_training_time:.2f} seconds')
    # Load the best model and test
    model.load_state_dict(best_model)
    #torch.save(model.state_dict(), best_model_path)
    test_loss, test_acc = validate(model, test_loader)
    fold_accuracy.append(test_acc)
    print(f"Fold {fold+1} Test Accuracy: {test_acc:.4f}")
# Mean and Std of accuracy
mean_accuracy = torch.tensor(fold_accuracy).mean().item()* 100
std_accuracy = torch.tensor(fold_accuracy).std().item()* 100
for fold, accuracy in enumerate(fold_accuracy):
    print(f"Fold {fold+1}: {accuracy:.4f}")
print(f"avg_acc±std: {mean_accuracy:.2f} ± {std_accuracy:.2f}")