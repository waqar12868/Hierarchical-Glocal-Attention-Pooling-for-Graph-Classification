import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GraphConv, BatchNorm, GATConv
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_networkx
import networkx as nx
import torch.nn as nn
from torch_geometric.data import Data
from collections import defaultdict
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import argparse
import os
from sklearn.model_selection import KFold
from torch.utils.data import random_split
parser = argparse.ArgumentParser()
import numpy as np
from torch import tensor
from layer import HGLA

# Define the neural network architecture
class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.conv1 = GraphConv(self.num_features, self.nhid)
        self.pool1 = HGLA(self.nhid, ratio=self.pooling_ratio)
        self.conv2 = GraphConv(self.nhid, self.nhid)
        self.pool2 = HGLA(self.nhid, ratio=self.pooling_ratio)
        self.conv3 = GraphConv(self.nhid, self.nhid)
        self.pool3 = HGLA(self.nhid, ratio=self.pooling_ratio)
        self.lin1 = torch.nn.Linear(self.nhid*2*3, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.lin3 = torch.nn.Linear(self.nhid//2, self. num_classes)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, batch = self.pool1(x, edge_index, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, batch = self.pool2(x, edge_index, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, batch = self.pool3(x, edge_index, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        max_size = max(x1.size(0), x2.size(0), x3.size(0))
        # Pad the tensors along dimension 0 to match the max_size
        x1 = F.pad(x1, (0, 0, 0, max_size - x1.size(0)))
        x2 = F.pad(x2, (0, 0, 0, max_size - x2.size(0)))
        x3 = F.pad(x3, (0, 0, 0, max_size - x3.size(0)))
        x = torch.cat([x1, x2, x3], dim=1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)
        return x