import torch
import math
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import to_networkx
import networkx as nx
from torch_geometric.data import Data
from collections import defaultdict
import argparse
import torch.nn as nn
parser = argparse.ArgumentParser()
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

class HGLA(torch.nn.Module):
    def __init__(self, in_channels, ratio, mlp_hidden=64):
        super(HGLA, self).__init__()
        self.pooling_ratio = ratio
        self.clique_ratio =0.8
        self.attention_GCNConv = GCNConv(in_channels, 1)
        self.attention_GATConv = GATConv(in_channels, 1, heads=1)
        self.alpha = torch.nn.Parameter(torch.tensor(0.6), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1)
        )
        #self.feature_transform = torch.nn.Linear(in_channels + 1, 1)  # New transformation layer
    def calc_structure_score(self, clique, graph, degree_centrality):
        # Use precomputed degree centrality
        avg_degree = sum(degree_centrality[node] for node in clique) / len(clique)
        
        neighbors = set()
        for node in clique:
            neighbors.update(graph.neighbors(node))
        unique_neighbors = len(neighbors - set(clique))
        # Combine degree centrality and unique neighbors to form structure score
        structure_score = self.beta * avg_degree + (1 - self.beta) * unique_neighbors
        return structure_score
    
    def extend_cliques_based_on_overlap(self, cliques, graph):
        # Precompute neighbors for each node
        neighbors = {node: set(graph.neighbors(node)) for node in graph.nodes()}
        # Clique association for each node
        node_to_cliques = defaultdict(set)
        for idx, clique in enumerate(cliques):
            for node in clique:
                node_to_cliques[node].add(idx)
        extended_cliques = list(cliques)
        for i, clique_a in enumerate(cliques):
            for node_a in clique_a:
                for j in node_to_cliques[node_a]:
                    # Skip if comparing the same clique
                    if i == j:
                        continue
                    clique_b = cliques[j]
                    neighbors_in_b = neighbors[node_a].intersection(clique_b)
                    if len(neighbors_in_b) > 0.5 * len(clique_b) and node_a not in extended_cliques[j]:
                        extended_cliques[j].append(node_a)
        return extended_cliques
    def forward(self, x, edge_index, batch):
        g = Data(x=x, edge_index=edge_index)
        g = to_networkx(g, to_undirected=True)
        cliques = list(nx.find_cliques(g))
        cliques = sorted(cliques, key=len, reverse=True)
        # Compute attention scores for each node
        node_features = self.attention_GCNConv(x,edge_index).view(-1, 1)
        node_features_GCN = F.softmax(node_features, dim=0)
        #handle the multiple assign to more than two cliques
        clusters = defaultdict(list)
        cluster_sz = {}
        considered_cliques = []
        cluster_idx = 0
        for clique in cliques:
            if all([len(clusters[v]) != 0 for v in clique]):
                continue
            considered_cliques.append(clique)
            cluster_sz[cluster_idx] = len(clique)
            for v in clique:
                clusters[v].append(cluster_idx)
            cluster_idx += 1
        extended_cliques = self.extend_cliques_based_on_overlap(considered_cliques, g)
        degree_centrality = nx.degree_centrality(g)
        # Calculate structure score for each clique
        structure_scores = [self.calc_structure_score(clique, g,degree_centrality) for clique in extended_cliques]
        # Calculate clique scores by summing node attention scores
        clique_scores = [node_features_GCN[clique].sum().item() for clique in extended_cliques]
        # Combine attention and structure scores
        combined_scores = [self.alpha * att_score + (1 - self.alpha) * struct_score 
                            for att_score, struct_score in zip(clique_scores, structure_scores)]
        # Sort cliques based on their combined scores
        sorted_cliques = sorted(zip(extended_cliques, combined_scores), key=lambda x: x[1], reverse=True)
        # Select the top k cliques based on the pooling ratio
        num_top_k = int(math.ceil(len(extended_cliques) * self.clique_ratio))
        
        selected_cliques = [clique for clique, _ in sorted_cliques[:num_top_k]]
        clique_mask = torch.zeros(x.size(0), dtype=torch.bool)
        for clique in selected_cliques:
            clique_mask[clique] = 1
        
        x_clique = x[clique_mask]
    
        node_mapping = {old: new for new, old in enumerate(clique_mask.nonzero().view(-1).tolist())}
        indices = torch.tensor([node_mapping.get(x, -1) for x in range(x.size(0))], device=x.device)
        edge_index = indices.index_select(0, edge_index.view(-1)).view(2, -1)
        
        edge_mask = (edge_index[0] != -1) & (edge_index[1] != -1)
        edge_index = edge_index[:, edge_mask]

        batch_clique = batch[clique_mask]

       # Compute attention scores for each node
        node_attention_GCN = self.attention_GCNConv(x_clique, edge_index).view(-1, 1)
        node_attention_GCN = F.softmax(node_attention_GCN, dim=0)

        node_attention_GAT = self.attention_GATConv(x_clique, edge_index).view(-1, 1)
        node_attention_GAT = F.softmax(node_attention_GAT, dim=0)

        #node_importance_GAT = node_attention_GAT.mean(dim=1)
        node_attention = torch.max(torch.cat((node_attention_GCN.unsqueeze(1), node_attention_GAT.unsqueeze(1)), dim=1), dim=1)[0]
      
        xscore = x_clique.sum(dim=1).unsqueeze(1)
        # Concatenate attention scores with node features
        x_with_attention = torch.cat([xscore, node_attention], dim=-1)
        
        scores = x_with_attention.sum(dim=1)
        
        k = int(math.ceil(self.pooling_ratio*x_clique.size(0)))

        _, topk_indices = torch.topk(scores, k)
      
        x_topk = x_clique[topk_indices]
        
        batch_topk = batch_clique[topk_indices]
       
       # Form a mask for edges that are between nodes in the top-k set
        mask_source = torch.any(edge_index[0].unsqueeze(-1) == topk_indices.unsqueeze(0), dim=-1)
        mask_target = torch.any(edge_index[1].unsqueeze(-1) == topk_indices.unsqueeze(0), dim=-1)

        # Combine the two masks with logical AND operation
        edge_mask = mask_source & mask_target
        
        edge_index_topk = edge_index[:, edge_mask]

        # Create a mapping from old node indices to new node indices
        sorted_topk_indices, new_indices = topk_indices.sort()
        node_mapping = torch.full((x_clique.size(0),), -1, dtype=torch.long, device=x.device)
        node_mapping[sorted_topk_indices] = new_indices

        # Remap node indices in edge_index
        edge_index_topk = node_mapping[edge_index_topk]

        # Remove edges that point to non-existent nodes (-1)
        edge_mask = (edge_index_topk[0] != -1) & (edge_index_topk[1] != -1)
        edge_index_topk = edge_index_topk[:, edge_mask]

        return x_topk, edge_index_topk, batch_topk