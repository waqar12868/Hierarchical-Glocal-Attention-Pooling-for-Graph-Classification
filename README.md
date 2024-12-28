# Hierarchical-Glocal-Attention-Pooling-for-Graph-Classification
Graph pooling is an essential operation in Graph Neural Networks that reduces the size of an input graph while preserving its core structural properties. Existing pooling methods find a compressed representation considering the Global Topological Structures (e.g., cliques, stars, clusters) or Local information at node level (e.g., top-
 informative nodes). However, an effective graph pooling method does not hierarchically integrate both Global and Local graph properties. To this end, we propose a dual-fold Hierarchical Global Local Attention Pooling (HGLA-Pool) layer that exploits the aforementioned graph properties, generating more robust graph representations. Exhaustive experiments on nine publicly available graph classification benchmarks under standard metrics show that HGLA-Pool significantly outperforms eleven state-of-the-art models on seven datasets while being on par for the remaining two.

![G-Local (1)](https://github.com/user-attachments/assets/fbe4816f-ba39-4c52-8293-113f9e941b4e)



# Requirements
pytorch=1.13.1
torch-geometric=2.3.0
torch-scatter=2.1.1
torch-sparse=0.6.17
torch-spline-conv=1.2.2
torch-cluster=1.6.1
