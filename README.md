# Hierarchical-Glocal-Attention-Pooling-for-Graph-Classification

Official Code Repository for the paper "Hierarchical Glocal Attention Pooling for Graph Classification" (Pattern Recognition Letter 2024): https://doi.org/10.1016/j.patrec.2024.09.009

In this repository, we implement the proposed Hierarchical Glocal Attention Pooling.

![G-Local (1)](https://github.com/user-attachments/assets/fbe4816f-ba39-4c52-8293-113f9e941b4e)

# Abstract
Graph pooling is an essential operation in Graph Neural Networks that reduces the size of an input graph while preserving its core structural properties. Existing pooling methods find a compressed representation considering the Global Topological Structures (e.g., cliques, stars, clusters) or Local information at node level (e.g., top-
 informative nodes). However, an effective graph pooling method does not hierarchically integrate both Global and Local graph properties. To this end, we propose a dual-fold Hierarchical Global Local Attention Pooling (HGLA-Pool) layer that exploits the aforementioned graph properties, generating more robust graph representations. Exhaustive experiments on nine publicly available graph classification benchmarks under standard metrics show that HGLA-Pool significantly outperforms eleven state-of-the-art models on seven datasets while being on par for the remaining two.

# Contribution of this work
• Propose dual-fold pooling to capture global and local properties for classification. <br>
• Fold1 uses a developed rule-based method to identify overlapping nodes among cliques. <br>
• Develop dynamic scoring to rank the most informative global structures like cliques. <br>
• Fold2 uses LocalPool to refine cliques by focusing on key nodes within cliques. <br>
• The proposed method outperforms 11 state-of-the-art models on seven diverse datasets.

# Dependencies
pytorch=1.13.1 <br>
torch-geometric=2.3.0 <br>
torch-scatter=2.1.1 <br>
torch-sparse=0.6.17 <br>
torch-spline-conv=1.2.2 <br>
torch-cluster=1.6.1

# Citation
If you found the provided code with our paper useful in your work, we kindly request that you cite our work.

@article{ali2024hierarchical,<br>
  title={Hierarchical glocal attention pooling for graph classification},<br>
  author={Ali, Waqar and Vascon, Sebastiano and Stadelmann, Thilo and Pelillo, Marcello},<br>
  journal={Pattern Recognition Letters},<br>
  volume={186},<br>
  pages={71--77},<br>
  year={2024},<br>
  publisher={Elsevier}<br>
}
