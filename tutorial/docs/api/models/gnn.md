# Graph Neural Networks (GNNs)

Graph Neural Networks (GNNs) are a specialized class of neural networks designed to perform inference on data structured as graphs. They are essential for tasks like node classification, link prediction, and graph classification, with applications in social networks, molecular chemistry, and recommendation systems.

xTorch provides implementations of several foundational and popular GNN architectures, enabling you to apply deep learning to graph-structured data.

All GNN models are located under the `xt::models` namespace and their headers can be found in the `<xtorch/models/graph_neural_networks/>` directory.

## General Usage

While GNN models are standard `torch::nn::Module`s, their usage differs slightly from traditional models like CNNs or MLPs, particularly in the signature of the `forward` method.

A GNN's `forward` pass typically requires two main inputs:
1.  **`x`**: A 2D tensor of node features, with shape `[num_nodes, num_node_features]`.
2.  **`edge_index`**: A 2D tensor representing the graph's connectivity (the adjacency list), with shape `[2, num_edges]`.

The model then uses message passing along the edges to update the node representations.

```cpp
#include <xtorch/xtorch.h>
#include <iostream>

int main() {
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    // --- Define Graph Properties ---
    const int num_nodes = 1000;
    const int num_node_features = 64;
    const int num_classes = 7; // E.g., for a node classification task
    const int num_edges = 4500;

    // --- Instantiate a GCN Model ---
    // A simple 2-layer Graph Convolutional Network
    xt::models::GCN model(num_node_features, /*hidden_channels=*/16, num_classes);
    model.to(device);
    model.train();

    std::cout << "GCN Model Instantiated." << std::endl;

    // --- Create Dummy Graph Data ---
    // Node feature matrix
    auto x = torch::randn({num_nodes, num_node_features}).to(device);
    // Edge index tensor (random edges for demonstration)
    auto edge_index = torch::randint(0, num_nodes, {2, num_edges}).to(device);

    // --- Perform a Forward Pass ---
    // Pass both node features and edge index to the model
    auto output_node_embeddings = model.forward(x, edge_index);

    std::cout << "Output shape: " << output_node_embeddings.sizes() << std::endl; // Should be
}
```

!!! note "Training GNNs"
Training GNNs can involve different strategies. For small graphs that fit in memory ("full-batch" training), you process the entire graph at once. For large graphs, you use neighborhood sampling to create mini-batches. Refer to the xTorch examples for practical training loops.

---

## Available GNN Models

xTorch provides the following GNN architectures:

| Model | Description | Header File |
|---|---|---|
| `GCN` | **Graph Convolutional Network**. A foundational GNN model that learns node features by aggregating information from their local neighborhoods. | `gcn.h` |
| `GraphSAGE` | **Graph SAmple and aggreGatE**. An inductive framework that generates node embeddings by sampling and aggregating features from a node's local neighborhood. | `graph_sage.h` |
| `GAT` | **Graph Attention Network**. A GNN that leverages masked self-attention to assign different importance weights to nodes within a neighborhood. | `gat.h` |
| `GIN` | **Graph Isomorphism Network**. A powerful GNN shown to be as discriminative as the Weisfeiler-Lehman graph isomorphism test. | `gin.h` |
| `DiffPool` | **Differentiable Pooling**. A module that learns a hierarchical representation of graphs, allowing GNNs to be used for graph classification. | `diff_pool.h` |
| `GraphUNet` | A GNN architecture inspired by U-Net, using graph pooling and unpooling to learn multi-scale node features. | `graph_unet.h` |
