# Graph Transforms

Data augmentation for graph-structured data is a powerful technique for improving the generalization of Graph Neural Networks (GNNs). Unlike image or audio transforms, graph transforms operate on the structure and features of a graph, such as its nodes, edges, and feature matrices.

These augmentations can help the model learn more robust representations by training on variations of the input graph, preventing it from overfitting to specific structural or feature patterns.

All graph transforms are located under the `xt::transforms` namespace and can be found in the `<xtorch/transforms/graph/>` header directory.

## General Usage

Graph transforms are designed to be used within a `Compose` pipeline, just like transforms for other data modalities. They are typically applied within a `Dataset`'s preprocessing step. A graph transform takes a graph data object (which usually contains node features `x` and an `edge_index` tensor) and returns a modified version.

```cpp
#include <xtorch/xtorch.h>
#include <iostream>

int main() {
    // 1. Define a pipeline of graph augmentations
    auto transform_pipeline = std::make_unique<xt::transforms::Compose>(
        // Randomly drop 10% of the nodes during training
        std::make_shared<xt::transforms::graph::NodeDrop>(/*p=*/0.1),
        // Randomly drop 20% of the edges
        std::make_shared<xt::transforms::graph::EdgeDrop>(/*p=*/0.2)
    );

    // 2. Pass the pipeline to a graph Dataset
    // The dataset will apply these augmentations to the graph data each time a sample is requested.
    auto dataset = xt::datasets::Cora(
        "./data",
        /*download=*/true,
        std::move(transform_pipeline)
    );

    // 3. The DataLoader will now provide augmented graphs during training
    xt::dataloaders::ExtendedDataLoader data_loader(dataset, /*batch_size=*/1);

    for (auto& batch : data_loader) {
        // The graph data in this batch has been randomly modified by the transforms
        auto augmented_graph_data = batch;
        // ... training step ...
    }
}
```

---

## Available Transforms

The graph transforms can be grouped into two main categories: those that modify the graph's structure and those that modify its features.

### Structural Augmentations

These transforms alter the topology of the graph by adding, removing, or modifying nodes and edges.

| Transform | Description | Header File |
|---|---|---|
| `NodeDrop` | Randomly removes a fraction of nodes (and their connections) from the graph. | `node_drop.h` |
| `EdgeDrop` | Randomly removes a fraction of edges from the graph. | `edge_drop.h` |
| `EdgeAdd` | Randomly adds new edges to the graph. | `edge_add.h` |
| `EdgePerturbation` | Randomly adds some edges and removes others. | `edge_perturbation.h` |
| `Subgraph` | Samples a subgraph from the input graph based on a subset of nodes. | `subgraph.h` |
| `RandomWalkSubgraph`| Samples a subgraph by performing random walks from a set of starting nodes. | `random_walk_subgraph.h`|
| `GraphCoarsening`| Reduces the size of the graph by merging nodes. | `graph_coarsening.h` |

### Feature Augmentations

These transforms alter the feature matrices of the nodes or edges.

| Transform | Description | Header File |
|---|---|---|
| `NodeFeatureMasking`| Randomly masks (sets to zero) the features of some nodes. | `node_feature_masking.h` |
| `NodeFeatureShuffling`| Randomly shuffles the features across different nodes. | `node_feature_shuffling.h`|
| `EdgeFeatureMasking`| Randomly masks the features of some edges. | `edge_feature_masking.h` |
| `FeatureDropout` | Applies dropout to the node or edge feature matrix. | `feature_dropout.h` |
| `FeatureAugmentation`| Augments node features by adding random noise or other perturbations. | `feature_augmentation.h` |

### Hybrid Augmentations

These transforms modify both the structure and features of the graph simultaneously.

| Transform | Description | Header File |
|---|---|---|
| `DropEdgeAndFeature`| Randomly drops edges and masks node features. | `drop_edge_and_feature.h` |
| `GraphMixUp` | A graph-based implementation of MixUp, creating new graphs by interpolating between two existing graphs. | `graph_mix_up.h` |
| `NodeMixUp` | Creates new node features by interpolating between features of other nodes. | `node_mix_up.h` |
| `GraphDiffusion` | Applies a diffusion process to the graph's features. | `graph_diffusion.h` |

!!! info "Constructor Options"
Many of these transforms have tunable parameters, such as the probability `p` of dropping a node/edge. These are typically passed as arguments to the constructor. Please refer to the specific header file for details on the available settings.
