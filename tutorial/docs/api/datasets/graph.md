# Graph Datasets

xTorch provides support for graph-based machine learning tasks with a collection of standard graph datasets. These are essential for developing and benchmarking Graph Neural Networks (GNNs).

Graph datasets are located under the `xt::datasets` namespace and can be found in the `<xtorch/datasets/graph_data/>` header directory.

## Graph Data Representation

Unlike image or text data, which is typically represented as a pair of `(data, target)` tensors, graph data has a more complex structure. In xTorch, a graph dataset typically returns a `torch::data::Example` containing multiple components:
- **`x`**: A `[num_nodes, num_node_features]` tensor of node features.
- **`edge_index`**: A `[2, num_edges]` tensor representing the graph's connectivity in COO (coordinate) format. Each column is an edge.
- **`y`**: A tensor of node or graph labels, depending on the task.

The `DataLoader` for graph data is designed to handle this structure and create mini-batches appropriately.

## General Usage

The workflow for using a graph dataset involves instantiating the dataset class and passing it to a data loader. Due to the nature of graph data, complex transformations are less common but still possible.

```cpp
#include <xtorch/xtorch.h>

int main() {
    // 1. Instantiate a dataset for the Cora citation network
    // This dataset is commonly used for node classification.
    auto dataset = xt::datasets::Cora(
        "./data",
        /*download=*/true
    );

    // Note: Graph datasets often represent a single large graph.
    // The "size" might be 1, and batching is handled differently by specialized GNN data loaders.
    std::cout << "Cora dataset loaded." << std::endl;

    // For demonstration, let's get the single graph object from the dataset
    auto graph_data = dataset.get(0);
    auto node_features = graph_data.data;
    auto edge_index = graph_data.target; // Example structure, might differ per dataset

    std::cout << "Node feature shape: " << node_features.sizes() << std::endl;
    std::cout << "Edge index shape: " << edge_index.sizes() << std::endl;

    // 2. Pass the dataset to a DataLoader
    // For GNNs, you might use a specialized graph data loader or a standard one with a batch size of 1
    // if you are doing full-graph training.
    xt::dataloaders::ExtendedDataLoader data_loader(dataset, /*batch_size=*/1, /*shuffle=*/false);

    // The data loader is now ready for use in a training loop
    for (auto& batch : data_loader) {
        // ... training step with a GNN model ...
    }
}
```

!!! warning "Graph Batching"
Batching multiple graphs into a single larger graph (a common technique in GNNs) is a specialized process. While the `ExtendedDataLoader` can iterate over datasets, you may need custom collation logic for advanced GNN training scenarios. For full-graph training (where the entire graph is processed at once), a batch size of 1 is appropriate.

---

## Available Datasets by Task

### Node Classification

Node classification is the task of predicting a label for each node in a graph, given the labels of some nodes.

| Dataset Class | Description | Header File |
|---|---|---|
| `Cora` | A citation network dataset where nodes are documents and edges are citation links. The task is to classify each document into one of seven classes. | `node_classification/cora.h` |

### Graph-Level Tasks (Graph Classification/Regression)

Graph-level tasks involve predicting a single property for an entire graph.

| Dataset Class | Description | Header File |
|---|---|---|
| `OGBMolHIV` | A molecular property prediction dataset from the Open Graph Benchmark. The task is to predict whether a molecule inhibits HIV virus replication. | `molecular_property_prediction/ogb_mo_ihiv.h` |

---

### Knowledge Graph Reasoning

| Dataset Class | Description | Header File |
|---|---|---|
| `Freebase` | A subset of the Freebase knowledge graph used for link prediction tasks. | `knowledge_graph_reasoning/freebase.h` |
| `Wikidata5M` | A large-scale knowledge graph distilled from Wikidata and Wikipedia. | `knowledge_graph_reasoning/wikidata_5m.h` |
