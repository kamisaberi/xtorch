# Time Series Datasets

Time-series data consists of sequences of data points indexed in time order. It is a fundamental data type in many domains, including finance, weather forecasting, and sensor data analysis. Common tasks include forecasting future values, classifying sequences, and detecting anomalies.

xTorch provides handlers for popular time-series datasets to facilitate research and development in this area. These datasets are located under the `xt::datasets` namespace and can be found in the `<xtorch/datasets/time_series/>` header directory.

## General Usage

Working with time-series data often involves specific preprocessing steps like creating sliding windows of data, normalization, and feature engineering. These can be applied using xTorch's `Transform` pipeline.

The general workflow involves instantiating the dataset, which handles loading the raw sequences, and then passing it to a `DataLoader` for batching.

```cpp
#include <xtorch/xtorch.h>
#include <iostream>

int main() {
    // 1. Define any necessary transformations (optional)
    // For example, normalizing the time-series values.
    // auto transforms = std::make_unique<xt::transforms::...>();

    // 2. Instantiate a dataset for the M4 Forecasting Competition.
    auto dataset = xt::datasets::M4Competition(
        "./data",
        /*download=*/true
        // std::move(transforms)
    );

    std::cout << "M4 Competition dataset loaded." << std::endl;
    std::cout << "Number of time series: " << *dataset.size() << std::endl;

    // 3. Pass the dataset to a DataLoader
    // Batching time-series data can depend heavily on the model architecture (e.g., RNNs vs Transformers).
    xt::dataloaders::ExtendedDataLoader data_loader(dataset, 32, true);

    // The data loader is now ready for use in a training loop
    for (auto& batch : data_loader) {
        auto history = batch.first;   // Input sequence
        auto future = batch.second;  // Sequence to predict
        // ... training step with a forecasting model like an LSTM or Informer ...
    }
}
```

!!! info "Data Structure"
The exact structure of the data returned by `get(index)` can vary depending on the dataset and task. For forecasting, it's often a pair of tensors representing historical context and future values. For classification, it's a sequence and a single class label.

---

## Available Datasets by Task

### Time Series Forecasting

The task of predicting future values in a sequence given past values.

| Dataset Class | Description | Header File |
|---|---|---|
| `M4Competition` | The dataset from the 4th Makridakis Forecasting Competition, containing a large and diverse set of time series from different domains. | `time_series_forecasting/m4_competition.h` |
| `ElectricityLoadDiagrams` | The ElectricityLoadDiagrams20112014 Data Set, which contains the electricity consumption of 370 clients. | `time_series_forecasting/electricity_load_diagrams.h` |

### Time Series Classification

The task of assigning a categorical label to an entire time-series sequence.

| Dataset Class | Description | Header File |
|---|---|---|
| `UCRTimeSeriesArchive`| A large collection of datasets from the UCR Time Series Classification Archive, widely used for benchmarking classification algorithms. | `time_series_classification/ucr_time_series_archive.h` |

### Anomaly Detection

The task of identifying rare items, events, or observations which raise suspicions by differing significantly from the majority of the data.

| Dataset Class | Description | Header File |
|---|---|---|
| `NAB` | The Numenta Anomaly Benchmark, a benchmark for evaluating algorithms for streaming anomaly detection. | `anomaly_detection/nab.h` |
