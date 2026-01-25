#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

// ============================
// Example Toy Dataset
// ============================
struct MyDataset : torch::data::datasets::Dataset<MyDataset, torch::data::Example<>> {
    size_t dataset_size;

    MyDataset(size_t size) : dataset_size(size) {}

    torch::data::Example<> get(size_t index) override {
        // Dummy data: 3-element tensor
        auto data = torch::tensor({float(index), float(index + 0.1f), float(index + 0.2f)});
        auto target = torch::tensor(static_cast<int64_t>(index % 2));
        return {data, target};
    }

    torch::optional<size_t> size() const override {
        return dataset_size;
    }
};

// ============================
// CustomDataLoader for Stacked Dataset
// ============================
template <typename Dataset>
class CustomDataLoader : public torch::data::DataLoaderBase<CustomDataLoader<Dataset>, Dataset> {
public:
    using BatchType = torch::data::Example<>;

    CustomDataLoader(Dataset dataset, const torch::data::DataLoaderOptions& options, bool shuffle = false)
        : dataset_(std::move(dataset)),
          batch_size_(options.batch_size()),
          drop_last_(options.drop_last()),
          shuffle_(shuffle) {

        size_t N = dataset_.size().value();
        indices_.resize(N);
        std::iota(indices_.begin(), indices_.end(), 0);

        if (shuffle_) {
            std::shuffle(indices_.begin(), indices_.end(), std::mt19937(std::random_device{}()));
        }
    }

    void reset()  {
        current_index_ = 0;
        if (shuffle_) {
            std::shuffle(indices_.begin(), indices_.end(), std::mt19937(std::random_device{}()));
        }
    }

    torch::optional<BatchType> next()  {
        if (current_index_ >= indices_.size()) {
            return torch::nullopt;
        }

        size_t start = current_index_;
        size_t end = std::min(current_index_ + batch_size_, indices_.size());

        if (drop_last_ && (end - start) < batch_size_) {
            return torch::nullopt;
        }

        std::vector<torch::data::Example<>> batch;
        for (size_t i = start; i < end; ++i) {
            batch.push_back(dataset_.get(indices_[i]));
        }

        current_index_ = end;

        // Since the dataset is stacked, we can return the stacked batch
        std::vector<torch::Tensor> data_list, target_list;
        for (const auto& item : batch) {
            data_list.push_back(item.data);
            target_list.push_back(item.target);
        }

        return torch::data::Example<>{
            torch::stack(data_list),
            torch::stack(target_list).squeeze()  // make targets [batch_size]
        };
    }

private:
    Dataset dataset_;
    std::vector<size_t> indices_;
    size_t current_index_ = 0;
    size_t batch_size_;
    bool drop_last_;
    bool shuffle_;
};

// ============================
// Main Function
// ============================
int main() {
    // Create dataset of 10 items and apply Stack<> so each get() returns a batch
    MyDataset base_dataset(10);
    auto stacked_dataset = base_dataset.map(torch::data::transforms::Stack<>());

    // Create custom data loader for stacked dataset
    auto options = torch::data::DataLoaderOptions().batch_size(4).drop_last(false);
    CustomDataLoader loader(stacked_dataset, options, /*shuffle=*/true);

    std::cout << "Iterating through batches:\n";
    loader.reset();

    while (auto batch = loader.next()) {
        std::cout << "Batch data size: " << batch->data.sizes()
                  << ", Batch target size: " << batch->target.sizes() << "\n";
        std::cout << "Data:\n" << batch->data << "\n";
        std::cout << "Targets:\n" << batch->target << "\n";
        std::cout << "----------------------------\n";
    }

    return 0;
}
