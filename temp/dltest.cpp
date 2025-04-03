#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <optional>

// Define CustomDataLoader as above
template <typename Dataset, typename Sampler>
class CustomDataLoader : public torch::data::detail::DataLoaderBase<Dataset, std::vector<size_t>, std::vector<torch::data::Example<>>> {
public:
    CustomDataLoader(Dataset dataset, Sampler sampler, const torch::data::DataLoaderOptions& options)
        : torch::data::detail::DataLoaderBase<Dataset, std::vector<size_t>, std::vector<torch::data::Example<>>>(std::move(dataset), std::move(sampler), options) {
    }

    std::optional<std::vector<size_t>> get_batch_request() override {
        return sampler_.next(options_.batch_size);
    }

    void printLoaderInfo() const {
        std::cout << "Custom DataLoader initialized with dataset size: "
                  << this->dataset().size().value() << std::endl;
    }
};

// Simple dataset for testing
class SimpleDataset : public torch::data::Dataset<SimpleDataset> {
public:
    SimpleDataset(size_t size) : size_(size) {}

    torch::data::Example<> get(size_t index) override {
        return {torch::tensor(static_cast<float>(index)), torch::tensor(static_cast<float>(index))};
    }

    torch::optional<size_t> size() const override {
        return size_;
    }

private:
    size_t size_;
};

int main() {
    // Initialize dataset and sampler
    SimpleDataset dataset(5);
    torch::data::samplers::SequentialSampler sampler(5);

    // Set data loader options
    auto options = torch::data::DataLoaderOptions().batch_size(2);

    // Instantiate the custom data loader
    CustomDataLoader<SimpleDataset, torch::data::samplers::SequentialSampler> data_loader(
        std::move(dataset), std::move(sampler), options);

    // Test the custom method
    data_loader.printLoaderInfo();

    // Iterate over the data loader
    std::cout << "Iterating over the DataLoader:\n";
    for (auto& batch : data_loader) {
        for (const auto& example : batch) {
            std::cout << "Data: " << example.data.item<float>()
                      << ", Target: " << example.target.item<float>() << std::endl;
        }
    }

    return 0;
}