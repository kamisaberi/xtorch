#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include "../include/datasets/image-classification/mnist.h"
#include "../include/models/cnn/lenet5.h"
#include "../include/definitions/transforms.h"
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <optional>


// Custom dataset that produces one Example (data, target) per sample
struct MyDataset : torch::data::datasets::Dataset<MyDataset, torch::data::Example<>> {
    size_t dataset_size;
    MyDataset(size_t size) : dataset_size(size) {}

    // Return an Example containing data and target for the given index
    torch::data::Example<> get(size_t index) override {
        // Dummy data: 1-D tensor of length 3, dummy target: scalar tensor (e.g., class label)
        auto data   = torch::tensor({float(index), float(index) + 0.1f, float(index) + 0.2f});
        auto target = torch::tensor({static_cast<long>(index % 2)});  // e.g., binary label 0 or 1
        target = target.squeeze();  // make target a 0-dim scalar tensor
        return {data, target};
    }

    // Return number of samples in dataset
    torch::optional<size_t> size() const override {
        return dataset_size;
    }
};

// Custom DataLoader for a pre-batched (Stack-transformed) dataset
template <typename Dataset>
class CustomDataLoader : public torch::data::DataLoaderBase<Dataset, typename Dataset::BatchType, std::vector<size_t>> {
    using BatchType        = typename Dataset::BatchType;          // e.g., Example<Tensor, Tensor>
    using BatchRequestType = std::vector<size_t>;                  // list of indices for one batch
    using Base = torch::data::DataLoaderBase<Dataset, BatchType, BatchRequestType>;

public:
    CustomDataLoader(Dataset dataset, const torch::data::DataLoaderOptions& options, bool shuffle = false)
        : Base(options, std::make_unique<Dataset>(std::move(dataset))), shuffle_(shuffle) {
        // Only single-thread (workers=0) is supported in this custom loader
        if (options.workers() != 0) {
            throw std::runtime_error("CustomDataLoader supports only workers=0 (single-threaded)");
        }
        dataset_ptr_ = Base::main_thread_dataset_.get();      // pointer to dataset (stored in base)
        batch_size_  = options.batch_size();                  // batch size per iteration
        drop_last_   = options.drop_last();                   // whether to drop last incomplete batch
        reset_indices();                                      // initialize index sequence
    }

//    // Iterator support for range-for loops
//    typename Base::iterator begin() {
//        this->reset();       // reset (and shuffle if needed) at start of epoch
//        return Base::begin();
//    }
//    typename Base::iterator end() {
//        return Base::end();
//    }

protected:
    // Provide the next batch of indices to fetch from the dataset
    std::optional<BatchRequestType> get_batch_request() override {
        if (current_index_ >= indices_.size()) {
            // No more indices -> signal end of data
            return std::nullopt;
        }
        // Determine the range [start_index, end_index) for the next batch of indices
        size_t start_index = current_index_;
        size_t end_index   = std::min(current_index_ + batch_size_, indices_.size());
        // If drop_last_ is true and the remaining indices are fewer than batch_size, stop here
        if (drop_last_ && (end_index - start_index) < batch_size_) {
            return std::nullopt;
        }
        // Collect indices for this batch and advance the pointer
        BatchRequestType batch_indices(indices_.begin() + start_index, indices_.begin() + end_index);
        current_index_ = end_index;
        return batch_indices;
    }

    // Reset and (optionally) shuffle indices for a new epoch
    void reset_indices() {
        const size_t N = dataset_ptr_->size().value();
        indices_.resize(N);
        std::iota(indices_.begin(), indices_.end(), 0);  // fill with 0,1,...,N-1
        if (shuffle_) {
            // Shuffle the indices to randomize batch order
            static std::mt19937 rng(std::random_device{}());  // fixed seeded RNG for reproducibility
            std::shuffle(indices_.begin(), indices_.end(), rng);
        }
        current_index_ = 0;
    }

    // Override base class reset() to shuffle indices each epoch (if enabled)
    void reset() override {
        reset_indices();
        Base::reset();  // let DataLoaderBase handle internal reset (e.g., for iterator state)
    }


private:
    Dataset* dataset_ptr_;           // raw pointer to the dataset (owned by base class)
    std::vector<size_t> indices_;    // sequence of indices to iterate over
    size_t current_index_ = 0;       // current position in indices_ vector
    size_t batch_size_;
    bool shuffle_;
    bool drop_last_;
};

int main() {
    // Create a dataset of 10 samples and apply the Stack<> transform to batch its outputs
//    MyDataset base_dataset(10);
    cout << "1-base_dataset..."  << endl;

    auto normalize_fn = torch::data::transforms::Normalize<>(0.5, 0.5);
    auto resize_fn  = xt::data::transforms::create_resize_transform({32,32});
//    auto compose = xt::data::transforms::Compose({resize_fn, normalize_fn});

    auto base_dataset = xt::data::datasets::MNIST("/home/kami/Documents/temp/", DataMode::TRAIN, true,{resize_fn, normalize_fn});

//    auto stacked_dataset = base_dataset.map(torch::data::transforms::Stack<>());

    cout << "2-base_dataset..."  << endl;
    auto stacked_dataset = base_dataset
//        .map(xt::data::transforms::resize({32, 32}))
//        .map(xt::data::transforms::normalize(0.5, 0.5))
        .map(torch::data::transforms::Stack<>());

    //return 0 ;
    // Configure DataLoaderOptions (batch size 4, single thread, and don't drop last batch)
    auto options = torch::data::DataLoaderOptions().batch_size(64).drop_last(false);
    // Instantiate CustomDataLoader with shuffle enabled
    CustomDataLoader loader(std::move(stacked_dataset), options, /*shuffle=*/true);

//    loader.reset();  // start of epoch
//
//    while (auto batch = loader.next()) {
//        std::cout << "Batch data size: " << batch->data.sizes()
//                  << ", Batch target size: " << batch->target.sizes() << "\n";
//        std::cout << "Data:\n" << batch->data << "\n";
//        std::cout << "Targets:\n" << batch->target << "\n";
//        std::cout << "-------------------------\n";
//    }

    std::vector<int64_t> size = {32, 32};
    std::cout.precision(10);
    torch::Device device(torch::kCPU);
    xt::models::LeNet5 model(10);
    model.to(device);
    model.train();
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
    for (size_t epoch = 0; epoch != 10; ++epoch) {
            cout << "epoch: " << epoch << endl;
            for (auto& batch : loader) {
                // Each `batch` is an Example<Tensor, Tensor> with shapes [batch_size, ...]
//                std::cout << "Batch data size: "    << batch.data.sizes() << ", Batch target size: " << batch.target.sizes() << "\n";
        //        std::cout << "Data:\n"    << batch.data << "\n";
        //        std::cout << "Targets:\n" << batch.target << "\n";
//                std::cout << "-------------------------\n";
                torch::Tensor data, targets;
                data = batch.data;
                targets = batch.target;
                optimizer.zero_grad();
                torch::Tensor output;
                output = model.forward(data);
                torch::Tensor loss;
                loss = torch::nll_loss(output, targets);
                loss.backward();
                optimizer.step();
//                std::cout << "Epoch: " << epoch << " | Batch: " <<  " | Loss: " << loss.item<float>() <<                            std::endl;

//            }

        }
    }


//    std::cout << "Iterating over CustomDataLoader:\n";
//    for (auto& batch : loader) {
//        // Each `batch` is an Example<Tensor, Tensor> with shapes [batch_size, ...]
//        std::cout << "Batch data size: "    << batch.data.sizes()
//                  << ", Batch target size: " << batch.target.sizes() << "\n";
////        std::cout << "Data:\n"    << batch.data << "\n";
////        std::cout << "Targets:\n" << batch.target << "\n";
//        std::cout << "-------------------------\n";
//    }
    return 0;
}
