#include "../include/datasets/image_classification/mnist.h"
#include "../include/models/computer_vision/image_classification/lenet5.h"
#include "../include/definitions/transforms.h"
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <optional>


using namespace std;
// using Example = torch::data::Example<torch::Tensor, torch::Tensor>;


template<typename Dataset>
class CustomDataLoader : public torch::data::DataLoaderBase<Dataset, typename Dataset::BatchType, std::vector<
            size_t> > {
    using Base = torch::data::DataLoaderBase<Dataset, typename Dataset::BatchType, std::vector<size_t> >;

public:
    CustomDataLoader(Dataset dataset, torch::data::DataLoaderOptions options)
        : Base(options), sampler_(dataset.size().value_or(0)) {
        // cout <<dataset << endl;
        if (this->options_.workers != 0) {
            throw std::runtime_error("CustomDataLoader: workers must be 0 for single-threaded loading.");
        }
        this->main_thread_dataset_ = std::make_unique<Dataset>(std::move(dataset));
    }

    ~CustomDataLoader() {
        this->join();
    }

protected:
    torch::data::samplers::SequentialSampler sampler_;

    std::optional<std::vector<size_t> > get_batch_request() override {
        auto indices = sampler_.next(this->options_.batch_size);
        if (!indices.has_value()) {
            return std::nullopt;
        }
        if (indices->size() < this->options_.batch_size && this->options_.drop_last) {
            return std::nullopt;
        }
        return indices;
    }

    void reset() override {
        sampler_.reset();
        Base::reset();
    }

public:
    std::optional<torch::data::Example<> > next_batch() {
        cout << "0-reading images..." << endl;
        std::optional<typename Dataset::BatchType> batch = Base::next();
        if (!batch.has_value()) {
            return std::nullopt;
        }

        const std::vector<torch::data::Example<> > &examples = *batch; // reference to the vector of samples
        size_t batch_size = examples.size();
        std::vector<torch::Tensor> image_tensors;
        std::vector<torch::Tensor> label_tensors;
        image_tensors.reserve(batch_size);
        label_tensors.reserve(batch_size);
        cout << "1-reading images..." << endl;

        for (const auto &example: examples) {
            image_tensors.push_back(example.data);
            label_tensors.push_back(example.target);
        }

        cout << "2-reading images..." << endl;
        torch::Tensor images_batch = torch::stack(image_tensors);
        torch::Tensor labels_batch = torch::stack(label_tensors);
        return torch::data::Example<>{images_batch, labels_batch};
    }
};


int main() {
    std::vector<int64_t> size = {32, 32};

    std::cout.precision(10);
    torch::Device device(torch::kCPU);

    vector<std::function<torch::Tensor(torch::Tensor)>> transforms;
    auto dataset = xt::data::datasets::MNIST("/home/kami/Documents/temp/", DataMode::TRAIN, true, transforms);

    auto transformed_dataset = dataset
            .map(xt::data::transforms::resize({32, 32}))
            .map(xt::data::transforms::normalize(0.5, 0.5))
            .map(torch::data::transforms::Stack<>());

    auto train_loader1 = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(transformed_dataset), 64);

    CustomDataLoader<decltype(transformed_dataset)> train_loader(transformed_dataset,
                                                                 torch::data::DataLoaderOptions().batch_size(64).
                                                                 workers(0));
    // cout << "Training ..." << endl;
    // cout << train_loader.next_batch().value().data << endl;


    // size_t batch_index = 0;
    // while (true) {
    //   auto batch = train_loader.next_batch();
    //   if (!batch.has_value()) {
    //     // No more batches available.
    //     break;
    //   }
    //
    //   // Each `batch` is a torch::data::Example<> where:
    //   //   batch->data  is a tensor containing a batch of images (shape: [batch_size, channels, height, width])
    //   //   batch->target is a tensor containing a batch of labels (shape: [batch_size])
    //   std::cout << "Batch " << batch_index++ << " - "
    //             << "Loaded " << batch->data.size(0) << " images, "
    //             << "Batch image tensor size: " << batch->data.sizes() << ", "
    //             << "Batch label tensor size: " << batch->target.sizes() << std::endl;
    //
    //   // Example usage: access the first image and label of the batch
    //   torch::Tensor first_image = batch->data[0];
    //   torch::Tensor first_label = batch->target[0];
    //   std::cout << "  First label in batch: " << first_label.item<int>() << std::endl;
    //
    //   // (You can now feed `batch->data` and `batch->target` into a model for training or inference)
    // }


    xt::models::LeNet5 model(10);
    model.to(device);
    model.train();
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
    for (size_t epoch = 0; epoch != 10; ++epoch) {
        cout << "epoch: " << epoch << endl;
        size_t batch_index = 0;
        auto train_loader_iterator = train_loader.begin();
        // cout <<  train_loader_iterator->data << endl;
        auto train_loader_end = train_loader.end();
        while (train_loader_iterator != train_loader_end) {
            torch::Tensor data, targets;
            auto batch = train_loader_iterator;
            data = batch->data;
            targets = batch->target;
            optimizer.zero_grad();
            torch::Tensor output;
            output = model.forward(data);
            torch::Tensor loss;
            loss = torch::nll_loss(output, targets);
            loss.backward();
            optimizer.step();
            if (++batch_index % 100 == 0) {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index << " | Loss: " << loss.item<float>() <<
                        std::endl;
            }
            ++train_loader_iterator;
        }
    }

    return 0;
}
