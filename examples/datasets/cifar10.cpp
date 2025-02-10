#include <torch/torch.h>
#include <torch/data/datasets/mnist.h>
#include <iostream>
#include <vector>

using namespace std;

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <torch/torch.h>

class CIFAR10 : public torch::data::Dataset<CIFAR10> {
public:
    CIFAR10(const std::string &root) {
        // Load data from the specified root directory
        load_data(root);
    }

    // Override the get method to return a sample
    torch::data::Example<> get(size_t index) override {
        // Return the tensor image and its corresponding label
        return {data[index].clone(), torch::tensor(labels[index])}; // Clone to ensure tensor validity
    }

    // Override the size method to return the number of samples
    torch::optional<size_t> size() const override {
        return data.size();
    }

private:
    std::vector<torch::Tensor> data; // Store image data as tensors
    std::vector<int64_t> labels;      // Store labels
    std::string base_folder = "cifar-10-batches-bin";
    std::string url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";
    std::string filename = "cifar-10-binary.tar.gz";


    void load_data(const std::string &root) {
        const int num_files = 5;
        for (int i = 1; i <= num_files; ++i) {
            std::string file_path = root + "/data_batch_" + std::to_string(i) + ".bin";
            std::ifstream file(file_path, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Failed to open file: " << file_path << std::endl;
                continue;
            }

            for (int j = 0; j < 10000; ++j) {
                uint8_t label;
                file.read(reinterpret_cast<char *>(&label), sizeof(label));
                labels.push_back(static_cast<int64_t>(label));

                std::vector<uint8_t> image(3072); // 32x32x3 = 3072
                file.read(reinterpret_cast<char *>(image.data()), image.size());

                // Reshape the image to 3x32x32 and convert to a Torch tensor
                auto tensor_image = torch::from_blob(image.data(), {3, 32, 32}, torch::kByte).clone(); // Clone to ensure memory management
                tensor_image = tensor_image.permute({0, 2, 1}); // Permute to get the correct order (C, H, W)

                data.push_back(tensor_image); // Store the tensor in the data vector
            }

            file.close();
        }
    }
};


//
//class CIFAR10 : public torch::data::Dataset<CIFAR10> {
//public:
//    CIFAR10(const std::string& root) {
//        // Load data from the specified root directory
//        load_data(root);
//    }
//
//    // Override the get method to return a sample
//    torch::data::Example<> get(size_t index) override {
//        return {torch::tensor(data[index]), torch::tensor(labels[index])};
//    }
//
//    // Override the size method to return the number of samples
//    torch::optional<size_t> size() const override {
//        return data.size();
//    }
//
//private:
//    std::vector<std::vector<uint8_t>> data; // Store image data
//    std::vector<int64_t> labels;             // Store labels
//
//    void load_data(const std::string& root) {
//        // Implement loading logic (e.g., reading binary files)
//        // This is a placeholder for loading CIFAR-10 data
//        // You will need to read CIFAR-10 binary files and populate data and labels
//    }
//};


int main() {
    std::string dataset_path = "/home/kami/datasets/cifar-10-batches-bin"; // Update with your path
    CIFAR10 cifar10(dataset_path);
    // Create a data loader
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(cifar10.map(torch::data::transforms::Normalize<>(0.5, 0.5)).map(
                    torch::data::transforms::Stack<>())),
            /*batch_size=*/64);

    cout << "test\n";
    for (auto &batch: *data_loader) {
        auto data = batch.data; // Get batch data
        auto targets = batch.target; // Get batch targets

        cout << targets << endl;

        // Perform operations on data and targets
    }

    return 0;
}

