#include <torch/torch.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <filesystem>
#include "../includes/base.h"

using namespace std;
namespace fs = std::filesystem;

namespace torch {
    namespace data {
        namespace datasets {

            class CIFAR1002 : public torch::data::Dataset<CIFAR1002> {
            public:
                CIFAR1002(const std::string &root, bool train = true, bool download = false) {
                    // Load data from the specified root directory
                    this->root = fs::path(root);
                    this->dataset_path = this->root / this->dataset_folder_name;

//                this->dataset_raw_path=this->dataset_path  /  fs::path("raw/");
                    if (download) {
                        auto [result, path] = download_data(this->download_url, this->root.string());
                        if (result) {
                            string pth = (this->root / this->archive_file_name).string();
                            extract(pth, this->root);
                        }
                    }

                    load_data(root, train);
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
                std::string download_url = "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz";
                fs::path archive_file_name = "cifar-100-binary.tar.gz";
                fs::path root;
                fs::path dataset_path;
//            fs::path dataset_raw_path;
                fs::path dataset_folder_name = "cifar-100-binary";


                void load_data(const std::string &root, bool train = true) {
                    const int num_files = 5;
                    std::string file_path = root + "/train.bin";
                    std::ifstream file(file_path, std::ios::binary);
                    if (!file.is_open()) {
                        std::cerr << "Failed to open file: " << file_path << std::endl;
                        return;
                    }

                    for (int j = 0; j < 50000; ++j) {
                        uint8_t label;
                        file.read(reinterpret_cast<char *>(&label), sizeof(label));
                        file.read(reinterpret_cast<char *>(&label), sizeof(label));

                        labels.push_back(static_cast<int64_t>(label));

                        std::vector<uint8_t> image(3072); // 32x32x3 = 3072
                        file.read(reinterpret_cast<char *>(image.data()), image.size());

                        // Reshape the image to 3x32x32 and convert to a Torch tensor
                        auto tensor_image = torch::from_blob(image.data(), {3, 32, 32},
                                                             torch::kByte).clone(); // Clone to ensure memory management
                        tensor_image = tensor_image.permute({0, 2, 1}); // Permute to get the correct order (C, H, W)

                        data.push_back(tensor_image); // Store the tensor in the data vector
                    }

                    file.close();
                }

            };


        }
    }
}

int main() {
    std::string dataset_path = "/home/kami/datasets/cifar-100-binary"; // Update with your path
    torch::data::datasets::CIFAR1002 cifar1002(dataset_path);
    // Create a data loader
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(cifar1002.map(torch::data::transforms::Normalize<>(0.5, 0.5)).map(
                    torch::data::transforms::Stack<>())),
            /*batch_size=*/64);

    int cnt = 0;
    for (auto &batch: *data_loader) {
        auto data = batch.data; // Get batch data
        auto targets = batch.target; // Get batch targets

        cout << targets << endl;
        if (cnt == 1) {
            break;
        }

        cnt++;

    }

    return 0;
}
