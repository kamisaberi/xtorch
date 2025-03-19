#include "../../include/datasets/cifar.h"

using namespace std;
namespace fs = std::filesystem;


namespace torch::ext::data::datasets {
    //------------------ CIFAR10 ------------------//
    CIFAR10::CIFAR10(const std::string &root, DataMode mode, bool download) {
        // Load data from the specified root directory
        this->root = fs::path(root);
        this->dataset_path = this->root / this->dataset_folder_name;
        bool res = true;
        if (download) {
            bool should_download = false;
            if (!fs::exists(this->root / this->archive_file_name)) {
                should_download = true;
            } else {
                std::string md5 = torch::ext::utils::get_md5_checksum((this->root / this->archive_file_name).string());
                if (md5 != archive_file_md5) {
                    should_download = true;
                }
            }
            if (should_download) {
                auto [result, path] = torch::ext::utils::download(this->url, this->root.string());
                res = result;
            }
            if (res) {
                string pth = (this->root / this->archive_file_name).string();
                res = torch::ext::utils::extract(pth, this->root);
            }
        }
        // if (res) {
        load_data(mode);
        // }
        cout << "DATA SIZES:" << this->data.size() << endl;
    }

    torch::data::Example<> CIFAR10::get(size_t index) {
        return {data[index].clone(), torch::tensor(labels[index])}; // Clone to ensure tensor validity
    }

    torch::optional<size_t> CIFAR10::size() const {
        return data.size();
    }

    void CIFAR10::load_data(DataMode mode) {
        const int num_files = 5;
        for (auto path : this->train_file_names) {
            std::string file_path = this->dataset_path / path;
            cout << "Loading file " << file_path << endl;
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
                auto tensor_image = torch::from_blob(image.data(), {3, 32, 32},
                                                     torch::kByte).clone(); // Clone to ensure memory management
                tensor_image = tensor_image.permute(
                    {0, 2, 1}); // Permute to get the correct order (C, H, W)

                data.push_back(tensor_image); // Store the tensor in the data vector
            }

            file.close();

        }
        // for (int i = 1; i <= num_files; ++i) {
        //     std::string file_path = root / "/data_batch_" / std::to_string(i) / ".bin";
        //     cout << "Loading file " << file_path << endl;
        //     std::ifstream file(file_path, std::ios::binary);
        //     if (!file.is_open()) {
        //         std::cerr << "Failed to open file: " << file_path << std::endl;
        //         continue;
        //     }
        //
        //     for (int j = 0; j < 10000; ++j) {
        //         uint8_t label;
        //         file.read(reinterpret_cast<char *>(&label), sizeof(label));
        //         labels.push_back(static_cast<int64_t>(label));
        //
        //         std::vector<uint8_t> image(3072); // 32x32x3 = 3072
        //         file.read(reinterpret_cast<char *>(image.data()), image.size());
        //
        //         // Reshape the image to 3x32x32 and convert to a Torch tensor
        //         auto tensor_image = torch::from_blob(image.data(), {3, 32, 32},
        //                                              torch::kByte).clone(); // Clone to ensure memory management
        //         tensor_image = tensor_image.permute(
        //             {0, 2, 1}); // Permute to get the correct order (C, H, W)
        //
        //         data.push_back(tensor_image); // Store the tensor in the data vector
        //     }
        //
        //     file.close();
        // }
    }

    //------------------ CIFAR100 ------------------//
    CIFAR100::CIFAR100(const std::string &root, DataMode mode, bool download) {
        // Load data from the specified root directory
        this->root = fs::path(root);
        this->dataset_path = this->root / this->dataset_folder_name;
        bool res = true;
        if (download) {
            bool should_download = false;
            if (!fs::exists(this->root / this->archive_file_name)) {
                should_download = true;
            } else {
                std::string md5 = torch::ext::utils::get_md5_checksum((this->root / this->archive_file_name).string());
                if (md5 != archive_file_md5) {
                    should_download = true;
                }
            }
            if (should_download) {
                auto [result, path] = torch::ext::utils::download(this->url, this->root.string());
                res = result;
            }
            if (res) {
                string pth = (this->root / this->archive_file_name).string();
                res = torch::ext::utils::extract(pth, this->root);
            }
        }
        if (res) {
            load_data(mode);
        }
    }

    torch::data::Example<> CIFAR100::get(size_t index) {
        return {data[index].clone(), torch::tensor(labels[index])}; // Clone to ensure tensor validity
    }

    torch::optional<size_t> CIFAR100::size() const {
        return data.size();
    }

    void CIFAR100::load_data(DataMode mode) {
        const int num_files = 5;
        std::string file_path = (dataset_path / train_file_name).string();
        cout << "train file path : " << file_path << endl;
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
}
