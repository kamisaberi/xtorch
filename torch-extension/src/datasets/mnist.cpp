#include "../../include/datasets/mnist.h"

namespace torch::ext::data::datasets {

    std::vector<std::vector<uint8_t>> read_mnist_images(const std::string &file_path, int num_images) {
        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + file_path);
        }

        // Read metadata
        int32_t magic_number, num_items, rows, cols;
        file.read(reinterpret_cast<char *>(&magic_number), 4);
        file.read(reinterpret_cast<char *>(&num_items), 4);
        file.read(reinterpret_cast<char *>(&rows), 4);
        file.read(reinterpret_cast<char *>(&cols), 4);

        // Convert endianess
        magic_number = __builtin_bswap32(magic_number);
        num_items = __builtin_bswap32(num_items);
        rows = __builtin_bswap32(rows);
        cols = __builtin_bswap32(cols);

        std::vector<std::vector<uint8_t>> images(num_images, std::vector<uint8_t>(rows * cols));
        for (int i = 0; i < num_images; i++) {
            file.read(reinterpret_cast<char *>(images[i].data()), rows * cols);
        }

        file.close();
        return images;
    }

    std::vector<uint8_t> read_mnist_labels(const std::string &file_path, int num_labels) {
        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + file_path);
        }

        // Read metadata
        int32_t magic_number, num_items;
        file.read(reinterpret_cast<char *>(&magic_number), 4);
        file.read(reinterpret_cast<char *>(&num_items), 4);

        // Convert endianess
        magic_number = __builtin_bswap32(magic_number);
        num_items = __builtin_bswap32(num_items);

        std::vector<uint8_t> labels(num_labels);
        file.read(reinterpret_cast<char *>(labels.data()), num_labels);

        file.close();
        return labels;
    }


    //------------------ MNIST ------------------//
    MNIST::MNIST(const std::string &root, bool train, bool download) {

        bool res = true;
        for (const auto &resource: this->resources) {
            auto pth = std::get<0>(resource);
            auto md = std::get<1>(resource);
        }

//        // Load data from the specified root directory
//        this->root = fs::path(root);
//        this->dataset_path = this->root / this->dataset_folder_name;

//        if (download) {
//            bool should_download = false;
//            if (!fs::exists(this->root / this->archive_file_name)) {
//                should_download = true;
//            } else {
//                std::string md5 = md5File((this->root / this->archive_file_name).string());
//                if (md5 != archive_file_md5) {
//                    should_download = true;
//                }
//            }
//            if (should_download) {
//                auto [result, path] = download_data(this->download_url, this->root.string());
//                res = result;
//            }
//            if (res) {
//                string pth = (this->root / this->archive_file_name).string();
//                res = extract(pth, this->root);
//            }
//        }
        if (res) {
            load_data(train);
        }
    }

    torch::data::Example<> MNIST::get(size_t index) {
        return {data[index].clone(), torch::tensor(labels[index])}; // Clone to ensure tensor validity
    }

    torch::optional<size_t> MNIST::size() const {
        return data.size();
    }

    void MNIST::load_data(bool train) {


        const int num_files = 5;
        for (int i = 1; i <= num_files; ++i) {
            std::string file_path = root / "/data_batch_" / std::to_string(i) / ".bin";
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
    }



    FashionMNIST::FashionMNIST(const std::string &images_path, const std::string &labels_path,
                               int num_samples) {
        auto images_data = read_mnist_images(images_path, num_samples);
        auto labels_data = read_mnist_labels(labels_path, num_samples);

        images_ = torch::empty({num_samples, 1, 28, 28}, torch::kUInt8);
        labels_ = torch::empty(num_samples, torch::kUInt8);

        for (int i = 0; i < num_samples; i++) {
            images_[i] = torch::from_blob(images_data[i].data(), {1, 28, 28}, torch::kUInt8).clone();
            labels_[i] = labels_data[i];
        }

        images_ = images_.to(torch::kFloat32).div_(255.0); // Normalize to [0, 1]
        labels_ = labels_.to(torch::kInt64);               // Convert to int64 for loss functions
    }

    // Override `get` method to return a single data sample
    torch::data::Example<> FashionMNIST::get(size_t index) {
        return {images_[index], labels_[index]};
    }

    // Override `size` method to return the number of samples
    torch::optional<size_t> FashionMNIST::size() const {
        return labels_.size(0);
    }


    KMINST::KMINST(const std::string &images_path, const std::string &labels_path,
                               int num_samples) {
        auto images_data = read_mnist_images(images_path, num_samples);
        auto labels_data = read_mnist_labels(labels_path, num_samples);

        images_ = torch::empty({num_samples, 1, 28, 28}, torch::kUInt8);
        labels_ = torch::empty(num_samples, torch::kUInt8);

        for (int i = 0; i < num_samples; i++) {
            images_[i] = torch::from_blob(images_data[i].data(), {1, 28, 28}, torch::kUInt8).clone();
            labels_[i] = labels_data[i];
        }

        images_ = images_.to(torch::kFloat32).div_(255.0); // Normalize to [0, 1]
        labels_ = labels_.to(torch::kInt64);               // Convert to int64 for loss functions
    }

    // Override `get` method to return a single data sample
    torch::data::Example<> KMINST::get(size_t index) {
        return {images_[index], labels_[index]};
    }

    // Override `size` method to return the number of samples
    torch::optional<size_t> KMINST::size() const {
        return labels_.size(0);
    }


}
