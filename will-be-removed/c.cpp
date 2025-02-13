#include "c.h"


namespace torch::ext::data::datasets {

    CIFAR100_T::CIFAR100_T(const std::string &root, bool train, bool download) {
        // Load data from the specified root directory
        this->root = fs::path(root);
        this->dataset_path = this->root / this->dataset_folder_name;

//                this->dataset_raw_path=this->dataset_path  /  fs::path("raw/");
//    if (download) {
//        auto [result, path] = download_data(this->download_url, this->root.string());
//        if (result) {
//            string pth = (this->root / this->archive_file_name).string();
//            extract(pth, this->root);
//        }
//    }

        load_data(root, train);
    }

    torch::data::Example<> CIFAR100_T::get(size_t index) {
// Return the tensor image and its corresponding label
        return {data[index].clone(), torch::tensor(labels[index])}; // Clone to ensure tensor validity
    }

//
    torch::optional<size_t> CIFAR100_T::size() const {
        return data.size();
    }

    void CIFAR100_T::load_data(const std::string &root, bool train) {
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


}


