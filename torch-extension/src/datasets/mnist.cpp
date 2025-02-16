#include "../../include/datasets/mnist.h"

namespace torch::ext::data::datasets {


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


}
