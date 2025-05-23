#include "datasets/computer_vision/image_classification/cifar_10.h"

using namespace std;
namespace fs = std::filesystem;

namespace xt::data::datasets
{
    CIFAR10::CIFAR10(const std::string& root): CIFAR10::CIFAR10(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    CIFAR10::CIFAR10(const std::string& root, xt::datasets::DataMode mode): CIFAR10::CIFAR10(
        root, mode, false, nullptr, nullptr)
    {
    }

    CIFAR10::CIFAR10(const std::string& root, xt::datasets::DataMode mode, bool download) :
        CIFAR10::CIFAR10(
            root, mode, download, nullptr, nullptr)
    {
    }

    CIFAR10::CIFAR10(const std::string& root, xt::datasets::DataMode mode, bool download,
                     std::unique_ptr<xt::Module> transformer) : CIFAR10::CIFAR10(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    CIFAR10::CIFAR10(const std::string& root, xt::datasets::DataMode mode, bool download,
                     std::unique_ptr<xt::Module> transformer,
                     std::unique_ptr<xt::Module> target_transformer) : xt::datasets::Dataset(
        mode, std::move(transformer), std::move(target_transformer))
    {
        // Same initialization as main constructor
        this->root = fs::path(root);
        this->dataset_path = this->root / this->dataset_folder_name;
        bool res = true;

        // Download handling (identical to main constructor)
        if (download)
        {
            bool should_download = false;
            if (!fs::exists(this->root / this->archive_file_name))
            {
                should_download = true;
            }
            else
            {
                std::string md5 = xt::utils::get_md5_checksum((this->root / this->archive_file_name).string());
                if (md5 != archive_file_md5)
                {
                    should_download = true;
                }
            }
            if (should_download)
            {
                auto [result, path] = xt::utils::download(this->url, this->root.string());
                res = result;
            }
            if (res)
            {
                string pth = (this->root / this->archive_file_name).string();
                res = xt::utils::extract(pth, this->root);
            }
        }

        // Load data (transforms will be applied through base class)
        load_data();
    }


    torch::data::Example<> CIFAR10::get(size_t index)
    {
        // Clone both data and label tensors to prevent memory issues
        return {data[index].clone(), torch::tensor(targets[index])};
    }

    torch::optional<size_t> CIFAR10::size() const
    {
        return data.size(); // Return count of loaded samples
    }

    void CIFAR10::load_data()
    {
        const int num_files = 5; // CIFAR-10 has 5 training batch files

        // Process each training file in sequence
        for (auto path : this->train_file_names)
        {
            // Construct full path to binary file
            std::string file_path = this->dataset_path / path;

            // Open file in binary mode
            std::ifstream file(file_path, std::ios::binary);
            if (!file.is_open())
            {
                std::cerr << "Failed to open file: " << file_path << std::endl;
                continue; // Skip to next file if open fails
            }

            // Each file contains exactly 10,000 samples
            for (int j = 0; j < 10000; ++j)
            {
                // Read single byte label (0-9)
                uint8_t label;
                file.read(reinterpret_cast<char*>(&label), sizeof(label));
                targets.push_back(static_cast<int64_t>(label)); // Store as int64

                // Read image data (32x32x3 = 3072 bytes)
                std::vector<uint8_t> image(3072);
                file.read(reinterpret_cast<char*>(image.data()), image.size());

                // Convert to tensor and reshape to 3x32x32
                auto tensor_image = torch::from_blob(image.data(), {3, 32, 32},
                                                     torch::kByte).clone(); // Clone for ownership

                // Permute dimensions from (C, H, W) to (C, W, H) and back to (C, H, W)
                // (Note: Original CIFAR-10 binary format has unusual dimension ordering)
                tensor_image = tensor_image.permute({0, 2, 1});

                // Store final tensor in data vector
                data.push_back(tensor_image);
            }

            file.close(); // Explicit close (RAII would handle this but clear intent)
        }
    }
}
