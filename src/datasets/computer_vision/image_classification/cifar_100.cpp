#include "include/datasets/computer_vision/image_classification/cifar_100.h"

using namespace std;
namespace fs = std::filesystem;

namespace xt::data::datasets
{
    CIFAR100::CIFAR100(const std::string& root): CIFAR100::CIFAR100(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    CIFAR100::CIFAR100(const std::string& root, xt::datasets::DataMode mode): CIFAR100::CIFAR100(
        root, mode, false, nullptr, nullptr)
    {
    }

    CIFAR100::CIFAR100(const std::string& root, xt::datasets::DataMode mode, bool download) :
        CIFAR100::CIFAR100(root, mode, download, nullptr, nullptr)
    {
    }

    CIFAR100::CIFAR100(const std::string& root, xt::datasets::DataMode mode, bool download,
                       std::unique_ptr<xt::Module> transformer) : CIFAR100::CIFAR100(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    CIFAR100::CIFAR100(const std::string& root, xt::datasets::DataMode mode, bool download,
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



    torch::data::Example<> CIFAR100::get(size_t index)
    {
        // Clone both data and label tensors to prevent memory issues
        return {data[index].clone(), torch::tensor(targets[index])};
    }

    torch::optional<size_t> CIFAR100::size() const
    {
        // Return count of loaded samples
        return data.size();
    }

    void CIFAR100::load_data()
    {
        // CIFAR-100 has single training file (50000 samples)
        std::string file_path = (dataset_path / train_file_name).string();
        cout << "train file path : " << file_path << endl;

        // Open binary file
        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open())
        {
            std::cerr << "Failed to open file: " << file_path << std::endl;
            return;
        }

        // Process all 50000 samples in file
        for (int j = 0; j < 50000; ++j)
        {
            // Read labels (CIFAR-100 specific format)
            uint8_t label;
            file.read(reinterpret_cast<char*>(&label), sizeof(label)); // Skip coarse label (not used)
            file.read(reinterpret_cast<char*>(&label), sizeof(label)); // Read fine label (0-99)

            // Store fine-grained label (100 classes)
            targets.push_back(static_cast<int64_t>(label));

            // Read image data (32x32x3 = 3072 bytes)
            std::vector<uint8_t> image(3072);
            file.read(reinterpret_cast<char*>(image.data()), image.size());

            // Convert to tensor and reshape to 3x32x32
            auto tensor_image = torch::from_blob(image.data(), {3, 32, 32},
                                                 torch::kByte).clone(); // Clone for memory ownership

            // Permute dimensions from (C, H, W) to (C, W, H) and back to (C, H, W)
            // (Original CIFAR-100 binary format has unusual dimension ordering)
            tensor_image = tensor_image.permute({0, 2, 1});

            // Store final tensor in data vector
            data.push_back(tensor_image);
        }

        file.close(); // Explicit close (RAII would handle this but clear intent)
    }
}
