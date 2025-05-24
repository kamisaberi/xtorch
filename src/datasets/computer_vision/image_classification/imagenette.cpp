#include "include/datasets/computer_vision/image_classification/imagenette.h"

namespace xt::datasets
{
    Imagenette::Imagenette(const std::string& root): Imagenette::Imagenette(
        root, xt::datasets::DataMode::TRAIN, false, ImageType::PX160, nullptr, nullptr)
    {
    }

    Imagenette::Imagenette(const std::string& root, xt::datasets::DataMode mode): Imagenette::Imagenette(
        root, mode, false, ImageType::PX160, nullptr, nullptr)
    {
    }

    Imagenette::Imagenette(const std::string& root, xt::datasets::DataMode mode, bool download): Imagenette::Imagenette(
        root, mode, download, ImageType::PX160, nullptr, nullptr)
    {
    }

    Imagenette::Imagenette(const std::string& root, xt::datasets::DataMode mode, bool download, ImageType type):
        Imagenette::Imagenette(
            root, mode, download, type, nullptr, nullptr)
    {
    }

    Imagenette::Imagenette(const std::string& root, xt::datasets::DataMode mode, bool download, ImageType type,
                           std::unique_ptr<xt::Module> transformer): Imagenette::Imagenette(
        root, mode, download, type, std::move(transformer), nullptr)
    {
    }

    Imagenette::Imagenette(const std::string& root, xt::datasets::DataMode mode, bool download, ImageType type,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources(root, download);
        load_data();
    }


    void Imagenette::check_resources(const std::string& root, bool download)
    {
        // Convert root path to filesystem path
        this->root = fs::path(root);

        // Verify root directory exists
        if (!fs::exists(this->root))
        {
            throw runtime_error("Dataset root path does not exist");
        }

        // Set up dataset directory path
        this->dataset_path = this->root / this->dataset_folder_name;

        // Create dataset directory if it doesn't exist
        if (!fs::exists(this->dataset_path))
        {
            fs::create_directories(this->dataset_path);
        }

        // Get resource information based on selected image type
        auto [url, dataset_filename, folder_name, md] = this->resources[getImageTypeValue(this->type)];
        fs::path fpth = this->dataset_path / dataset_filename;

        // Verify file exists and has correct checksum
        if (!(fs::exists(fpth) && xt::utils::get_md5_checksum(fpth.string()) == md))
        {
            if (download)
            {
                // Download dataset if enabled
                string u = url.string();
                auto [r, path] = xt::utils::download(u, this->dataset_path.string());
            }
            else
            {
                throw runtime_error("Dataset resources not found. Set download=true to download automatically");
            }
        }

        // Extract downloaded archive
        xt::utils::extractTgz(fpth, this->dataset_path.string());
    }

    void Imagenette::load_data(DataMode mode)
    {
        // Get resource information based on image type
        auto [url, dataset_filename, folder_name, md] = this->resources[getImageTypeValue(this->type)];

        if (mode == DataMode::TRAIN)
        {
            // Set path to training data
            fs::path path = this->dataset_path / folder_name / fs::path("train");

            // Iterate through each class directory
            for (auto& p : fs::directory_iterator(path))
            {
                if (fs::is_directory(p.path()))
                {
                    // Add class name to label list
                    string u = p.path().filename().string();
                    labels_name.push_back(u);

                    // Process each image in class directory
                    for (auto& img : fs::directory_iterator(p.path()))
                    {
                        // Convert image to tensor using OpenCV
                        torch::Tensor tensor = torch::ext::media::opencv::convertImageToTensor(img.path());

                        // Apply transforms if specified
                        if (transformer != nullptr)
                        {
                            tensor = (*transformer)(tensor);
                        }

                        // Store tensor and corresponding label
                        data.push_back(tensor);
                        targets.push_back(labels_name.size() - 1); // Use label index
                    }
                }
            }
        }
        else
        {
            // Set path to validation data
            fs::path path = this->dataset_path / folder_name / fs::path("val");

            // Iterate through each class directory
            for (auto& p : fs::directory_iterator(path))
            {
                if (fs::is_directory(p.path()))
                {
                    // Add class name to label list
                    string u = p.path().filename().string();
                    labels_name.push_back(u);

                    // Process each image in class directory
                    for (auto& img : fs::directory_iterator(p.path()))
                    {
                        // Convert image to tensor using OpenCV
                        torch::Tensor tensor = torch::ext::media::opencv::convertImageToTensor(img.path());

                        // Apply transforms if specified
                        if (transformer != nullptr)
                        {
                            tensor = (*transformer)(tensor);
                        }

                        // Store tensor and corresponding label
                        data.push_back(tensor);
                        targets.push_back(labels_name.size() - 1); // Use label index
                    }
                }
            }
        }
    }
} // namespace xt::datasets
