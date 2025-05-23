#include "datasets/computer_vision/image_classification/imagenette.h"

namespace xt::data::datasets {

    Imagenette::Imagenette(const std::string &root): Imagenette::Imagenette(
        root, DataMode::TRAIN, false, ImageType::PX160) {
    }

    Imagenette::Imagenette(const std::string &root, DataMode mode): Imagenette::Imagenette(
        root, mode, false, ImageType::PX160) {
    }

    Imagenette::Imagenette(const std::string &root, DataMode mode, bool download): Imagenette::Imagenette(
        root, mode, download, ImageType::PX160) {
    }

    Imagenette::Imagenette(const std::string &root, DataMode mode, bool download, ImageType type)
        : BaseDataset(root, mode, download) , type(type) {
        // Verify dataset resources and download if needed
        check_resources(root, download);
        // Load data according to specified mode
        load_data(mode);
    }

    Imagenette::Imagenette(const std::string &root, DataMode mode, bool download, ImageType type,
                           TransformType transforms): BaseDataset(
        root, mode, download) , type(type) {
        // Initialize transforms if provided
        if (!transforms.empty()) {
            this->transforms = transforms;
            // this->compose = xt::transforms::Compose(transforms);
        }
        // Verify dataset resources
        check_resources(root, download);
        // Load data according to specified mode
        load_data(mode);
    }

    void Imagenette::check_resources(const std::string &root, bool download) {
        // Convert root path to filesystem path
        this->root = fs::path(root);

        // Verify root directory exists
        if (!fs::exists(this->root)) {
            throw runtime_error("Dataset root path does not exist");
        }

        // Set up dataset directory path
        this->dataset_path = this->root / this->dataset_folder_name;

        // Create dataset directory if it doesn't exist
        if (!fs::exists(this->dataset_path)) {
            fs::create_directories(this->dataset_path);
        }

        // Get resource information based on selected image type
        auto [url, dataset_filename, folder_name, md] = this->resources[getImageTypeValue(this->type)];
        fs::path fpth = this->dataset_path / dataset_filename;

        // Verify file exists and has correct checksum
        if (!(fs::exists(fpth) && xt::utils::get_md5_checksum(fpth.string()) == md)) {
            if (download) {
                // Download dataset if enabled
                string u = url.string();
                auto [r, path] = xt::utils::download(u, this->dataset_path.string());
            } else {
                throw runtime_error("Dataset resources not found. Set download=true to download automatically");
            }
        }

        // Extract downloaded archive
        xt::utils::extractTgz(fpth, this->dataset_path.string());
    }

    void Imagenette::load_data(DataMode mode) {
        // Get resource information based on image type
        auto [url, dataset_filename, folder_name, md] = this->resources[getImageTypeValue(this->type)];

        if (mode == DataMode::TRAIN) {
            // Set path to training data
            fs::path path = this->dataset_path / folder_name / fs::path("train");

            // Iterate through each class directory
            for (auto &p: fs::directory_iterator(path)) {
                if (fs::is_directory(p.path())) {
                    // Add class name to label list
                    string u = p.path().filename().string();
                    labels_name.push_back(u);

                    // Process each image in class directory
                    for (auto &img: fs::directory_iterator(p.path())) {
                        // Convert image to tensor using OpenCV
                        torch::Tensor tensor = torch::ext::media::opencv::convertImageToTensor(img.path());

                        // Apply transforms if specified
                        if (!transforms.empty()) {
                            // tensor = this->compose(tensor);
                        }

                        // Store tensor and corresponding label
                        data.push_back(tensor);
                        labels.push_back(labels_name.size() - 1);  // Use label index
                    }
                }
            }
        } else {
            // Set path to validation data
            fs::path path = this->dataset_path / folder_name / fs::path("val");

            // Iterate through each class directory
            for (auto &p: fs::directory_iterator(path)) {
                if (fs::is_directory(p.path())) {
                    // Add class name to label list
                    string u = p.path().filename().string();
                    labels_name.push_back(u);

                    // Process each image in class directory
                    for (auto &img: fs::directory_iterator(p.path())) {
                        // Convert image to tensor using OpenCV
                        torch::Tensor tensor = torch::ext::media::opencv::convertImageToTensor(img.path());

                        // Apply transforms if specified
                        if (!transforms.empty()) {
                            // tensor = this->compose(tensor);
                        }

                        // Store tensor and corresponding label
                        data.push_back(tensor);
                        labels.push_back(labels_name.size() - 1);  // Use label index
                    }
                }
            }
        }
    }
} // namespace xt::data::datasets

