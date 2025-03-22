#include "../../include/datasets/imagenette.h"

namespace torch::ext::data::datasets {
    Imagenette::Imagenette(const std::string &root, DataMode mode, bool download, ImageType type) : BaseDataset(
        root, mode, download) {
        this->type = type;
        check_resources(root, download);
        load_data(mode);
    }

    Imagenette::Imagenette(const fs::path &root, DatasetArguments args) : BaseDataset(root, args) {
        auto [mode , download , transforms] = args;
        check_resources(root, download);
        load_data(mode);
        // if (!transforms.empty()) {
        //     this->transform_data(transforms);
        // }
    }


    void Imagenette::check_resources(const std::string &root, bool download) {
        this->root = fs::path(root);
        if (!fs::exists(this->root)) {
            throw runtime_error("path is not exists");
        }
        this->dataset_path = this->root / this->dataset_folder_name;
        if (!fs::exists(this->dataset_path)) {
            fs::create_directories(this->dataset_path);
        }
        auto [url , dataset_filename , folder_name, md] = this->resources[getImageTypeValue(this->type)];
        fs::path fpth = this->dataset_path / dataset_filename;
        if (!(fs::exists(fpth) && torch::ext::utils::get_md5_checksum(fpth.string()) == md)) {
            if (download) {
                string u = url.string();
                auto [r, path] = torch::ext::utils::download(u, this->dataset_path.string());
            } else {
                throw runtime_error("Resources files dent exist. please try again with download = true");
            }
        }

        torch::ext::utils::extractTgz(fpth, this->dataset_path.string());
    }


    void Imagenette::load_data(DataMode mode) {
        auto [url , dataset_filename , folder_name, md] = this->resources[getImageTypeValue(this->type)];
        if (mode == DataMode::TRAIN) {
            fs::path path = this->dataset_path / folder_name / fs::path("train");
            for (auto &p: fs::directory_iterator(path)) {
                if (fs::is_directory(p.path())) {
                    string u = p.path().filename().string();
                    labels_name.push_back(u);
                    for (auto &img: fs::directory_iterator(p.path())) {

                        torch::Tensor tensor = torch::ext::media::opencv::convertImageToTensor(img.path());

                        // cv::Mat image = cv::imread(img.path().string(), cv::IMREAD_COLOR);
                        // if (image.empty()) {
                        //     throw std::runtime_error("Could not load image at: " + img.path().string());
                        // }
                        // // 2. Convert BGR (OpenCV default) to RGB
                        // cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
                        //
                        // // 3. Convert image data to float and normalize to [0, 1]
                        // // image.convertTo(image, CV_32F, 1.0 / 255.0);
                        // image.convertTo(image, CV_32F);
                        //
                        // // 4. Create a tensor from the image data
                        // // OpenCV uses HWC (Height, Width, Channels) format
                        // torch::Tensor tensor = torch::from_blob(image.data, {image.rows, image.cols, image.channels()},
                        //                                         torch::kFloat32
                        // );
                        //
                        // // 5. Permute to CHW (Channels, Height, Width) format, which is PyTorch's default
                        // tensor = tensor.permute({2, 0, 1});
                        //
                        // // 6. Make sure the tensor is contiguous in memory
                        // tensor = tensor.contiguous();
                        data.push_back(tensor);
                        labels.push_back(labels_name.size() - 1);
                    }
                }
            }
        } else {
            fs::path path = this->dataset_path / folder_name / fs::path("val");

            for (auto &p: fs::directory_iterator(path)) {
                if (fs::is_directory(p.path())) {
                    string u = p.path().filename().string();
                    labels_name.push_back(u);
                    for (auto &img: fs::directory_iterator(p.path())) {

                        torch::Tensor tensor = torch::ext::media::opencv::convertImageToTensor(img.path());

                        // cv::Mat image = cv::imread(img.path().string(), cv::IMREAD_COLOR);
                        // if (image.empty()) {
                        //     throw std::runtime_error("Could not load image at: " + img.path().string());
                        // }
                        // // 2. Convert BGR (OpenCV default) to RGB
                        // cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
                        //
                        // // 3. Convert image data to float and normalize to [0, 1]
                        // // image.convertTo(image, CV_32F, 1.0 / 255.0);
                        // image.convertTo(image, CV_32F);
                        //
                        // // 4. Create a tensor from the image data
                        // // OpenCV uses HWC (Height, Width, Channels) format
                        // torch::Tensor tensor = torch::from_blob(image.data, {image.rows, image.cols, image.channels()},
                        //                                         torch::kFloat32
                        // );
                        //
                        // // 5. Permute to CHW (Channels, Height, Width) format, which is PyTorch's default
                        // tensor = tensor.permute({2, 0, 1});
                        //
                        // // 6. Make sure the tensor is contiguous in memory
                        // tensor = tensor.contiguous();
                        data.push_back(tensor);
                        labels.push_back(labels_name.size() - 1);
                    }
                }
            }
        }
    }
}
