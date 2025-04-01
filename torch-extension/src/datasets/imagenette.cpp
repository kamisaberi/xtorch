#include "../../include/datasets/imagenette.h"

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
        check_resources(root, download);
        load_data(mode);
    }

    Imagenette::Imagenette(const std::string &root, DataMode mode, bool download, ImageType type,
                           vector<std::function<torch::Tensor(torch::Tensor)> > transforms): BaseDataset(
        root, mode, download) , type(type) {
        if (!transforms.empty()) {
            this->transforms = transforms;
            this->compose = xt::data::transforms::Compose(transforms);
        }
        check_resources(root, download);
        load_data(mode);
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
        if (!(fs::exists(fpth) && xt::utils::get_md5_checksum(fpth.string()) == md)) {
            if (download) {
                string u = url.string();
                auto [r, path] = xt::utils::download(u, this->dataset_path.string());
            } else {
                throw runtime_error("Resources files dent exist. please try again with download = true");
            }
        }

        xt::utils::extractTgz(fpth, this->dataset_path.string());
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
                        if (!transforms.empty()) {
                            tensor = this->compose(tensor);
                        }
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
                        if (!transforms.empty()) {
                            tensor = this->compose(tensor);
                        }
                        data.push_back(tensor);
                        labels.push_back(labels_name.size() - 1);
                    }
                }
            }
        }
    }
}
