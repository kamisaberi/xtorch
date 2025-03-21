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

        // bool res = true;
        // for (const auto &resource: this->resources) {
        //     fs::path pth = std::get<0>(resource);
        //     std::string md = std::get<1>(resource);
        //     fs::path fpth = this->dataset_path / pth;
        //     if (!(fs::exists(fpth) && torch::ext::utils::get_md5_checksum(fpth.string()) == md)) {
        //         if (download) {
        //             string u = (this->url / pth).string();
        //             auto [r, path] = torch::ext::utils::download(u, this->dataset_path.string());
        //         } else {
        //             throw runtime_error("Resources files dent exist. please try again with download = true");
        //         }
        //     }
        //     torch::ext::utils::extractGzip(fpth);
        // }
    }


    void Imagenette::load_data(DataMode mode) {
        auto [url , dataset_filename , folder_name, md] = this->resources[getImageTypeValue(this->type)];
        if (mode == DataMode::TRAIN) {
            fs::path path = this->dataset_path / folder_name / fs::path("train");
            for (auto &p: fs::directory_iterator(path)) {
                if (fs::is_directory(p.path())) {
                    string u = p.path().string();
                    cout << u << endl;
                }
            }

            //     fs::path imgs = this->dataset_path / std::get<0>(files["train"]);
            //     fs::path lbls = this->dataset_path / std::get<1>(files["train"]);
            //     cout << imgs.string() << "  " << lbls.string() << endl;
            //     cout << imgs << endl;
            //     this->read_images(imgs.string(), 50000);
            //     this->read_labels(lbls.string(), 50000);
            //     // auto images = read_mnist_images(imgs.string(), 50000);
            //     // auto labels = read_mnist_labels(lbls.string(), 50000);
            //     // cout << labels[0] << endl;
            //     // this->data = images;
            //     // this->labels = labels;
        } else {
            fs::path path = this->dataset_path / folder_name / fs::path("val");
            //     fs::path imgs = this->dataset_path / std::get<0>(files["test"]);
            //     fs::path lbls = this->dataset_path / std::get<1>(files["test"]);
            //     cout << imgs << endl;
            //     this->read_images(imgs.string(), 10000);
            //     this->read_labels(imgs.string(), 10000);
            //     // auto images = read_mnist_images(imgs.string(), 10000);
            //     // auto labels = read_mnist_labels(lbls.string(), 10000);
            //     // this->data = images;
            //     // this->labels = labels;
        }
    }
}
