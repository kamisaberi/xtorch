#include "../../../include/datasets/image_classification/qmnist.h"

namespace xt::data::datasets
{
    // QMNIST::QMNIST(const std::string &root, DataMode mode, bool download) : MNISTBase(root, mode, download) {
    //     check_resources(root, download);
    //     load_data(mode);
    // }
    //
    //
    // QMNIST::QMNIST(const fs::path &root, DatasetArguments args) : MNISTBase(root, args) {
    //     auto [mode , download , transforms] = args;
    //     // cout << "MNIST SIZE: " << this->data.size() << endl;
    //     check_resources(root, download);
    //     load_data(mode);
    //     // cout << "MNIST SIZE: " << this->data.size() << endl;
    //     if (!transforms.empty()) {
    //         // cout << "Transforms 11111111111111111111" << endl;
    //         // this->transform_data(transforms);
    //     }
    //     // cout << "MNIST SIZE: " << this->data.size() << endl;
    // }


    void QMNIST::load_data(DataMode mode)
    {
        if (mode == DataMode::TRAIN)
        {
            fs::path imgs = this->dataset_path / std::get<0>(resources["train"][0]);
            fs::path lbls = this->dataset_path / std::get<0>(resources["train"][1]);
            cout << imgs << endl;
            this->read_images(imgs.string(), 50000);
            this->read_labels(lbls.string(), 50000);
            //
            // auto images = read_mnist_images(imgs.string(), 50000);
            // auto labels = read_mnist_labels(lbls.string(), 50000);
            // cout << images.size() << endl;
            // cout << labels.size() << endl;
            // this->data = images;
            // this->labels = labels;
        }
        else
        {
            fs::path imgs = this->dataset_path / std::get<0>(resources["test"][0]);
            fs::path lbls = this->dataset_path / std::get<0>(resources["test"][1]);
            cout << imgs << endl;
            this->read_images(imgs.string(), 10000);
            this->read_labels(lbls.string(), 10000);

            // auto images = read_mnist_images(imgs.string(), 10000);
            // auto labels = read_mnist_labels(lbls.string(), 10000);
            // cout << images.size() << endl;
            // cout << labels.size() << endl;
            // this->data = images;
            // this->labels = labels;
        }
    }

    void QMNIST::check_resources(const std::string& root, bool download)
    {
        // this->root = fs::path(root);
        // if (!fs::exists(this->root)) {
        //     throw runtime_error("path is not exists");
        // }
        // this->dataset_path = this->root / this->dataset_folder_name;
        // if (!fs::exists(this->dataset_path)) {
        //     fs::create_directories(this->dataset_path);
        // }
        //
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
}
