#include "datasets/computer_vision/image_classification/fashion_mnist.h"

namespace xt::data::datasets {



    FashionMNIST::FashionMNIST(const std::string& root): FashionMNIST(
    root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    FashionMNIST::FashionMNIST(const std::string& root, xt::datasets::DataMode mode): FashionMNIST(
        root, mode, false, nullptr, nullptr)
    {
    }

    FashionMNIST::FashionMNIST(const std::string& root, xt::datasets::DataMode mode, bool download) :
        FashionMNIST(
            root, mode, download, nullptr, nullptr)
    {
    }

    FashionMNIST::FashionMNIST(const std::string& root, xt::datasets::DataMode mode, bool download,
                 std::unique_ptr<xt::Module> transformer) : FashionMNIST(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    FashionMNIST::FashionMNIST(const std::string& root, xt::datasets::DataMode mode, bool download,
                 std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();
    }




    void FashionMNIST::load_data() {
        if (mode == xt::datasets::DataMode::TRAIN) {
            fs::path imgs = this->dataset_path / std::get<0>(files["train"]);
            fs::path lbls = this->dataset_path / std::get<1>(files["train"]);
            cout << imgs.string() << "  " << lbls.string() << endl;
            cout << imgs << endl;
            this->read_images(imgs.string(), 60000);
            this->read_labels(lbls.string(), 60000);
            //
            // auto images = read_mnist_images(imgs.string(), 50000);
            // auto labels = read_mnist_labels(lbls.string(), 50000);
            // cout << labels[0] << endl;
            // this->data = images;
            // this->labels = labels;
        } else {
            fs::path imgs = this->dataset_path / std::get<0>(files["test"]);
            fs::path lbls = this->dataset_path / std::get<1>(files["test"]);
            cout << imgs << endl;
            this->read_images(imgs.string(), 10000);
            this->read_labels(lbls.string(), 10000);
            // auto images = read_mnist_images(imgs.string(), 10000);
            // auto labels = read_mnist_labels(lbls.string(), 10000);
            // this->data = images;
            // this->labels = labels;
        }
    }


    void FashionMNIST::check_resources() {
        this->root = fs::path(root);
        if (!fs::exists(this->root)) {
            throw runtime_error("path is not exists");
        }
        this->dataset_path = this->root / this->dataset_folder_name;
        if (!fs::exists(this->dataset_path)) {
            fs::create_directories(this->dataset_path);
        }

        bool res = true;
        for (const auto &resource: this->resources) {
            fs::path pth = std::get<0>(resource);
            std::string md = std::get<1>(resource);
            fs::path fpth = this->dataset_path / pth;
            if (!(fs::exists(fpth) && xt::utils::get_md5_checksum(fpth.string()) == md)) {
                if (download) {
                    string u = (this->url / pth).string();
                    auto [r, path] = xt::utils::download(u, this->dataset_path.string());
                } else {
                    throw runtime_error("Resources files dent exist. please try again with download = true");
                }
            }
            xt::utils::extractGzip(fpth);
        }
    }

    // KMNIST::KMNIST(const std::string &root, DataMode mode, bool download) : MNISTBase(root, mode, download) {
    //     check_resources(root, download);
    //     load_data(mode);
    // }


}
