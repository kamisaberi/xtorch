#include "../../include/datasets/mnist.h"

namespace torch::ext::data::datasets {
    std::vector<torch::Tensor> read_mnist_images(const std::string &file_path, int num_images) {
        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + file_path);
        }

        // Read metadata
        int32_t magic_number, num_items, rows, cols;
        file.read(reinterpret_cast<char *>(&magic_number), 4);
        file.read(reinterpret_cast<char *>(&num_items), 4);
        file.read(reinterpret_cast<char *>(&rows), 4);
        file.read(reinterpret_cast<char *>(&cols), 4);

        // Convert endianess
        magic_number = __builtin_bswap32(magic_number);
        num_items = __builtin_bswap32(num_items);
        rows = __builtin_bswap32(rows);
        cols = __builtin_bswap32(cols);

        std::vector<torch::Tensor> fimages;
        std::vector<std::vector<uint8_t> > images(num_images, std::vector<uint8_t>(rows * cols));
        for (int i = 0; i < num_images; i++) {
            file.read(reinterpret_cast<char *>(images[i].data()), rows * cols);
            auto tensor_image = torch::from_blob(images[i].data(), {1, 28, 28},
                                                 torch::kByte).clone();
            fimages.push_back(tensor_image);
        }
        //        cout << fimages[0] << endl;


        file.close();
        return fimages;
    }

    std::vector<uint8_t> read_mnist_labels(const std::string &file_path, int num_labels) {
        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + file_path);
        }

        // Read metadata
        int32_t magic_number, num_items;
        file.read(reinterpret_cast<char *>(&magic_number), 4);
        file.read(reinterpret_cast<char *>(&num_items), 4);

        // Convert endianess
        cout << magic_number << "\t";
        magic_number = __builtin_bswap32(magic_number);
        num_items = __builtin_bswap32(num_items);

        std::vector<uint8_t> labels(num_labels);
        file.read(reinterpret_cast<char *>(labels.data()), num_labels);

        cout << labels.data() << endl;
        file.close();
        return labels;
    }



    //------------------ MNIST ------------------//
    MNIST::MNIST(const std::string &root, DataMode mode, bool download) {
        check_resources(root, download);
        load_data(mode);
    }

    MNIST::MNIST(const fs::path &root, DatasetArguments args) {
        auto [mode , download , transforms] = args;
        check_resources(root, download);
        load_data(mode);


    }

    torch::data::Example<> MNIST::get(size_t index) {
        return {data[index].clone(), torch::tensor(labels[index])}; // Clone to ensure tensor validity
    }

    torch::optional<size_t> MNIST::size() const {
        return data.size();
    }

    void MNIST::check_resources(const std::string &root, bool download) {
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
            if (!(fs::exists(fpth) && md5File(fpth.string()) == md)) {
                if (download) {
                    string u = (this->url / pth).string();
                    auto [r, path] = download_data(u, this->dataset_path.string());
                } else {
                    throw runtime_error("Resources files dent exist. please try again with download = true");
                }
            }
            extractGzip(fpth);
        }
    }


    void MNIST::load_data(DataMode mode) {
        if (mode == DataMode::TRAIN) {
            fs::path imgs = this->dataset_path / std::get<0>(files["train"]);
            fs::path lbls = this->dataset_path / std::get<1>(files["train"]);
            cout << imgs << endl;
            auto images = read_mnist_images(imgs.string(), 50000);
            auto labels = read_mnist_labels(lbls.string(), 50000);
            this->data = images;
            this->labels = labels;
        } else {
            fs::path imgs = this->dataset_path / std::get<0>(files["test"]);
            fs::path lbls = this->dataset_path / std::get<1>(files["test"]);
            cout << imgs << endl;
            auto images = read_mnist_images(imgs.string(), 10000);
            auto labels = read_mnist_labels(lbls.string(), 10000);
            this->data = images;
            this->labels = labels;
        }
    }


    FashionMNIST::FashionMNIST(const std::string &root, bool train, bool download) {
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
            if (!(fs::exists(fpth) && md5File(fpth.string()) == md)) {
                if (download) {
                    string u = (this->url / pth).string();
                    auto [r, path] = download_data(u, this->dataset_path.string());
                } else {
                    throw runtime_error("Resources files dent exist. please try again with download = true");
                }
            }
            extractGzip(fpth);
        }
        load_data(train);
    }

    // Override `get` method to return a single data sample
    torch::data::Example<> FashionMNIST::get(size_t index) {
        return {data[index], data[index]};
    }

    // Override `size` method to return the number of samples
    torch::optional<size_t> FashionMNIST::size() const {
        return data.size();
    }

    void FashionMNIST::load_data(bool train) {
        if (train) {
            fs::path imgs = this->dataset_path / std::get<0>(files["train"]);
            fs::path lbls = this->dataset_path / std::get<1>(files["train"]);
            cout << imgs << endl;
            auto images = read_mnist_images(imgs.string(), 50000);
            auto labels = read_mnist_labels(lbls.string(), 50000);
            cout << images.size() << endl;
            cout << labels.size() << endl;
            this->data = images;
            this->labels = labels;
            //             for (int i = 0; i < 100; i++) {
            // //                for (auto row : images[i]) {
            // //                    cout << (unsigned int) row << " -- ";
            // //                }
            //                 cout << (unsigned int) labels[i] << "\t";
            //             }
            //             cout << endl;
        } else {
            fs::path imgs = this->dataset_path / std::get<0>(files["test"]);
            fs::path lbls = this->dataset_path / std::get<1>(files["test"]);
            cout << imgs << endl;
            auto images = read_mnist_images(imgs.string(), 10000);
            auto labels = read_mnist_labels(lbls.string(), 10000);
            cout << images.size() << endl;
            cout << labels.size() << endl;
            this->data = images;
            this->labels = labels;
        }
    }


    KMNIST::KMNIST(const std::string &root, bool train, bool download) {
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
            if (!(fs::exists(fpth) && md5File(fpth.string()) == md)) {
                if (download) {
                    string u = (this->url / pth).string();
                    auto [r, path] = download_data(u, this->dataset_path.string());
                } else {
                    throw runtime_error("Resources files dent exist. please try again with download = true");
                }
            }
            extractGzip(fpth);
        }
        load_data(train);
    }

    // Override `get` method to return a single data sample
    torch::data::Example<> KMNIST::get(size_t index) {
        return {data[index], data[index]};
    }

    // Override `size` method to return the number of samples
    torch::optional<size_t> KMNIST::size() const {
        return labels.size();
    }

    void KMNIST::load_data(bool train) {
        if (train) {
            fs::path imgs = this->dataset_path / std::get<0>(files["train"]);
            fs::path lbls = this->dataset_path / std::get<1>(files["train"]);
            cout << imgs << endl;
            auto images = read_mnist_images(imgs.string(), 50000);
            auto labels = read_mnist_labels(lbls.string(), 50000);
            cout << images.size() << endl;
            cout << labels.size() << endl;
            this->data = images;
            this->labels = labels;
            //             for (int i = 0; i < 100; i++) {
            // //                for (auto row : images[i]) {
            // //                    cout << (unsigned int) row << " -- ";
            // //                }
            //                 cout << (unsigned int) labels[i] << "\t";
            //             }
            //             cout << endl;
        } else {
            fs::path imgs = this->dataset_path / std::get<0>(files["test"]);
            fs::path lbls = this->dataset_path / std::get<1>(files["test"]);
            cout << imgs << endl;
            auto images = read_mnist_images(imgs.string(), 10000);
            auto labels = read_mnist_labels(lbls.string(), 10000);
            cout << images.size() << endl;
            cout << labels.size() << endl;
            this->data = images;
            this->labels = labels;
        }
    }


    EMNIST::EMNIST(const std::string &root, bool train, bool download) {
        this->root = fs::path(root);
        if (!fs::exists(this->root)) {
            throw runtime_error("path is not exists");
        }
        this->dataset_path = this->root / this->dataset_folder_name;
        if (!fs::exists(this->dataset_path)) {
            fs::create_directories(this->dataset_path);
        }

        bool res = true;
        fs::path fpth = this->dataset_path / archive_file_name;
        if (!(fs::exists(fpth) && md5File(fpth.string()) == archive_file_md5)) {
            if (download) {
                string u = (this->url / archive_file_name).string();
                auto [r, path] = download_data(u, this->dataset_path.string());
            } else {
                throw runtime_error("Resources files dent exist. please try again with download = true");
            }
        }
        extractZip(fpth);
        load_data(train);
    }

    void EMNIST::load_data(bool train) {
        // if (train) {
        //     fs::path imgs = this->dataset_path / std::get<0>(files["train"]);
        //     fs::path lbls = this->dataset_path / std::get<1>(files["train"]);
        //     cout << imgs << endl;
        //     auto images = read_mnist_images(imgs.string(), 50000);
        //     auto labels = read_mnist_labels(lbls.string(), 50000);
        //     cout << images.size() << endl;
        //     cout << labels.size() << endl;
        //     this->data = images;
        //     this->labels = labels;
        // } else {
        //     fs::path imgs = this->dataset_path / std::get<0>(files["test"]);
        //     fs::path lbls = this->dataset_path / std::get<1>(files["test"]);
        //     cout << imgs << endl;
        //     auto images = read_mnist_images(imgs.string(), 10000);
        //     auto labels = read_mnist_labels(lbls.string(), 10000);
        //     cout << images.size() << endl;
        //     cout << labels.size() << endl;
        //     this->data = images;
        //     this->labels = labels;
        // }
    }


    // Override `get` method to return a single data sample
    torch::data::Example<> EMNIST::get(size_t index) {
        return {data[index], data[index]};
    }

    // Override `size` method to return the number of samples
    torch::optional<size_t> EMNIST::size() const {
        return labels.size();
    }


    QMNIST::QMNIST(const std::string &root, bool train, bool download) {
    }

    // Override `get` method to return a single data sample
    torch::data::Example<> QMNIST::get(size_t index) {
        return {data[index], data[index]};
    }

    // Override `size` method to return the number of samples
    torch::optional<size_t> QMNIST::size() const {
        return labels.size();
    }

    void QMNIST::load_data(bool train) {
        if (train) {
            fs::path imgs = this->dataset_path / std::get<0>(resources["train"][0]);
            fs::path lbls = this->dataset_path / std::get<0>(resources["train"][1]);
            cout << imgs << endl;
            auto images = read_mnist_images(imgs.string(), 50000);
            auto labels = read_mnist_labels(lbls.string(), 50000);
            cout << images.size() << endl;
            cout << labels.size() << endl;
            this->data = images;
            this->labels = labels;
        } else {
            fs::path imgs = this->dataset_path / std::get<0>(resources["test"][0]);
            fs::path lbls = this->dataset_path / std::get<0>(resources["test"][1]);
            cout << imgs << endl;
            auto images = read_mnist_images(imgs.string(), 10000);
            auto labels = read_mnist_labels(lbls.string(), 10000);
            cout << images.size() << endl;
            cout << labels.size() << endl;
            this->data = images;
            this->labels = labels;
        }
    }
}
