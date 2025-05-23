#include "datasets/computer_vision/image_classification/food.h"

namespace xt::data::datasets {
    Food101::Food101(const std::string &root): Food101::Food101(root, DataMode::TRAIN, false) {
    }

    Food101::Food101(const std::string &root, DataMode mode): Food101::Food101(root, mode, false) {
    }

    Food101::Food101(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        check_resources();  // Verify dataset files exist
        load_classes();     // Load class names and mapping
        load_data();        // Load image data and labels
    }

    Food101::Food101(const std::string &root, DataMode mode, bool download,
                     vector<std::function<torch::Tensor(torch::Tensor)> > transforms) : BaseDataset(
        root, mode, download, transforms) {
        this->transforms = transforms;
        if (!transforms.empty()) {
            // this->compose = xt::transforms::Compose(transforms);
        }
        check_resources();
        load_classes();
        load_data();
    }

    void Food101::check_resources() {
        fs::path archive_file_abs_path = this->root / this->dataset_file_name;
        this->dataset_path = this->root / this->dataset_folder_name;
        fs::path images_path = this->dataset_path / fs::path("images");

        // Check if extracted dataset exists
        if (fs::exists(images_path)) {
            size_t cnt = xt::utils::fs::countFiles(images_path, true);
            // Verify complete dataset
            if (cnt != this->images_number) {
                if (fs::exists(archive_file_abs_path)) {
                    xt::utils::extract(archive_file_abs_path, this->root.string());
                } else {
                    // Download if archive missing
                    xt::utils::download(this->url, archive_file_abs_path.string());
                    xt::utils::extract(archive_file_abs_path, this->root.string());
                }
            }
        } else {
            // Handle case where dataset not extracted
            if (fs::exists(archive_file_abs_path)) {
                xt::utils::extract(archive_file_abs_path, this->root.string());
            } else {
                // Full download and extract
                xt::utils::download(this->url, archive_file_abs_path.string());
                xt::utils::extract(archive_file_abs_path, this->root.string());
            }
        }
    }

    void Food101::load_classes() {
        cout << "Loading classes..." << endl;
        fs::path meta_path = this->dataset_path / fs::path("meta");
        fs::path classes_file_path = meta_path / fs::path("classes.txt");

        // Read class names line by line
        ifstream ifs(classes_file_path.string());
        while (!ifs.eof()) {
            std::string line;
            std::getline(ifs, line);
            line = xt::utils::string::trim(line);
            if (line.empty())
                continue;

            // Add to name vector and create mapping
            classes_name.push_back(line);
            classes_map.insert({line, classes_name.size() - 1});
        }
    }

    void Food101::load_data() {
        fs::path images_path = this->dataset_path / fs::path("images");
        fs::path meta_path = this->dataset_path / fs::path("meta");

        // Handle training set
        if (this->mode == DataMode::TRAIN) {
            fs::path train_file_path = meta_path / fs::path("train.txt");
            ifstream ifs(train_file_path.string());

            while (!ifs.eof()) {
                string line;
                getline(ifs, line);
                line = xt::utils::string::trim(line);
                if (line.empty())
                    continue;

                // Parse line format: "class_name/image_name"
                vector<string> tokens = xt::utils::string::split(line, "/");
                string label = tokens[0];

                // Load and process image
                fs::path img_path = images_path / fs::path(line + ".jpg");
                torch::Tensor tensor = torch::ext::media::opencv::convertImageToTensor(img_path);

                // Apply transforms if specified
                if (!transforms.empty()) {
                    // tensor = this->compose(tensor);
                }

                // Store results
                this->data.push_back(tensor);
                this->labels.push_back(classes_map[label]);
            }
        }
        // Handle test set
        else {
            fs::path train_file_path = meta_path / fs::path("test.txt");
            ifstream ifs(train_file_path.string());

            while (!ifs.eof()) {
                string line;
                getline(ifs, line);
                line = xt::utils::string::trim(line);
                if (line.empty())
                    continue;

                // Parse line format: "class_name/image_name"
                vector<string> tokens = xt::utils::string::split(line, "/");
                string label = tokens[0];

                // Load and process image
                fs::path img_path = images_path / fs::path(line + ".jpg");
                torch::Tensor tensor = torch::ext::media::opencv::convertImageToTensor(img_path);

                // Apply transforms if specified
                if (!transforms.empty()) {
                    // tensor = this->compose(tensor);
                }

                // Store results
                this->data.push_back(tensor);
                this->labels.push_back(classes_map[label]);
            }
        }
    }

}