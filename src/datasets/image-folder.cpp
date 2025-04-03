#include "../../include/datasets/image-folder.h"

namespace xt::data::datasets {
    ImageFolder::ImageFolder(const std::string &root): ImageFolder::ImageFolder(
        root, false, DataMode::TRAIN, LabelsType::BY_FOLDER) {
    }

    ImageFolder::ImageFolder(const std::string &root, bool load_sub_folders): ImageFolder::ImageFolder(
        root, load_sub_folders, DataMode::TRAIN, LabelsType::BY_FOLDER) {
    }

    ImageFolder::ImageFolder(const std::string &root,bool load_sub_folders, DataMode mode): ImageFolder::ImageFolder(
        root, load_sub_folders, mode, LabelsType::BY_FOLDER) {
    }

    ImageFolder::ImageFolder(const std::string &root, bool load_sub_folders, DataMode mode, LabelsType label_type)
        : BaseDataset(root, mode, false) {
        this->label_type = label_type;
        this->load_sub_folders = load_sub_folders;
        // check_resources(root, download);
        load_data();
    }

    ImageFolder::ImageFolder(const std::string &root, bool load_sub_folders, DataMode mode, LabelsType label_type,
                             vector<std::function<torch::Tensor(torch::Tensor)> > transforms): BaseDataset(
        root, mode, false) {
        this->label_type = label_type;
        this->load_sub_folders = load_sub_folders;
        if (!transforms.empty()) {
            this->transforms = transforms;
            this->compose = xt::data::transforms::Compose(transforms);
        }
        // check_resources(root, download);
        load_data();
    }

    void ImageFolder::load_data() {
        if (!fs::exists(this->root)) {
            throw runtime_error("path is not exists");
        }
        this->dataset_path = this->root;
        for (auto &entry: fs::directory_iterator(this->dataset_path)) {
            if (entry.is_directory()) {
                string path = entry.path().string();
                string cat = entry.path().filename().string();
                if (label_type == LabelsType::BY_FOLDER) {
                    labels_name.push_back(cat);
                }


                if (this->load_sub_folders == false) {
                    for (auto &file: fs::directory_iterator(entry.path())) {
                        if (!file.is_directory()) {
                            torch::Tensor tensor = torch::ext::media::opencv::convertImageToTensor(file.path());
                            data.push_back(tensor);
                            labels.push_back(labels_name.size() - 1);
                        }
                    }
                } else {
                    for (auto &file: fs::recursive_directory_iterator(entry.path())) {
                        if (!file.is_directory()) {
                            torch::Tensor tensor = torch::ext::media::opencv::convertImageToTensor(file.path());
                            data.push_back(tensor);
                            labels.push_back(labels_name.size() - 1);
                        }
                    }
                }
            }
        }
    }
}
