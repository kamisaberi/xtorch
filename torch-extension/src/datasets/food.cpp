#include "../../include/datasets/food.h"

namespace torch::ext::data::datasets {
    Food101::Food101(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {



    }

    void Food101::check_resources() {

        fs::path archive_file_abs_path =this->root / this->dataset_file_name;
        this->dataset_path = this->root / this->dataset_folder_name;
        if (! fs::exists(archive_file_abs_path)) {
            //TODO Check folder and files inside it and after that go to download file
        }








    }
    // Food101::Food101() {
    //     throw NotImplementedException();
    // }
}
