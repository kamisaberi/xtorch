#include "../../include/datasets/imagenette.h"

namespace torch::ext::data::datasets {
    Imagenette::Imagenette(const std::string &root, DataMode mode, bool download, ImageType type) : BaseDataset(root, mode, download) {
        check_resources(root, download);
        load_data(mode);


    }

    void Imagenette::load_data(DataMode mode) {

    }

    void Imagenette::check_resources(const std::string &root, bool download) {

    }
}
