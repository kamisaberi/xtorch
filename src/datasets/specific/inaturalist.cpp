#include "../../../include/datasets/specific/inaturalist.h"

namespace xt::data::datasets {

    INaturalist::INaturalist(const std::string &root): INaturalist::INaturalist(root, DataMode::TRAIN, false) {
    }

    INaturalist::INaturalist(const std::string &root, DataMode mode): INaturalist::INaturalist(root, mode, false) {
    }

    INaturalist::INaturalist(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("INaturalist: INaturalist not implemented");
    }


    INaturalist::INaturalist(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("INaturalist: INaturalist not implemented");
    }

}
