#include "../../include/datasets/places.h"

namespace xt::data::datasets {

    Places365::Places365(const std::string &root): Places365::Places365(root, DataMode::TRAIN, false) {
    }

    Places365::Places365(const std::string &root, DataMode mode): Places365::Places365(root, mode, false) {
    }

    Places365::Places365(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("Places365: Places365 not implemented");
    }


    Places365::Places365(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("Places365: Places365 not implemented");
    }



}
