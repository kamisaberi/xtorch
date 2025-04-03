#include "../../../include/datasets/specific/country.h"

namespace xt::data::datasets {

    Country211::Country211(const std::string &root): Country211::Country211(root, DataMode::TRAIN, false) {
    }

    Country211::Country211(const std::string &root, DataMode mode): Country211::Country211(root, mode, false) {
    }

    Country211::Country211(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("Country211: Country211 not implemented");
    }


    Country211::Country211(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("Country211: Country211 not implemented");
    }

}
