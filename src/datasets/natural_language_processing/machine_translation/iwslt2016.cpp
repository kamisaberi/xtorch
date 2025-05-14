#include "datasets/natural_language_processing/machine_translation/iwslt2016.h"

namespace xt::data::datasets {

    IWSLT20169::IWSLT20169(const std::string &root): IWSLT20169::IWSLT20169(root, DataMode::TRAIN, false) {
    }

    IWSLT20169::IWSLT20169(const std::string &root, DataMode mode): IWSLT20169::IWSLT20169(root, mode, false) {
    }

    IWSLT20169::IWSLT20169(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("IWSLT: IWSLT not implemented");
    }


    IWSLT20169::IWSLT20169(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("IWSLT: IWSLT not implemented");
    }


}
