#include "datasets/natural_language_processing/machine_translation/iwslt2017.h"

namespace xt::data::datasets {

    IWSLT2017::IWSLT2017(const std::string &root): IWSLT2017::IWSLT2017(root, DataMode::TRAIN, false) {
    }

    IWSLT2017::IWSLT2017(const std::string &root, DataMode mode): IWSLT2017::IWSLT2017(root, mode, false) {
    }

    IWSLT2017::IWSLT2017(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("IWSLT: IWSLT not implemented");
    }


    IWSLT2017::IWSLT2017(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("IWSLT: IWSLT not implemented");
    }


}
