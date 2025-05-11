#include "../../../../include/datasets/natural_language_processing/machine_translation/multi30k.h"

namespace xt::data::datasets {

    MULTI::MULTI(const std::string &root): MULTI::MULTI(root, DataMode::TRAIN, false) {
    }

    MULTI::MULTI(const std::string &root, DataMode mode): MULTI::MULTI(root, mode, false) {
    }

    MULTI::MULTI(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("MULTI: MULTI not implemented");
    }


    MULTI::MULTI(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("MULTI: MULTI not implemented");
    }


}
