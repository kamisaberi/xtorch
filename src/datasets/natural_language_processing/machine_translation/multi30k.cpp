#include "datasets/natural_language_processing/machine_translation/multi30k.h"

namespace xt::data::datasets {

    MULTI30k::MULTI30k(const std::string &root): MULTI30k::MULTI30k(root, DataMode::TRAIN, false) {
    }

    MULTI30k::MULTI30k(const std::string &root, DataMode mode): MULTI30k::MULTI30k(root, mode, false) {
    }

    MULTI30k::MULTI30k(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("MULTI: MULTI not implemented");
    }


    MULTI30k::MULTI30k(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("MULTI: MULTI not implemented");
    }


}
