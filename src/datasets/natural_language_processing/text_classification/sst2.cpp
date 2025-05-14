#include "datasets/natural_language_processing/text_classification/sst2.h"

namespace xt::data::datasets {

    SST2::SST2(const std::string &root): SST2::SST2(root, DataMode::TRAIN, false) {
    }

    SST2::SST2(const std::string &root, DataMode mode): SST2::SST2(root, mode, false) {
    }

    SST2::SST2(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("SST: SST not implemented");
    }


    SST2::SST2(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("SST: SST not implemented");
    }


}
