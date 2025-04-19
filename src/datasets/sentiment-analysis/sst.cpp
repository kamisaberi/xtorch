#include "../../../include/datasets/sentiment-analysis/sst.h"

namespace xt::data::datasets {

    SST::SST(const std::string &root): SST::SST(root, DataMode::TRAIN, false) {
    }

    SST::SST(const std::string &root, DataMode mode): SST::SST(root, mode, false) {
    }

    SST::SST(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("SST: SST not implemented");
    }


    SST::SST(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("SST: SST not implemented");
    }


}
