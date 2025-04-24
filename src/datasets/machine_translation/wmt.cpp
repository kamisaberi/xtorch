#include "../../../include/datasets/machine-translation/wmt.h"

namespace xt::data::datasets {

    WMT::WMT(const std::string &root): WMT::WMT(root, DataMode::TRAIN, false) {
    }

    WMT::WMT(const std::string &root, DataMode mode): WMT::WMT(root, mode, false) {
    }

    WMT::WMT(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("WMT: WMT not implemented");
    }


    WMT::WMT(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("WMT: WMT not implemented");
    }


}
