#include "datasets/natural_language_processing/machine_translation/wmt14.h"

namespace xt::data::datasets {

    WMT14::WMT14(const std::string &root): WMT14::WMT14(root, DataMode::TRAIN, false) {
    }

    WMT14::WMT14(const std::string &root, DataMode mode): WMT14::WMT14(root, mode, false) {
    }

    WMT14::WMT14(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("WMT: WMT not implemented");
    }


    WMT14::WMT14(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("WMT: WMT not implemented");
    }


}
