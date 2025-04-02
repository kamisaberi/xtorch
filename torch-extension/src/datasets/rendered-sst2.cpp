#include "../../include/datasets/rendered-sst2.h"

namespace xt::data::datasets {

    RenderedSST2::RenderedSST2(const std::string &root): RenderedSST2::RenderedSST2(root, DataMode::TRAIN, false) {
    }

    RenderedSST2::RenderedSST2(const std::string &root, DataMode mode): RenderedSST2::RenderedSST2(root, mode, false) {
    }

    RenderedSST2::RenderedSST2(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("RenderedSST2: RenderedSST2 not implemented");
    }


    RenderedSST2::RenderedSST2(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("RenderedSST2: RenderedSST2 not implemented");
    }


}
