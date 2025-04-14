#include "../../../include/datasets/image-classification/fgvc-aircraft.h"

namespace xt::data::datasets {

    FGVCAircraft::FGVCAircraft(const std::string &root): FGVCAircraft::FGVCAircraft(root, DataMode::TRAIN, false) {
    }

    FGVCAircraft::FGVCAircraft(const std::string &root, DataMode mode): FGVCAircraft::FGVCAircraft(root, mode, false) {
    }

    FGVCAircraft::FGVCAircraft(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("FGVCAircraft: FGVCAircraft not implemented");
    }


    FGVCAircraft::FGVCAircraft(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("FGVCAircraft: FGVCAircraft not implemented");
    }


}
