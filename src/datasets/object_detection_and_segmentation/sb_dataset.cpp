#include "../../../include/datasets/object_detection_and_segmentation/sb_dataset.h"

namespace xt::data::datasets {

    SBDataset::SBDataset(const std::string &root): SBDataset::SBDataset(root, DataMode::TRAIN, false) {
    }

    SBDataset::SBDataset(const std::string &root, DataMode mode): SBDataset::SBDataset(root, mode, false) {
    }

    SBDataset::SBDataset(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("SBDataset: SBDataset not implemented");
    }


    SBDataset::SBDataset(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("SBDataset: SBDataset not implemented");
    }


}
