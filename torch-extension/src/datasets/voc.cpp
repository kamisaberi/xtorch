#include "../../include/datasets/voc.h"

namespace xt::data::datasets {

    // ---------------------- VOCDetection ---------------------- //
    VOCDetection::VOCDetection(const std::string &root): VOCDetection::VOCDetection(root, DataMode::TRAIN, false) {
    }

    VOCDetection::VOCDetection(const std::string &root, DataMode mode): VOCDetection::VOCDetection(root, mode, false) {
    }

    VOCDetection::VOCDetection(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("VOCDetection: VOCDetection not implemented");
    }


    VOCDetection::VOCDetection(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("VOCDetection: VOCDetection not implemented");
    }



    // ---------------------- Caltech101 ---------------------- //
    Caltech101::Caltech101(const std::string &root): Caltech101::Caltech101(root, DataMode::TRAIN, false) {
    }

    Caltech101::Caltech101(const std::string &root, DataMode mode): Caltech101::Caltech101(root, mode, false) {
    }

    Caltech101::Caltech101(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("Caltech101: Caltech101 not implemented");
    }


    Caltech101::Caltech101(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("Caltech101: Caltech101 not implemented");
    }

}
