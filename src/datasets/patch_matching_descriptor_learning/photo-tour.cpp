#include "../../../include/datasets/patch_matching_descriptor_learning/photo_tour.h"

namespace xt::data::datasets {

    PhotoTour::PhotoTour(const std::string &root): PhotoTour::PhotoTour(root, DataMode::TRAIN, false) {
    }

    PhotoTour::PhotoTour(const std::string &root, DataMode mode): PhotoTour::PhotoTour(root, mode, false) {
    }

    PhotoTour::PhotoTour(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("PhotoTour: PhotoTour not implemented");
    }


    PhotoTour::PhotoTour(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("PhotoTour: PhotoTour not implemented");
    }

}
