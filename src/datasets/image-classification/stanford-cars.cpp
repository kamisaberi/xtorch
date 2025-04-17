#include "../../../include/datasets/image-classification/stanford-cars.h"

namespace xt::data::datasets {

    StanfordCars::StanfordCars(const std::string &root): StanfordCars::StanfordCars(root, DataMode::TRAIN, false) {
    }

    StanfordCars::StanfordCars(const std::string &root, DataMode mode): StanfordCars::StanfordCars(root, mode, false) {
    }

    StanfordCars::StanfordCars(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("StanfordCars: StanfordCars not implemented");
    }


    StanfordCars::StanfordCars(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("StanfordCars: StanfordCars not implemented");
    }


}
