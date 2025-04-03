#include "../../../include/datasets/specific/falling-things-stereo.h"

namespace xt::data::datasets {

    FallingThingsStereo::FallingThingsStereo(const std::string &root): FallingThingsStereo::FallingThingsStereo(root, DataMode::TRAIN, false) {
    }

    FallingThingsStereo::FallingThingsStereo(const std::string &root, DataMode mode): FallingThingsStereo::FallingThingsStereo(root, mode, false) {
    }

    FallingThingsStereo::FallingThingsStereo(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("FallingThingsStereo: FallingThingsStereo not implemented");
    }


    FallingThingsStereo::FallingThingsStereo(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("FallingThingsStereo: FallingThingsStereo not implemented");
    }

}
