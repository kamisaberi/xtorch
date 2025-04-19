#pragma once

#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class GTZAN : BaseDataset {
    public :
        explicit GTZAN(const std::string &root);

        GTZAN(const std::string &root, DataMode mode);

        GTZAN(const std::string &root, DataMode mode, bool download);

        GTZAN(const std::string &root, DataMode mode, bool download, TransformType transforms);

    private :
        void load_data();

        void check_resources();
    };
}
