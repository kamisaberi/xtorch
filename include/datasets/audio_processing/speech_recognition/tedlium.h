#pragma once

#include "datasets/base/base.h"
#include "datasets/common.h"


namespace xt::data::datasets {
    class Tedlium : BaseDataset {
    public :
        explicit Tedlium(const std::string &root);

        Tedlium(const std::string &root, DataMode mode);

        Tedlium(const std::string &root, DataMode mode, bool download);

        Tedlium(const std::string &root, DataMode mode, bool download, TransformType transforms);

    private :
        void load_data();

        void check_resources();
    };
}
