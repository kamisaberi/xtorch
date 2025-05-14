#pragma once

#include "datasets/base/base.h"
#include "datasets/common.h"


namespace xt::data::datasets {
    class TIMIT : BaseDataset {
    public :
        explicit TIMIT(const std::string &root);

        TIMIT(const std::string &root, DataMode mode);

        TIMIT(const std::string &root, DataMode mode, bool download);

        TIMIT(const std::string &root, DataMode mode, bool download, TransformType transforms);

    private :
        void load_data();

        void check_resources();
    };
}
