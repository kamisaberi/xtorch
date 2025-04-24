#pragma once

#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class CremaD : BaseDataset {
    public :
        explicit CremaD(const std::string &root);

        CremaD(const std::string &root, DataMode mode);

        CremaD(const std::string &root, DataMode mode, bool download);

        CremaD(const std::string &root, DataMode mode, bool download, TransformType transforms);

    private :
        void load_data();

        void check_resources();
    };
}
