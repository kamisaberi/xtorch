#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class CMUArctic : BaseDataset {
    public :
        explicit CMUArctic(const std::string &root);
        CMUArctic(const std::string &root, DataMode mode);
        CMUArctic(const std::string &root, DataMode mode , bool download);
        CMUArctic(const std::string &root, DataMode mode , bool download, TransformType transforms);

    private :
        void load_data();

        void check_resources();
    };
}
