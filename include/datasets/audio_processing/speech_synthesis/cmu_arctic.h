#pragma once
#include "datasets/base/base.h"
#include "datasets/common.h"


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
