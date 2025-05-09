#pragma once
#include "../../base/base.h"
#include "../../../headers/datasets.h"


namespace xt::data::datasets {
    class CocoCaptions : public BaseDataset {
    public :
        explicit  CocoCaptions(const std::string &root);
        CocoCaptions(const std::string &root, DataMode mode);
        CocoCaptions(const std::string &root, DataMode mode , bool download);
        CocoCaptions(const std::string &root, DataMode mode , bool download, TransformType transforms);

    private :
        void load_data();

        void check_resources();
    };
}
