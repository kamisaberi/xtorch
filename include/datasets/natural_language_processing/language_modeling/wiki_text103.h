#pragma once

#include "datasets/base/base.h"

namespace xt::data::datasets
{
    class WikiText103 : BaseDataset
    {
    public :
        explicit WikiText103(const std::string& root);
        WikiText103(const std::string& root, DataMode mode);
        WikiText103(const std::string& root, DataMode mode, bool download);
        WikiText103(const std::string& root, DataMode mode, bool download, TransformType transforms);

    private :
        void load_data();

        void check_resources();
    };
}
