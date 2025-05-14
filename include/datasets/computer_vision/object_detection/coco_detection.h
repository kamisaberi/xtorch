#pragma once
#include "datasets/base/base.h"


namespace xt::data::datasets {
    class CocoDetection : public BaseDataset {
    public :
        explicit  CocoDetection(const std::string &root);
        CocoDetection(const std::string &root, DataMode mode);
        CocoDetection(const std::string &root, DataMode mode , bool download);
        CocoDetection(const std::string &root, DataMode mode , bool download, TransformType transforms);


    private :
        void load_data();

        void check_resources();
    };


}
