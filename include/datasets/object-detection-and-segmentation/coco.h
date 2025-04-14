#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


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
