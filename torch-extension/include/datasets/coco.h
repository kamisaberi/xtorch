#pragma once
#include "../headers/datasets.h"
#include "base.h"


namespace xt::data::datasets {
    class CocoDetection : public BaseDataset {
    public :
        CocoDetection(const std::string &root);
        CocoDetection(const std::string &root, DataMode mode);
        CocoDetection(const std::string &root, DataMode mode , bool download);
        CocoDetection(const std::string &root, DataMode mode , bool download, vector<std::function<torch::Tensor(torch::Tensor)>> transforms);


    private :
        void load_data();

        void check_resources();
    };

    class CocoCaptions : public BaseDataset {
    public :
        CocoCaptions(const std::string &root);
        CocoCaptions(const std::string &root, DataMode mode);
        CocoCaptions(const std::string &root, DataMode mode , bool download);
        CocoCaptions(const std::string &root, DataMode mode , bool download, vector<std::function<torch::Tensor(torch::Tensor)>> transforms);

    private :
        void load_data();

        void check_resources();
    };
}
