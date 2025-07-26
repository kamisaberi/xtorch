#pragma once


#include "../common.h"



namespace xt::datasets {

    class PairedImageDataset : public xt::datasets::Dataset {
    public :
        PairedImageDataset(const std::string &file_path);
        PairedImageDataset(const std::string &file_path,DataMode mode);
        PairedImageDataset(const std::string &file_path,DataMode mode , std::unique_ptr<xt::Module> target_transformer);

    private:
        vector<string> labels_name;
        bool load_sub_folders = false;
        void load_data();
    };





}
