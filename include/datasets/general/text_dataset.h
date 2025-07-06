#pragma once


#include "../common.h"


namespace xt::datasets {



    class TextDataset : public xt::datasets::Dataset {
    public :
        TextDataset(const std::string &file_path);
        TextDataset(const std::string &file_path,DataMode mode);
        TextDataset(const std::string &file_path,DataMode mode , std::unique_ptr<xt::Module> transformer);

    private:
        vector<string> labels_name;
        bool load_sub_folders = false;
        void load_data();
    };





}
