#pragma once


#include "../common.h"


namespace xt::datasets
{
    class TextFolder : public xt::datasets::Dataset
    {
    public :
        TextFolder(const std::string& file_path);
        TextFolder(const std::string& file_path, DataMode mode);
        TextFolder(const std::string& file_path, DataMode mode, std::unique_ptr<xt::Module> transformer);

    private:
        vector<string> labels_name;
        bool load_sub_folders = false;
        void load_data();
    };
}
