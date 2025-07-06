#pragma once


#include "../common.h"


namespace xt::datasets
{
    class VideoFolder : public xt::datasets::Dataset
    {
    public :
        VideoFolder(const std::string& folder_path);
        VideoFolder(const std::string& folder_path, DataMode mode);
        VideoFolder(const std::string& folder_path, DataMode mode, bool load_sub_folders);
        VideoFolder(const std::string& folder_path, DataMode mode, bool load_sub_folders,
                    std::unique_ptr<xt::Module> transformer);

    private:
        vector<string> labels_name;
        bool load_sub_folders = false;
        void load_data();
    };
}
