

# File video-dataset.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**general**](dir_3e490c73b2bbc01f3b90ef3b6e284c64.md) **>** [**video-dataset.h**](video-dataset_8h.md)

[Go to the documentation of this file](video-dataset_8h.md)


```C++
#pragma once


#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {



    class VideoDataset : public BaseDataset {
    public :
        VideoDataset(const std::string &file_path);
        VideoDataset(const std::string &file_path,DataMode mode);
        VideoDataset(const std::string &file_path,DataMode mode , vector<std::function<torch::Tensor(torch::Tensor)>> transforms);

    private:
        vector<string> labels_name;
        bool load_sub_folders = false;
        void load_data();
    };


    class StackedVideoDataset : public BaseDataset {
    public :
        StackedVideoDataset(const std::string &folder_path);
        StackedVideoDataset(const std::string &folder_path,DataMode mode);
        StackedVideoDataset(const std::string &folder_path,DataMode mode, bool load_sub_folders);
        StackedVideoDataset(const std::string &folder_path,DataMode mode, bool load_sub_folders , vector<std::function<torch::Tensor(torch::Tensor)>> transforms);

    private:
        vector<string> labels_name;
        bool load_sub_folders = false;
        void load_data();
    };



}
```


