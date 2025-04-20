

# File food.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**image-classification**](dir_9d21d6f83a70094db43fe94b096ae893.md) **>** [**food.h**](food_8h.md)

[Go to the documentation of this file](food_8h.md)


```C++
#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"
#include  "../../utils/filesystem.h"
#include  "../../utils/string.h"
#include <fstream>
#include <map>


using namespace std;
namespace fs = std::filesystem;

namespace xt::data::datasets {
    class Food101 : public BaseDataset {
    public :

        explicit Food101(const std::string &root);
        Food101(const std::string &root, DataMode mode);
        Food101(const std::string &root, DataMode mode , bool download);
        Food101(const std::string &root, DataMode mode , bool download, vector<std::function<torch::Tensor(torch::Tensor)>> transforms);


    private :
        std::string url = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz";
        fs::path dataset_file_name = "food-101.tar.gz";
        std::string dataset_file_md5 = "85eeb15f3717b99a5da872d97d918f87";
        fs::path dataset_folder_name = "food-101";
        std::size_t images_number = 101'000;
        std::vector<string> classes_name;
        std::map<string, int> classes_map;

        void load_data();

        void load_classes();

        void check_resources();
    };
}
```


