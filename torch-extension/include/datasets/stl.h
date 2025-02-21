#pragma once
//#include <vector>
#include <fstream>
//#include <iostream>
//#include <string>
#include <filesystem>
//#include <curl/curl.h>
#include <torch/torch.h>
#include <vector>
#include <tuple>
#include <map>
//#include <fstream>
//#include <iostream>
//#include <string>
//#include <filesystem>
#include "../utils/downloader.h"
#include "../utils/archiver.h"
#include "../utils/md5.h"
#include "../exceptions/implementation.h"

using namespace std;
namespace fs = std::filesystem;


namespace torch::ext::data::datasets {
    class STL10 : torch::data::Dataset<STL10> {
    private :

        fs::path base_folder = fs::path("stl10_binary");
        fs::path url = fs::path("http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz");
        fs::path filename = fs::path("stl10_binary.tar.gz");
        std::string tgz_md5 = "91f7769df0f17e558f3565bffb0c7dfb";
        fs::path class_names_file = fs::path("class_names.txt");
        fs::path folds_list_file = fs::path("fold_indices.txt");

        std::vector<std::tuple<fs::path, std::string >> train_list = {
                {"train_X.bin", "918c2871b30a85fa023e0c44e0bee87f"},
                {"train_y.bin", "5a34089d4802c674881badbb80307741"},
                {"unlabeled_X.bin", "5242ba1fed5e4be9e1e742405eb56ca4"}
        };

        std::vector<std::tuple<fs::path, std::string >> test_list = {{"test_X.bin", "7f263ba9f9e0b06b93213547f721ac82"},
                                                                     {"test_y.bin", "36f9794fa4beb8a2c72628de14fa638e"}};
        std::vector<std::string> splits = {"train", "train+unlabeled", "unlabeled", "test"};


    public :
        STL10();
    };
}
