#pragma once

#include "../../base/base.h"
#include "../../../headers/datasets.h"


using namespace std;
namespace fs = std::filesystem;


namespace xt::data::datasets {

    class Caltech256 : public BaseDataset {
    public :
        explicit  Caltech256(const std::string &root);
        Caltech256(const std::string &root, DataMode mode);
        Caltech256(const std::string &root, DataMode mode , bool download);
        Caltech256(const std::string &root, DataMode mode , bool download, TransformType transforms);


    private:
        vector<std::tuple<string, string, string> > resources = {
            {
                "https://drive.google.com/file/d/1r6o0pSROcV1_VwT4oSjA2FBUSCWGuxLK", "256_ObjectCategories.tar",
                "67b4f42ca05d46448c6bb8ecd2220f6d"
            },
        };
        fs::path dataset_folder_name = "caltech256";

        void load_data();

        void check_resources();
    };
}
