#pragma once
#include "../headers/datasets.h"
#include "base.h"

using namespace std;
namespace fs = std::filesystem;


namespace xt::data::datasets {
    class SBDataset : BaseDataset {
    public :
        SBDataset(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        SBDataset(const fs::path &root, DatasetArguments args);

    private:
        fs::path url = fs::path("https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz");
        std::string md5 = "82b4d87ceb2ed10f6038a1cba92111cb";
        fs::path filename = fs::path("benchmark.tgz");

        fs::path voc_train_url = fs::path("https://www.cs.cornell.edu/~bharathh/train_noval.txt");
        fs::path voc_split_filename = fs::path("train_noval.txt");
        std::string voc_split_md5 = "79bff800c5f0b1ec6b21080a3c066722";
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);


    };
}
