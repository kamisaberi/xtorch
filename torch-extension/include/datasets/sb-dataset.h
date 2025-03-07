#pragma once
#include "../base/datasets.h"

using namespace std;
namespace fs = std::filesystem;


namespace torch::ext::data::datasets {
    class SBDataset : torch::data::Dataset<SBDataset> {
    private:
        fs::path url = fs::path("https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz");
        std::string md5 = "82b4d87ceb2ed10f6038a1cba92111cb";
        fs::path filename = fs::path("benchmark.tgz");

        fs::path voc_train_url = fs::path("https://www.cs.cornell.edu/~bharathh/train_noval.txt");
        fs::path voc_split_filename = fs::path("train_noval.txt");
        std::string voc_split_md5 = "79bff800c5f0b1ec6b21080a3c066722";

    public :
        SBDataset();
    };
}
