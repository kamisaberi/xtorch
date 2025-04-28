#pragma once
#include "base.h"
#include "../../headers/datasets.h"


using namespace std;
namespace fs = std::filesystem;


namespace xt::data::datasets
{
    class MNISTBase : public BaseDataset
    {
    public:
        explicit MNISTBase(const std::string& root);
        MNISTBase(const std::string& root, DataMode mode);
        MNISTBase(const std::string& root, DataMode mode, bool download);
        MNISTBase(const std::string& root, DataMode mode, bool download,
                  vector<std::function<torch::Tensor(torch::Tensor)>> transforms);
        void read_images(const std::string& file_path, int num_images);
        void read_labels(const std::string& file_path, int num_labels);
    };
}
