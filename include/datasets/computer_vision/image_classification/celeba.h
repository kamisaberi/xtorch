#pragma once

#include "../../common.h"


using namespace std;
namespace fs = std::filesystem;


namespace xt::datasets
{
    class CelebA : public xt::datasets::Dataset
    {
    public :
        explicit CelebA(const std::string& root);
        CelebA(const std::string& root, xt::datasets::DataMode mode);
        CelebA(const std::string& root, xt::datasets::DataMode mode, bool download);
        CelebA(const std::string& root, xt::datasets::DataMode mode, bool download,
               std::unique_ptr<xt::Module> transformer);
        CelebA(const std::string& root, xt::datasets::DataMode mode, bool download,
               std::unique_ptr<xt::Module> transformer,
               std::unique_ptr<xt::Module> target_transformer);

        torch::data::Example<> get(size_t index) override;

        torch::optional<size_t> size() const override;

    private:
        vector<std::tuple<string, string, fs::path>> resources = {
            {"0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", fs::path("img_align_celeba.zip")},
            {"0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", fs::path("list_attr_celeba.txt")},
            {"1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", fs::path("identity_CelebA.txt")},
            {"0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", fs::path("list_bbox_celeba.txt")},
            {
                "0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c",
                fs::path("list_landmarks_align_celeba.txt")
            },
            {"0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", fs::path("list_eval_partition.txt")}
        };


        fs::path dataset_folder_name = "celeba";
        fs::path images_folder = "img_align_celeba";
        bool download = false;
        fs::path root;
        fs::path dataset_path;
        vector<fs::path> files;


        void load_data();

        void check_resources();
    };
}
