#pragma once

#include "../../common.h"
using namespace std;
namespace fs = std::filesystem;

namespace xt::datasets
{
    class FlyingChairs : public xt::datasets::Dataset
    {
        /*
        """`FlyingChairs <https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs>`_ Dataset for optical flow.
        You will also need to download the FlyingChairs_train_val.txt file from the dataset page.
        The dataset is expected to have the following structure: ::

            root
                FlyingChairs
                    data
                        00001_flow.flo
                        00001_img1.ppm
                        00001_img2.ppm
                        ...
                    FlyingChairs_train_val.txt
        Args:
            root (str or ``pathlib.Path``): Root directory of the FlyingChairs Dataset.
            split (string, optional): The dataset split, either "train" (default) or "val"
            transforms (callable, optional): A function/transform that takes in
                ``img1, img2, flow, valid_flow_mask`` and returns a transformed version.
                ``valid_flow_mask`` is expected for consistency with other datasets which
                return a built-in valid mask, such as :class:`~torchvision.datasets.KittiFlow`.
        """
        */
    public :
        explicit FlyingChairs(const std::string& root);
        FlyingChairs(const std::string& root, xt::datasets::DataMode mode);
        FlyingChairs(const std::string& root, xt::datasets::DataMode mode, bool download);
        FlyingChairs(const std::string& root, xt::datasets::DataMode mode, bool download,
                     std::unique_ptr<xt::Module> transformer);
        FlyingChairs(const std::string& root, xt::datasets::DataMode mode, bool download,
                     std::unique_ptr<xt::Module> transformer,
                     std::unique_ptr<xt::Module> target_transformer);

    private :
        // TODO fs::path dataset_folder_name
        fs::path dataset_folder_name = "?";

        bool download = false;
        fs::path root;
        fs::path dataset_path;

        void load_data();

        void check_resources();
    };
}
