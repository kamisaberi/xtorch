#pragma once

#include "include/datasets/common.h"

using namespace std;
namespace fs = std::filesystem;

namespace xt::data::datasets
{
    class InStereo2k : xt::datasets::Dataset
    {
        /*
        """`InStereo2k <https://github.com/YuhuaXu/StereoDataset>`_ dataset.

        The dataset is expected to have the following structure: ::

            root
                InStereo2k
                    train
                        scene1
                            left.png
                            right.png
                            left_disp.png
                            right_disp.png
                            ...
                        scene2
                        ...
                    test
                        scene1
                            left.png
                            right.png
                            left_disp.png
                            right_disp.png
                            ...
                        scene2
                        ...

        Args:
            root (str or ``pathlib.Path``): Root directory where InStereo2k is located.
            split (string): Either "train" or "test".
            transforms (callable, optional): A function/transform that takes in a sample and returns a transformed version.
        """
         */
    public :
        explicit InStereo2k(const std::string& root);
        InStereo2k(const std::string& root, xt::datasets::DataMode mode);
        InStereo2k(const std::string& root, xt::datasets::DataMode mode, bool download);
        InStereo2k(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer);
        InStereo2k(const std::string& root, xt::datasets::DataMode mode, bool download,
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
