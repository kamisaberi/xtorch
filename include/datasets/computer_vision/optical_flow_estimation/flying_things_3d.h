#pragma once

#include "include/datasets/common.h"

using namespace std;
namespace fs = std::filesystem;

namespace xt::data::datasets {
    class FlyingThings3D : public xt::datasets::Dataset {
        /*
        """`FlyingThings3D <https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html>`_ dataset for optical flow.
       The dataset is expected to have the following structure: ::
            root
                FlyingThings3D
                    frames_cleanpass
                        TEST
                        TRAIN
                    frames_finalpass
                        TEST
                        TRAIN
                    optical_flow
                        TEST
                        TRAIN

        Args:
            root (str or ``pathlib.Path``): Root directory of the intel FlyingThings3D Dataset.
            split (string, optional): The dataset split, either "train" (default) or "test"
            pass_name (string, optional): The pass to use, either "clean" (default) or "final" or "both". See link above for
                details on the different passes.
            camera (string, optional): Which camera to return images from. Can be either "left" (default) or "right" or "both".
            transforms (callable, optional): A function/transform that takes in
                ``img1, img2, flow, valid_flow_mask`` and returns a transformed version.
                ``valid_flow_mask`` is expected for consistency with other datasets which
                return a built-in valid mask, such as :class:`~torchvision.datasets.KittiFlow`.
        """
    */
    public :
        explicit FlyingThings3D(const std::string& root);
        FlyingThings3D(const std::string& root, xt::datasets::DataMode mode);
        FlyingThings3D(const std::string& root, xt::datasets::DataMode mode, bool download);
        FlyingThings3D(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer);
        FlyingThings3D(const std::string& root, xt::datasets::DataMode mode, bool download,
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
