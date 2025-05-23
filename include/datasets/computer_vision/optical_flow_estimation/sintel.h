#pragma once

#include "datasets/common.h"

using namespace std;
namespace fs = std::filesystem;


namespace xt::data::datasets
{
    class Sintel : public xt::datasets::Dataset
    {
        /*
               """`Sintel <http://sintel.is.tue.mpg.de/>`_ Dataset for optical flow.

                   The dataset is expected to have the following structure: ::

                       root
                           Sintel
                               testing
                                   clean
                                       scene_1
                                       scene_2
                                       ...
                                   final
                                       scene_1
                                       scene_2
                                       ...
                               training
                                   clean
                                       scene_1
                                       scene_2
                                       ...
                                   final
                                       scene_1
                                       scene_2
                                       ...
                                   flow
                                       scene_1
                                       scene_2
                                       ...

                   Args:
                       root (str or ``pathlib.Path``): Root directory of the Sintel Dataset.
                       split (string, optional): The dataset split, either "train" (default) or "test"
                       pass_name (string, optional): The pass to use, either "clean" (default), "final", or "both". See link above for
                           details on the different passes.
                       transforms (callable, optional): A function/transform that takes in
                           ``img1, img2, flow, valid_flow_mask`` and returns a transformed version.
                           ``valid_flow_mask`` is expected for consistency with other datasets which
                           return a built-in valid mask, such as :class:`~torchvision.datasets.KittiFlow`.
                   """

         */
    public :
        explicit Sintel(const std::string& root);
        Sintel(const std::string& root, xt::datasets::DataMode mode);
        Sintel(const std::string& root, xt::datasets::DataMode mode, bool download);
        Sintel(const std::string& root, xt::datasets::DataMode mode, bool download,
               std::unique_ptr<xt::Module> transformer);
        Sintel(const std::string& root, xt::datasets::DataMode mode, bool download,
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
