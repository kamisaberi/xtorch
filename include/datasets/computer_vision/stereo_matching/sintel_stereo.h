#pragma once

#include "datasets/common.h"

using namespace std;
namespace fs = std::filesystem;

namespace xt::data::datasets
{
    class SintelStereo : xt::datasets::Dataset
    {
        /*
               """Sintel `Stereo Dataset <http://sintel.is.tue.mpg.de/stereo>`_.

                   The dataset is expected to have the following structure: ::

                       root
                           Sintel
                               training
                                   final_left
                                       scene1
                                           img1.png
                                           img2.png
                                           ...
                                       ...
                                   final_right
                                       scene2
                                           img1.png
                                           img2.png
                                           ...
                                       ...
                                   disparities
                                       scene1
                                           img1.png
                                           img2.png
                                           ...
                                       ...
                                   occlusions
                                       scene1
                                           img1.png
                                           img2.png
                                           ...
                                       ...
                                   outofframe
                                       scene1
                                           img1.png
                                           img2.png
                                           ...
                                       ...

                   Args:
                       root (str or ``pathlib.Path``): Root directory where Sintel Stereo is located.
                       pass_name (string): The name of the pass to use, either "final", "clean" or "both".
                       transforms (callable, optional): A function/transform that takes in a sample and returns a transformed version.
                   """

         */
    public :
        explicit SintelStereo(const std::string& root);
        SintelStereo(const std::string& root, xt::datasets::DataMode mode);
        SintelStereo(const std::string& root, xt::datasets::DataMode mode, bool download);
        SintelStereo(const std::string& root, xt::datasets::DataMode mode, bool download,
                     std::unique_ptr<xt::Module> transformer);
        SintelStereo(const std::string& root, xt::datasets::DataMode mode, bool download,
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
