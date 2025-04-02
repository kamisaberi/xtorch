#pragma once
#include "../headers/datasets.h"
#include "base.h"


namespace xt::data::datasets {
   class Sintel : BaseDataset {
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
       Sintel(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

       Sintel(const fs::path &root, DatasetArguments args);
   private :
       void load_data(DataMode mode = DataMode::TRAIN);

       void check_resources(const std::string &root, bool download = false);

   };
   class SintelStereo : torch::data::Dataset<SintelStereo> {
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
       SintelStereo(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

       SintelStereo(const fs::path &root, DatasetArguments args);
   private :
       void load_data(DataMode mode = DataMode::TRAIN);

       void check_resources(const std::string &root, bool download = false);
    };
}
