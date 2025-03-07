#pragma once
#include "../base/datasets.h"



namespace torch::ext::data::datasets {
   class OxfordIIITPet : torch::data::Dataset<OxfordIIITPet> {

       vector<std::tuple<fs::path , std::string >> _RESOURCES = {
               {fs::path("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"), "5c4f3ee8e5d25df40f4fd59a7f44e54c"},
               {fs::path("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"), "95a8c909bbe2e81eed6a22bccdf3f68f"}
       };
       vector<std::string> _VALID_TARGET_TYPES = {"category", "binary-category", "segmentation"};

   public :
       OxfordIIITPet();
    };
}
