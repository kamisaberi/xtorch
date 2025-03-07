#pragma once

#include "../base/datasets.h"


using namespace std;
namespace fs = std::filesystem;


namespace torch::ext::data::datasets {
   class EuroSAT : torch::data::Dataset<EuroSAT> {
   private:
       std::string download_url = "https://huggingface.co/datasets/torchgeo/eurosat/resolve/c877bcd43f099cd0196738f714544e355477f3fd/EuroSAT.zip";
       fs::path archive_file_name = "EuroSAT.zip";
       std::string archive_file_md5 = "c8fa014336c82ac7804f0398fcb19387";

   public :
       EuroSAT();
    };
}
