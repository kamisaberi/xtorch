#pragma once

#include "datasets/common.h"

using namespace std;
namespace fs = std::filesystem;

namespace xt::data::datasets {
    class CarlaStereo : public  xt::datasets::Dataset {
    public :
        explicit CarlaStereo(const std::string& root);
        CarlaStereo(const std::string& root, xt::datasets::DataMode mode);
        CarlaStereo(const std::string& root, xt::datasets::DataMode mode, bool download);
        CarlaStereo(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer);
        CarlaStereo(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer,
                   std::unique_ptr<xt::Module> target_transformer);


    private :
        void load_data();

        void check_resources();
    };



}
