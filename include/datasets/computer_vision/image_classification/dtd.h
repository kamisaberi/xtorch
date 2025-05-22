#pragma once

#include "datasets/common.h"


using namespace std;
namespace fs = std::filesystem;


namespace xt::data::datasets {
    class DTD : public xt::datasets::Dataset {
    public :
        explicit DTD(const std::string& root);
        DTD(const std::string& root, xt::datasets::DataMode mode);
        DTD(const std::string& root, xt::datasets::DataMode mode, bool download);
        DTD(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer);
        DTD(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer,
                   std::unique_ptr<xt::Module> target_transformer);

    private :
        void load_data();

        void check_resources();
    };
}
