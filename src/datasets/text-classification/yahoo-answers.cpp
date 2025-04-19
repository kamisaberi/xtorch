#include "../../../include/datasets/text-classification/yahoo-answers.h"

namespace xt::data::datasets {

    YahooAnswers::YahooAnswers(const std::string &root): YahooAnswers::YahooAnswers(root, DataMode::TRAIN, false) {
    }

    YahooAnswers::YahooAnswers(const std::string &root, DataMode mode): YahooAnswers::YahooAnswers(root, mode, false) {
    }

    YahooAnswers::YahooAnswers(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("YahooAnswers: YahooAnswers not implemented");
    }


    YahooAnswers::YahooAnswers(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("YahooAnswers: YahooAnswers not implemented");
    }


}
