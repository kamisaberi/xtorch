#include "include/transforms/target/multi_label_binarizer.h"

namespace xt::transforms::target
{
    MultiLabelBinarizer::MultiLabelBinarizer() = default;

    MultiLabelBinarizer::MultiLabelBinarizer(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto MultiLabelBinarizer::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
