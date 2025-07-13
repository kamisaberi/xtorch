#pragma once

#include "../common.h"


#include <torch/torch.h> // Assumes LibTorch is installed
#include <string>
#include <vector>
#include <any>
#include <unordered_map>

namespace xt::transforms::target {

    /**
     * @class MultiLabelBinarizer
     * @brief A target transformation that converts a list of active labels for a
     *        sample into a multi-hot binary tensor.
     *
     * This is the standard label format for multi-label classification problems.
     * Given a pre-defined vocabulary mapping labels to indices, this transform
     * creates a binary vector that is all zeros except for a '1' at the index
     * of each active label for the sample.
     */
    class MultiLabelBinarizer : public xt::Module {
    public:
        /**
         * @brief Constructs the MultiLabelBinarizer.
         *
         * @param class_to_index A map where keys are the string names of all
         *                       possible classes and values are their unique integer
         *                       indices (from 0 to num_classes-1). The size of
         *                       this map determines the length of the output tensor.
         */
        explicit MultiLabelBinarizer(const std::unordered_map<std::string, long>& class_to_index);

        /**
         * @brief Executes the multi-hot encoding operation.
         * @param tensors An initializer list expected to contain a single
         *                `std::vector<std::string>` representing the list of
         *                active labels for one sample.
         * @return An std::any containing the multi-hot encoded `torch::Tensor`
         *         of shape (num_classes,).
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        const std::unordered_map<std::string, long>& class_to_index_;
        int num_classes_;
    };

} // namespace xt::transforms::target