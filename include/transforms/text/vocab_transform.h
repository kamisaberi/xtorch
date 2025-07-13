#pragma once

#include "../common.h"

#include <string>
#include <vector>
#include <unordered_map>

namespace xt::transforms::text {

    /**
     * @class VocabTransform
     * @brief A module that loads a vocabulary from a file into an in-memory map.
     *
     * This transform reads a text file where each line corresponds to a token,
     * and its line number corresponds to its integer ID. It outputs a
     * `std::unordered_map<std::string, long>` which can then be used by other
     * transforms, such as `StrToIntTransform`.
     */
    class VocabTransform : public xt::Module {
    public:
        /**
         * @brief Constructs the VocabTransform.
         * @param vocab_path The path to the line-delimited vocabulary file.
         */
        explicit VocabTransform(const std::string& vocab_path);

        /**
         * @brief Executes the vocabulary loading operation.
         *
         * This method does not require any input tensors; it acts as a data source.
         *
         * @param tensors An initializer list that is expected to be empty.
         * @return An std::any containing a `std::unordered_map<std::string, long>`
         *         representing the loaded vocabulary.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        std::string vocab_path_;
    };

} // namespace xt::transforms::text