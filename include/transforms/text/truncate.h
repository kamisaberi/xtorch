#pragma once

#include "../common.h"


#include <vector>
#include <string>

namespace xt::transforms::text {

    /**
     * @brief Specifies the direction for truncation.
     */
    enum class TruncationDirectionT {
        RIGHT, // Remove elements from the end of the sequence (default).
        LEFT   // Remove elements from the beginning of the sequence.
    };


    /**
     * @class Truncate
     * @brief A transformation that truncates a sequence to a maximum length.
     *
     * This module ensures that a sequence (of either tokens or token IDs) does
     * not exceed a specified length. This is often a necessary prerequisite
     * before padding or feeding data into a model.
     */
    class Truncate : public xt::Module {
    public:
        /**
         * @brief Constructs the Truncate transform.
         *
         * @param max_len The maximum allowed length for the sequence.
         * @param trunc_dir The direction from which to remove elements if the
         *                  sequence is too long. Defaults to RIGHT.
         */
        explicit Truncate(
                int max_len,
                TruncationDirectionT trunc_dir = TruncationDirectionT::RIGHT
        );

        /**
         * @brief Executes the truncation operation.
         * @param tensors An initializer list expected to contain a single sequence.
         *                The sequence can be either a std::vector<std::string> (tokens)
         *                or a std::vector<long> (token IDs).
         * @return An std::any containing the resulting truncated sequence, with the
         *         same type as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int max_len_;
        TruncationDirectionT trunc_dir_;
    };

} // namespace xt::transforms::text