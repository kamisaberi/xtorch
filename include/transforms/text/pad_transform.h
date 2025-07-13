#pragma once

#include "../common.h"

namespace xt::transforms::text {

    /**
     * @brief Specifies the direction for padding.
     */
    enum class PaddingDirectionP {
        RIGHT, // Add padding to the end of the sequence.
        LEFT   // Add padding to the beginning of the sequence.
    };

    /**
     * @brief Specifies the direction for truncation.
     */
    enum class TruncationDirectionP {
        RIGHT, // Remove elements from the end of the sequence.
        LEFT   // Remove elements from the beginning of the sequence.
    };


    /**
     * @class PadTransform
     * @brief A text transformation that pads or truncates a sequence of token IDs
     *        to a fixed length.
     *
     * This is a crucial step for batching sequences for input into many neural
     * network models, which require fixed-size inputs.
     */
    class PadTransform : public xt::Module {
    public:
        /**
         * @brief Constructs the PadTransform.
         *
         * @param max_len The target length for all sequences.
         * @param pad_id The integer ID used for padding. Defaults to 0, which is
         *               standard for many models.
         * @param pad_dir The direction to add padding. Defaults to RIGHT.
         * @param trunc_dir The direction to truncate from if the sequence is too
         *                  long. Defaults to RIGHT.
         */
        explicit PadTransform(
                int max_len,
                long pad_id = 0,
                PaddingDirectionP pad_dir = PaddingDirectionP::RIGHT,
                TruncationDirectionP trunc_dir = TruncationDirectionP::RIGHT
        );

        /**
         * @brief Executes the padding and truncation operation.
         * @param tensors An initializer list expected to contain a single
         *                std::vector<long> representing the token IDs.
         * @return An std::any containing the resulting fixed-length
         *         std::vector<long>.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int max_len_;
        long pad_id_;
        PaddingDirectionP pad_dir_;
        TruncationDirectionP trunc_dir_;
    };

} // namespace xt::transforms::text