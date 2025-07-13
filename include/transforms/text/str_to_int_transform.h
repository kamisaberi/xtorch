#pragma once

#include "../common.h"


namespace xt::transforms::text {

    /**
     * @class StrToIntTransform
     * @brief A text transformation that converts a sequence of string tokens into
     *        a sequence of integer IDs based on a provided vocabulary.
     *
     * This module acts as a vocabulary lookup. It is essential for converting the
     * human-readable output of a tokenizer into the numerical format required by
     * neural network models.
     */
    class StrToIntTransform : public xt::Module {
    public:
        /**
         * @brief Constructs the StrToIntTransform.
         *
         * @param vocab A map where keys are string tokens and values are their
         *              corresponding integer IDs.
         * @param unk_token The string representation of the "unknown" token (e.g., "[UNK]").
         *                  The transform will use the ID of this token for any
         *                  word not found in the vocabulary. Defaults to "[UNK]".
         */
        explicit StrToIntTransform(
                const std::unordered_map<std::string, long>& vocab,
                const std::string& unk_token = "[UNK]"
        );

        /**
         * @brief Executes the token-to-ID conversion.
         * @param tensors An initializer list expected to contain a single
         *                std::vector<std::string> representing the tokenized text.
         * @return An std::any containing the resulting std::vector<long> of integer IDs.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        const std::unordered_map<std::string, long>& vocab_;
        long unk_id_;
    };

} // namespace xt::transforms::text