#pragma once

#include "../common.h"


namespace xt::transforms::text {

    /**
     * @brief An enum to specify where the token should be added.
     */
    enum class AddTokenLocation {
        BEGIN,
        END
    };

    /**
     * @class AddToken
     * @brief A text transformation that adds a special token to a sequence of tokens.
     *
     * This is commonly used to add classification ([CLS]), separator ([SEP]),
     * or beginning-of-sequence ([BOS]) tokens for language models.
     */
    class AddToken : public xt::Module {
    public:
        /**
         * @brief Constructs the AddToken transform.
         * @param token The string token to add (e.g., "[CLS]").
         * @param location Where to add the token in the sequence (BEGIN or END).
         *                 Defaults to BEGIN.
         */
        explicit AddToken(
                const std::string& token,
                AddTokenLocation location = AddTokenLocation::BEGIN
        );

        /**
         * @brief Executes the token addition operation.
         * @param tensors An initializer list expected to contain a single
         *                std::vector<std::string> representing the tokenized text.
         * @return An std::any containing the resulting std::vector<std::string>
         *         with the added token.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        std::string token_;
        AddTokenLocation location_;
    };

} // namespace xt::transforms::text