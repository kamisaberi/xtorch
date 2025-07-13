#pragma once

#include "../common.h"



namespace xt::transforms::text {

    /**
     * @class TranslationClient
     * @brief An abstract interface for a text translation service.
     *
     * This class defines the contract for any translation client. A concrete
     * implementation would handle the actual logic of calling a translation
     * model or API. This allows the BackTranslation transform to be independent
     * of any specific translation service (e.g., Google Translate, Hugging Face, etc.).
     */
    class TranslationClient {
    public:
        virtual ~TranslationClient() = default;

        /**
         * @brief Translates a given text string.
         * @param text The text to translate.
         * @return The translated string.
         */
        virtual auto translate(const std::string& text) -> std::string = 0;
    };


    /**
     * @class BackTranslation
     * @brief A text data augmentation transform using back-translation.
     *
     * This transform takes a text string, translates it to an intermediate
     * language, and then translates it back to the original language. This
     * is a powerful technique for creating paraphrased versions of the text.
     */
    class BackTranslation : public xt::Module {
    public:
        /**
         * @brief Constructs the BackTranslation transform.
         *
         * This constructor uses dependency injection. It requires two pre-configured
         * translation clients.
         *
         * @param to_intermediate_client A shared pointer to a client that translates
         *                               from the source language to the intermediate language.
         * @param from_intermediate_client A shared pointer to a client that translates
         *                                 from the intermediate language back to the source.
         */
        explicit BackTranslation(
                std::shared_ptr<TranslationClient> to_intermediate_client,
                std::shared_ptr<TranslationClient> from_intermediate_client
        );

        /**
         * @brief Executes the back-translation operation.
         * @param tensors An initializer list expected to contain a single
         *                std::string to be augmented.
         * @return An std::any containing the resulting back-translated std::string.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        std::shared_ptr<TranslationClient> to_intermediate_client_;
        std::shared_ptr<TranslationClient> from_intermediate_client_;
    };

} // namespace xt::transforms::text