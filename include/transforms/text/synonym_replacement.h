#pragma once

#include "../common.h"


namespace xt::transforms::text {

    /**
     * @class SynonymDatabase
     * @brief An abstract interface for a synonym lookup service.
     *
     * This defines the contract for any class that can provide synonyms for a
     * given word. This allows the SynonymReplacement transform to be independent
     * of the actual data source (e.g., a file, an API, a map).
     */
    class SynonymDatabase {
    public:
        virtual ~SynonymDatabase() = default;

        /**
         * @brief Retrieves a list of synonyms for a word.
         * @param word The word to look up.
         * @return A vector of strings containing synonyms. If the word is not
         *         found or has no synonyms, an empty vector is returned.
         */
        virtual auto get_synonyms(const std::string& word) const -> std::vector<std::string> = 0;
    };


    /**
     * @class SynonymReplacement
     * @brief A text augmentation transform that randomly replaces words with their synonyms.
     *
     * This transform iterates through a sequence of tokens and, for each token,
     * randomly decides whether to replace it with one of its synonyms.
     */
    class SynonymReplacement : public xt::Module {
    public:
        /**
         * @brief Constructs the SynonymReplacement transform.
         *
         * @param synonym_db A shared pointer to a concrete SynonymDatabase implementation.
         * @param replacement_prob The probability (0.0 to 1.0) of replacing a word
         *                         that has synonyms. Defaults to 0.2 (i.e., 20% chance).
         */
        explicit SynonymReplacement(
                std::shared_ptr<SynonymDatabase> synonym_db,
                float replacement_prob = 0.2f
        );

        /**
         * @brief Executes the synonym replacement operation.
         * @param tensors An initializer list expected to contain a single
         *                std::vector<std::string> of tokens.
         * @return An std::any containing the resulting augmented std::vector<std::string>.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        /**
         * @brief Gets a random synonym for a word from the database.
         * @param word The word to replace.
         * @return A randomly chosen synonym, or an empty string if none are available.
         */
        auto get_random_synonym(const std::string& word) -> std::string;

        std::shared_ptr<SynonymDatabase> synonym_db_;
        float replacement_prob_;

        // Random number generation engine for reproducibility and efficiency
        std::mt19937 random_engine_;
        std::uniform_real_distribution<float> prob_distribution_;
    };

} // namespace xt::transforms::text