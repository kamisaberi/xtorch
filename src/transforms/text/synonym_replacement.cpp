#include <transforms/text/synonym_replacement.h>

#include <stdexcept>
#include <chrono>

// Example Main - Uncomment to run a standalone test
/*
#include <iostream>
#include <unordered_map>

// Mock xt::Module for testing purposes
class xt::Module {
public:
    virtual ~Module() = default;
    virtual auto forward(std::initializer_list<std::any> tensors) -> std::any = 0;
};

// Helper to print a vector of tokens
void print_tokens(const std::string& name, const std::vector<std::string>& tokens) {
    std::cout << name << "[ ";
    for (const auto& token : tokens) {
        std::cout << "\"" << token << "\" ";
    }
    std::cout << "]" << std::endl;
}


// --- A Mock Synonym Database for the Example ---
// In a real application, this class might read from a large WordNet file.
class MockSynonymDatabase : public xt::transforms::text::SynonymDatabase {
public:
    MockSynonymDatabase() {
        // Populate our simple database
        db_["quick"] = {"fast", "swift", "rapid"};
        db_["brown"] = {"dark", "tawny"};
        db_["jumps"] = {"leaps", "hops", "springs"};
        db_["lazy"] = {"idle", "slothful"};
    }

    auto get_synonyms(const std::string& word) const -> std::vector<std::string> override {
        auto it = db_.find(word);
        if (it != db_.end()) {
            return it->second;
        }
        return {}; // Return empty vector if not found
    }

private:
    std::unordered_map<std::string, std::vector<std::string>> db_;
};


int main() {
    // 1. --- Setup ---
    // Create an instance of our mock database
    auto db = std::make_shared<MockSynonymDatabase>();

    // Instantiate the transform with a high probability to see more changes
    xt::transforms::text::SynonymReplacement augmenter(db, 0.5f); // 50% chance to replace

    // 2. --- Input Data ---
    std::vector<std::string> tokens = {"the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"};
    print_tokens("Original:  ", tokens);

    // 3. --- Run the Transform Multiple Times ---
    // Since it's random, the output will be different each time.
    std::cout << "\nRunning augmentation 5 times..." << std::endl;
    for (int i = 0; i < 5; ++i) {
        auto augmented_any = augmenter.forward({tokens});
        auto augmented_tokens = std::any_cast<std::vector<std::string>>(augmented_any);
        print_tokens("Augmented " + std::to_string(i + 1) + ": ", augmented_tokens);
    }

    return 0;
}
*/

namespace xt::transforms::text {

    SynonymReplacement::SynonymReplacement(std::shared_ptr<SynonymDatabase> synonym_db, float replacement_prob)
            : synonym_db_(synonym_db), replacement_prob_(replacement_prob), prob_distribution_(0.0f, 1.0f) {

        if (!synonym_db_) {
            throw std::invalid_argument("SynonymDatabase provided to SynonymReplacement must not be null.");
        }
        if (replacement_prob_ < 0.0f || replacement_prob_ > 1.0f) {
            throw std::invalid_argument("Replacement probability must be between 0.0 and 1.0.");
        }

        // Seed the random number generator for different results on each run.
        // Using a high-resolution clock ensures a good seed.
        unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        random_engine_.seed(seed);
    }

    auto SynonymReplacement::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("SynonymReplacement::forward received an empty list.");
        }

        std::vector<std::string> input_tokens;
        try {
            input_tokens = std::any_cast<std::vector<std::string>>(any_vec[0]);
        } catch (const std::bad_any_cast& e) {
            throw std::invalid_argument("Input to SynonymReplacement must be of type std::vector<std::string>.");
        }

        std::vector<std::string> output_tokens = input_tokens; // Work on a copy

        // 2. --- Augmentation Loop ---
        for (size_t i = 0; i < output_tokens.size(); ++i) {
            // Check if we should try to replace this word based on the probability
            if (prob_distribution_(random_engine_) < replacement_prob_) {
                std::string synonym = get_random_synonym(output_tokens[i]);
                if (!synonym.empty()) {
                    output_tokens[i] = synonym;
                }
            }
        }

        return output_tokens;
    }

    auto SynonymReplacement::get_random_synonym(const std::string& word) -> std::string {
        // Retrieve all possible synonyms for the word
        std::vector<std::string> synonyms = synonym_db_->get_synonyms(word);

        if (synonyms.empty()) {
            return ""; // No synonyms found
        }

        // Pick a random index for the synonym
        std::uniform_int_distribution<size_t> synonym_dist(0, synonyms.size() - 1);
        size_t random_index = synonym_dist(random_engine_);

        // Ensure the chosen synonym is not the original word itself
        if (synonyms[random_index] == word && synonyms.size() > 1) {
            // Simple strategy: just pick the next one in the list (and wrap around)
            return synonyms[(random_index + 1) % synonyms.size()];
        }

        return synonyms[random_index];
    }

} // namespace xt::transforms::text```