#include "include/dropouts/embedding_dropout.h"



#include <torch/torch.h>
#include <vector>
#include <ostream> // For std::ostream

struct EmbeddingDropoutImpl : torch::nn::Module {
    double p_drop_entire_embedding_; // Probability of dropping an entire embedding vector for a token
    double epsilon_ = 1e-7;           // For numerical stability

    EmbeddingDropoutImpl(double p_drop_entire_embedding = 0.1)
        : p_drop_entire_embedding_(p_drop_entire_embedding) {
        TORCH_CHECK(p_drop_entire_embedding_ >= 0.0 && p_drop_entire_embedding_ <= 1.0,
                    "EmbeddingDropout probability must be between 0 and 1.");
    }

    // The input to this forward method is expected to be the result of an embedding lookup,
    // typically of shape (Batch, SequenceLength, EmbeddingDim) or (Batch, EmbeddingDim) if SeqLen=1.
    // It could also be (SequenceLength, EmbeddingDim) if Batch=1 and squeezed, or just (VocabSize, EmbeddingDim)
    // if applied to the whole weight matrix (though less common for "EmbeddingDropout" in this specific way).
    // This implementation assumes dropout is applied per token *within a sequence*.
    torch::Tensor forward(const torch::Tensor& embedding_output) {
        if (!this->is_training() || p_drop_entire_embedding_ == 0.0) {
            return embedding_output;
        }
        if (p_drop_entire_embedding_ == 1.0) {
            return torch::zeros_like(embedding_output);
        }

        TORCH_CHECK(embedding_output.dim() >= 2, "EmbeddingDropout expects input with at least 2 dimensions (e.g., [SeqLen, EmbDim] or [Batch, SeqLen, EmbDim]).");

        double keep_prob = 1.0 - p_drop_entire_embedding_;

        // Determine the shape for the mask. We want to drop entire embedding vectors.
        // If input is (B, S, E), mask should be (B, S, 1) and broadcast.
        // If input is (S, E), mask should be (S, 1) and broadcast.
        torch::IntArrayRef input_sizes = embedding_output.sizes();
        std::vector<int64_t> mask_shape_vec;

        if (input_sizes.size() == 2) { // (SeqLen or Batch, EmbeddingDim)
            mask_shape_vec.push_back(input_sizes[0]); // Dropout per row of this 2D tensor
            mask_shape_vec.push_back(1);              // Broadcast across embedding dim
        } else if (input_sizes.size() == 3) { // (Batch, SeqLen, EmbeddingDim)
            mask_shape_vec.push_back(input_sizes[0]); // Batch
            mask_shape_vec.push_back(input_sizes[1]); // Sequence Length - dropout per token in sequence
            mask_shape_vec.push_back(1);              // Broadcast across embedding dim
        } else {
            // For other dimensions, this simple approach might not be ideal.
            // A common case is (Batch, SeqLen, EmbDim).
            // We'll proceed assuming the last dimension is EmbeddingDim.
            // This will create a mask that drops elements along all but the last dimension.
            // This is a more general way: create mask for all but last dim, then unsqueeze last dim.
            std::vector<int64_t> first_dims_shape;
            for (size_t i = 0; i < input_sizes.size() -1; ++i) {
                first_dims_shape.push_back(input_sizes[i]);
            }
            if (first_dims_shape.empty()) { // Input was (EmbeddingDim), unlikely but handle
                first_dims_shape.push_back(1); // Treat as (1, EmbeddingDim)
            }
            torch::Tensor mask_first_dims = torch::bernoulli(
                torch::full(first_dims_shape, keep_prob, embedding_output.options())
            ).to(embedding_output.dtype());
            torch::Tensor mask = mask_first_dims.unsqueeze(-1); // Add dim for broadcasting over EmbeddingDim
            return (embedding_output * mask) / (keep_prob + epsilon_);
        }

        // Generate mask with shape (B, S, 1) or (S, 1)
        torch::Tensor mask = torch::bernoulli(
            torch::full(mask_shape_vec, keep_prob, embedding_output.options())
        ).to(embedding_output.dtype());

        // Apply mask and scale (inverted dropout)
        return (embedding_output * mask) / (keep_prob + epsilon_);
    }

    void pretty_print(std::ostream& stream) const override {
        stream << "EmbeddingDropout(p_drop_entire_embedding=" << p_drop_entire_embedding_ << ")";
    }
};

TORCH_MODULE(EmbeddingDropout);


/*
// --- Example usage within a model that has an Embedding layer ---
// This is for context and not part of the EmbeddingDropoutImpl module itself.

struct MyModelWithEmbeddingDropout : torch::nn::Module {
    torch::nn::Embedding embedding_layer{nullptr};
    EmbeddingDropout embedding_dropout_module; // Instance of EmbeddingDropout

    MyModelWithEmbeddingDropout(int64_t num_embeddings, int64_t embedding_dim, double ed_p_drop = 0.1)
        : embedding_dropout_module(ed_p_drop) {
        embedding_layer = register_module("embedding", torch::nn::Embedding(num_embeddings, embedding_dim));
    }

    torch::Tensor forward(const torch::Tensor& input_indices) {
        // input_indices: (Batch, SeqLen) tensor of Long type
        torch::Tensor embeddings = embedding_layer->forward(input_indices); // Shape: (Batch, SeqLen, EmbeddingDim)

        // Apply EmbeddingDropout to the looked-up embeddings
        // The EmbeddingDropout module itself handles the training/eval mode check.
        embeddings = embedding_dropout_module->forward(embeddings);

        // ... rest of the model ...
        return embeddings.sum({1,2}); // Example: sum all features for a dummy output
    }
};
TORCH_MODULE(MyModelWithEmbeddingDropout);


#include <iostream>
void run_embedding_dropout_example() {
    torch::manual_seed(0);

    int vocab_size = 10;
    int embedding_dimension = 4;
    double dropout_p = 0.5;

    MyModelWithEmbeddingDropout model(vocab_size, embedding_dimension, dropout_p);
    std::cout << "Model with EmbeddingDropout: " << model << std::endl;

    // Example input: Batch of 2 sequences, each of length 3
    torch::Tensor input_ids = torch::randint(0, vocab_size, {2, 3}, torch::kLong);
    std::cout << "Input IDs (Batch=2, SeqLen=3):\n" << input_ids << std::endl;

    // --- Training mode ---
    model->train();
    torch::Tensor output_train = model->forward(input_ids);
    std::cout << "Model output (train):\n" << output_train << std::endl;
    // Inspect the intermediate embeddings (not directly accessible without modifying MyModel or hooking)
    // but the effect is that some (token, embedding_vector) pairs will be zeroed out.
    // For Batch=2, SeqLen=3, there are 2*3 = 6 tokens.
    // With p_drop=0.5, expect ~3 of these tokens' embeddings to be zeroed out.

    // --- Evaluation mode ---
    model->eval();
    torch::Tensor output_eval = model->forward(input_ids);
    std::cout << "Model output (eval):\n" << output_eval << std::endl;
    // In eval mode, EmbeddingDropout should be an identity operation.
    // The output should be consistent if called multiple times with the same input.


    // --- Test EmbeddingDropout directly with a pre-computed embedding tensor ---
    std::cout << "\n--- Direct EmbeddingDropout Test ---" << std::endl;
    EmbeddingDropout direct_embed_dropout(0.25);
    direct_embed_dropout->train();

    torch::Tensor example_embeddings = torch::randn({2, 3, embedding_dimension}); // (B, S, E)
    std::cout << "Original example_embeddings (sum per token):\n"
              << example_embeddings.sum(-1) << std::endl; // Sum over embedding_dim

    torch::Tensor dropped_embeddings = direct_embed_dropout->forward(example_embeddings);
    std::cout << "After EmbeddingDropout (sum per token):\n"
              << dropped_embeddings.sum(-1) << std::endl;
    // Expect some rows in the (B, S) sum view to be zero, others scaled.
}

// int main() {
//    run_embedding_dropout_example();
//    return 0;
// }
*/


namespace xt::dropouts
{
    torch::Tensor embedding_dropout(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto EmbeddingDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::embedding_dropout(torch::zeros(10));
    }
}
