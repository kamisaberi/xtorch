# Natural Language Processing (NLP) Models

The field of Natural Language Processing has been revolutionized by deep learning, particularly by the advent of the Transformer architecture. To empower developers to build state-of-the-art NLP applications, xTorch provides a comprehensive zoo of pre-built models, ranging from classic recurrent architectures to a wide variety of modern Transformers.

All NLP models are located under the `xt::models` namespace and their headers can be found in the `<xtorch/models/natural_language_processing/>` directory.

## General Usage

NLP models do not operate on raw text. Instead, they require the input text to be preprocessed into a numerical format. This typically involves:
1.  **Tokenization**: Breaking the text into sub-word units (tokens).
2.  **Numericalization**: Converting each token into a unique integer ID from a vocabulary.
3.  **Formatting**: Adding special tokens (like `[CLS]`, `[SEP]`), creating an attention mask to handle padding, and arranging the data into tensors.

The `forward` pass of a typical Transformer-based model takes these tensors as input.

```cpp
#include <xtorch/xtorch.h>
#include <iostream>

int main() {
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    // --- Model Configuration (Example for a BERT-like model) ---
    const int vocab_size = 30522;
    const int hidden_size = 768;
    const int num_attention_heads = 12;
    const int num_hidden_layers = 12;
    const int max_position_embeddings = 512;

    // --- Instantiate a BERT Model ---
    xt::models::BERT model(
        vocab_size,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        max_position_embeddings
    );
    model.to(device);
    model.train();

    std::cout << "BERT Model Instantiated." << std::endl;

    // --- Create Dummy Input Data ---
    const int batch_size = 8;
    const int sequence_length = 128;

    // Batch of token IDs
    auto input_ids = torch::randint(0, vocab_size, {batch_size, sequence_length}).to(device);
    // Attention mask to indicate which tokens are real vs. padding
    auto attention_mask = torch::ones({batch_size, sequence_length}).to(device);

    // --- Perform a Forward Pass ---
    auto output = model.forward(input_ids, attention_mask);
    // Output is often a tuple or struct containing last_hidden_state, pooler_output, etc.
    auto last_hidden_state = output.last_hidden_state;

    std::cout << "Output hidden state shape: " << last_hidden_state.sizes() << std::endl;
}
```

---

## Available Models by Family

### Transformer Architectures

This is the largest and most powerful family of models, forming the basis of modern NLP.

| Model Family | Description | Header File |
|---|---|---|
| `BERT` | Bidirectional Encoder Representations from Transformers, a powerful pre-trained encoder model. | `transformers/bert.h` |
| `RoBERTa` | A Robustly Optimized BERT Pretraining Approach. | `transformers/roberta.h` |
| `ALBERT` | A Lite BERT for Self-supervised Learning of Language Representations. | `transformers/albert.h` |
| `DistilBERT` | A smaller, faster, and lighter version of BERT, trained using knowledge distillation. | `transformers/distil_bert.h`|
| `ELECTRA` | A pre-training method that is more efficient than masked language modeling. | `transformers/electra.h` |
| `GPT` | Generative Pre-trained Transformer, a family of powerful auto-regressive language models. | `transformers/gpt.h` |
| `Llama` | A family of large language models released by Meta AI. | `transformers/llama.h` |
| `Mistral` | A family of high-performance large language models. | `transformers/mistral.h` |
| `Grok` | The open-source version of xAI's large language model. | `transformers/grok.h` |
| `DeepSeek` | A family of open-source LLMs from DeepSeek AI. | `transformers/deepseek.h` |
| `T5` | Text-To-Text Transfer Transformer, which frames all NLP tasks as a text-to-text problem. | `transformers/t5.h` |
| `BART` | A denoising autoencoder for pretraining sequence-to-sequence models. | `transformers/bart.h` |
| `XLNet` | A generalized autoregressive pretraining method that combines ideas from autoregressive and autoencoding models. | `transformers/xlnet.h` |
| `Longformer`| A Transformer variant with an attention mechanism that scales linearly with sequence length. | `transformers/long_former.h`|
| `Reformer` | An efficient Transformer variant that uses locality-sensitive hashing. | `transformers/reformer.h` |
| `BigBird` | A sparse attention mechanism that can handle long sequences. | `transformers/big_bird.h` |

### Recurrent Architectures (RNNs)

These models process sequences step-by-step and are foundational to sequence-based tasks.

| Model | Description | Header File |
|---|---|---|
| `Seq2Seq` | A standard sequence-to-sequence model using an Encoder-Decoder architecture with RNNs. | `rnn/seq2seq.h` |
| `AttentionBasedSeq2Seq` | An extension of Seq2Seq that incorporates an attention mechanism to improve performance. | `rnn/attention_based_seq2seq.h` |

### Other & Classic Models

These models are primarily used for learning static word embeddings.

| Model | Description | Header File |
|---|---|---|
| `Word2Vec` | A classic model that learns word associations from a large corpus of text. | `others/word2vec.h` |
| `GloVe` | Global Vectors for Word Representation, an unsupervised learning algorithm for obtaining vector representations for words. | `others/glove.h` |
| `FastText` | An extension of Word2Vec that learns vectors for n-grams of characters, allowing it to handle out-of-vocabulary words. | `others/fast_text.h` |
| `ELMo` | Embeddings from Language Models, a deep contextualized word representation. | `others/elmo.h` |
