# Text Transforms

Text transforms are essential for preparing raw text data for use in neural networks. Unlike images, which are already numerical, text is a sequence of characters that must be converted into a structured, numerical format—typically a tensor of integer IDs.

The two primary steps in any NLP data pipeline are:
1.  **Tokenization**: The process of breaking a raw string of text into smaller pieces called "tokens". These can be words, subwords, or characters.
2.  **Numericalization**: The process of converting each token into a unique integer ID based on a pre-defined "vocabulary".

xTorch provides a suite of transforms to handle these steps, from powerful tokenizers to utilities for managing sequence length.

All text transforms are located under the `xt::transforms::text` namespace and can be found in the `<xtorch/transforms/text/>` header directory.

## General Usage

Text transforms are used within a `Compose` pipeline, which is then passed to an NLP `Dataset`. A key difference from other modalities is the reliance on a **vocabulary**, a mapping from tokens to integers. This vocabulary is often built from the training corpus or loaded from a pre-trained model's files.

```cpp
#include <xtorch/xtorch.h>
#include <iostream>

int main() {
    // In a real application, you would load a vocabulary from a file.
    // For example, for a BERT model, this would be a 'vocab.txt' file.
    // std::string vocab_path = "path/to/bert/vocab.txt";

    // 1. Define a pipeline of text transformations.
    auto text_pipeline = std::make_unique<xt::transforms::Compose>(
        // Tokenize the input string using the BERT WordPiece tokenizer
        std::make_shared<xt::transforms::text::BertTokenizer>(/*vocab_path=*/vocab_path),
        // Truncate sequences to a maximum length of 512 tokens
        std::make_shared<xt::transforms::text::Truncate>(512),
        // Add special tokens like [CLS] and [SEP]
        std::make_shared<xt::transforms::text::AddToken>("[CLS]", /*at_beginning=*/true),
        std::make_shared<xt::transforms::text::AddToken>("[SEP]", /*at_beginning=*/false)
        // Note: Padding is often handled by the DataLoader's collate function,
        // but can also be a transform.
    );

    // 2. Pass the pipeline to an NLP Dataset
    auto dataset = xt::datasets::IMDB(
        "./data",
        xt::datasets::DataMode::TRAIN,
        /*download=*/true,
        std::move(text_pipeline)
    );

    // 3. The DataLoader will now yield batches of tokenized and numericalized text
    xt::dataloaders::ExtendedDataLoader data_loader(dataset, 16);
    // ...
}
```

---

## Available Transforms by Category

### Tokenizers

These modules are responsible for the first step: converting a string into a sequence of tokens.

| Transform | Description | Header File |
|---|---|---|
| `BertTokenizer` | Implements the WordPiece tokenization algorithm used by BERT. It requires a `vocab.txt` file. | `bert_tokenizer.h` |
| `SentencePieceTokenizer`| A tokenizer that uses the SentencePiece library, common in models like XLNet and T5. It requires a `spm.model` file. | `sentence_piece_tokenizer.h` |

### Vocabulary and Numericalization

These transforms handle the conversion between tokens and integer IDs.

| Transform | Description | Header File |
|---|---|---|
| `VocabTransform` | A transform that takes a vocabulary object and converts a sequence of string tokens into a sequence of integer IDs. | `vocab_transform.h` |
| `StrToIntTransform`| A lower-level transform for converting strings to integers, often used internally by `VocabTransform`. | `str_to_int_transform.h` |

### Sequence Utilities

These transforms are used to format the token sequences to meet the model's requirements.

| Transform | Description | Header File |
|---|---|---|
| `PadTransform` | Pads a sequence to a specified length with a given padding token ID. | `pad_transform.h` |
| `Truncate` | Truncates a sequence to a maximum specified length. | `truncate.h` |
| `AddToken` | Adds a special token (e.g., `[CLS]`, `[SEP]`, `[EOS]`) to the beginning or end of a sequence. | `add_token.h` |

### Data Augmentation

These transforms modify the input text to create new training samples, which can help improve model robustness.

| Transform | Description | Header File |
|---|---|---|
| `SynonymReplacement`| Randomly replaces words in a sentence with their synonyms. | `synonym_replacement.h` |
| `BackTranslation` | Augments text by translating it to another language and then translating it back to the original language. (Note: May require an external translation API). | `back_translation.h` |
| `TextStyleTransfer`| A transform for altering the style of the text. | `text_style_transfer.h` |
