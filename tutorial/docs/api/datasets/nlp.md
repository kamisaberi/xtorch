# Natural Language Processing Datasets

xTorch includes a comprehensive suite of built-in dataset handlers for a wide variety of Natural Language Processing (NLP) tasks. These handlers manage the downloading and parsing of standard text corpora, allowing you to focus on model development.

All NLP datasets are located under the `xt::datasets` namespace and can be found in the `<xtorch/datasets/natural_language_processing/>` header directory.

## General Usage

Working with NLP datasets requires an additional preprocessing step compared to other modalities: **tokenization and numericalization**. Raw text must be converted into a sequence of integer IDs that can be fed into a model. This is typically handled by a text `Transform`.

The general workflow is:
1.  Build or load a vocabulary.
2.  Create a text transformation pipeline for tokenizing and numericalizing the text.
3.  Instantiate the desired dataset with this pipeline.
4.  Pass the dataset to a `DataLoader`.

```cpp
#include <xtorch/xtorch.h>
#include <iostream>

int main() {
    // 1. & 2. Define a text transformation pipeline
    // This is a simplified example. In a real scenario, you would build a vocabulary
    // from the training data and use it to convert tokens to integers.
    // xTorch's text transforms help with this process.
    // For now, let's assume we have a simple tokenizer.
    // auto text_transforms = std::make_unique<xt::transforms::text::...>();

    // 3. Instantiate a dataset for the IMDB movie review task
    auto dataset = xt::datasets::IMDB(
        "./data",
        xt::datasets::DataMode::TRAIN,
        /*download=*/true
        // std::move(text_transforms) // Pass transforms here
    );

    std::cout << "IMDB dataset size: " << *dataset.size() << std::endl;

    // 4. Pass the dataset to a DataLoader
    // Note: For NLP, you often need a custom collate function to handle padding
    // of sequences with different lengths.
    xt::dataloaders::ExtendedDataLoader data_loader(dataset, 32, true);

    // The data loader is now ready for use in a training loop
    for (auto& batch : data_loader) {
        auto numericalized_text = batch.first;
        auto labels = batch.second;
        // ... training step with an RNN or Transformer ...
    }
}
```

!!! info "Standard Dataset Constructors"
Most dataset constructors follow a standard pattern:
`DatasetName(const std::string& root, DataMode mode, bool download, TextTransformPtr transforms)`
- `root`: The directory where the data is stored or will be downloaded.
- `mode`: `DataMode::TRAIN`, `DataMode::TEST`, or `DataMode::VALIDATION`.
- `download`: If `true`, the dataset will be downloaded if not found in the root directory.
- `transforms`: A `unique_ptr` to a text transform pipeline.

---

## Available Datasets by Task

### Text Classification

This is the task of assigning a category or label to a piece of text.

| Dataset Class | Description | Header File |
|---|---|---|
| `IMDB` | A large dataset of movie reviews for binary sentiment classification. | `text_classification/imdb.h` |
| `AG_NEWS` | A dataset of news articles from 4 categories. | `text_classification/ag_news.h` |
| `SST2` | Stanford Sentiment Treebank, for sentiment analysis. | `text_classification/sst2.h` |
| `YelpReviewPolarity`| A large dataset of Yelp reviews for binary sentiment analysis. | `text_classification/yelp_review_polarity.h` |
| `AmazonReviewPolarity`| A dataset of Amazon product reviews for binary sentiment analysis. | `text_classification/amazon_review_polarity.h` |
| `DBpedia` | A large-scale, multi-class text classification dataset with 14 classes. | `text_classification/db_pedia.h` |
| `SNLI` | Stanford Natural Language Inference corpus. | `text_classification/snli.h` |
| `MNLI` | Multi-Genre Natural Language Inference corpus. | `text_classification/mnli.h` |

### Language Modeling

The task of predicting the next word in a sequence.

| Dataset Class | Description | Header File |
|---|---|---|
| `WikiText2` | A small, high-quality corpus from Wikipedia articles. | `language_modeling/wiki_text_2.h` |
| `WikiText103` | A larger version of the WikiText corpus. | `language_modeling/wiki_text103.h` |
| `PennTreebank` | A classic, smaller dataset widely used for language modeling. | `language_modeling/penn_treebank.h` |

### Machine Translation

The task of translating a sequence of text from a source language to a target language.

| Dataset Class | Description | Header File |
|---|---|---|
| `WMT14` | The dataset from the 2014 Workshop on Machine Translation. | `machine_translation/wmt14.h` |
| `IWSLT2017` | The dataset from the International Workshop on Spoken Language Translation 2017. | `machine_translation/iwslt2017.h` |
| `Multi30k` | A dataset of 30,000 sentences with English, German, French, and Czech translations. | `machine_translation/multi30k.h` |

### Question Answering

The task of answering a question based on a given context passage.

| Dataset Class | Description | Header File |
|---|---|---|
| `SQuAD1` | The Stanford Question Answering Dataset, version 1.1. | `question_answering/squad1_0.h` |
| `SQuAD2` | Version 2.0 of SQuAD, which includes unanswerable questions. | `question_answering/squad2_0.h` |
| `NaturalQuestions`| A large-scale QA dataset from Google. | `question_answering/natural_questions.h`|
| `TriviaQA` | A challenging QA dataset with questions authored by trivia enthusiasts. | `question_answering/trivia_qa.h`|

### Text Summarization

The task of generating a short summary from a longer document.

| Dataset Class | Description | Header File |
|---|---|---|
| `CNNDailyMail` | A large dataset of news articles and their summaries. | `text_summarization/cnn_daily_mail.h`|
| `XSum` | A dataset for extreme summarization, with highly abstractive summaries. | `text_summarization/xsum.h` |

### Sequence Tagging
| Dataset Class | Description | Header File |
|---|---|---|
| `CoNLL2000Chunking`| A dataset for the task of chunking (shallow parsing). | `sequence_tagging/co_nll2000_chunking.h` |
| `UDPOS` | Universal Dependencies dataset for Part-of-Speech tagging. | `sequence_tagging/udpos.h` |

### Dialogue Generation
| Dataset Class | Description | Header File |
|---|---|---|
| `DailyDialog` | A high-quality, multi-turn dialogue dataset. | `dialogue_generation/daily_dialog.h` |
| `PersonaChat` | A conversational dataset where models are primed with a "persona". | `dialogue_generation/persona_chat.h` |

### Math Word Problems
| Dataset Class | Description | Header File |
|---|---|---|
| `GSM8K` | A dataset of 8,000 grade-school math word problems. | `math_word_problems/gsm8k.h` |
