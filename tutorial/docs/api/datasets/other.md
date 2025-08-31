# Other Specialized Datasets

Beyond the primary domains of vision, language, and audio, xTorch also provides support for datasets from more specialized fields. This allows researchers and developers to work on a diverse range of tasks using the same consistent data loading interface.

This section covers datasets for:
-   Biomedical Data Analysis
-   Recommendation Systems
-   Reinforcement Learning

## General Usage

The usage pattern for these specialized datasets is similar to others: you instantiate the dataset class and pass it to a `DataLoader`. However, the structure of the data and the required preprocessing can be highly specific to the domain.

For example, reinforcement learning datasets might represent an entire environment, while recommendation system datasets often consist of user-item interaction pairs. Always refer to the specific dataset's documentation or header file for details on the data format.

```cpp
#include <xtorch/xtorch.h>
#include <iostream>

int main() {
    // Example: Loading the MovieLens dataset for a recommendation task
    auto dataset = xt::datasets::MovieLens(
        "./data",
        /*download=*/true
    );

    std::cout << "MovieLens dataset loaded." << std::endl;
    std::cout << "Number of ratings: " << *dataset.size() << std::endl;

    // The data loader will provide batches of user-item-rating triplets
    xt::dataloaders::ExtendedDataLoader data_loader(dataset, 256, true);

    for (auto& batch : data_loader) {
        // Data structure depends on the dataset, check the implementation.
        // For MovieLens, it could be a tensor of user IDs, item IDs, and ratings.
        // auto user_ids = batch.first;
        // auto item_ids = batch.second;
        // auto ratings = ...
    }
}
```

---

## Available Datasets by Domain

### Biomedical Data

These datasets are used for tasks like disease classification and genomic analysis.

| Dataset Class | Description | Header File |
|---|---|---|
| `ADNI` | The Alzheimer's Disease Neuroimaging Initiative dataset, used for classifying stages of Alzheimer's from medical imaging and clinical data. | `biomedical_data/alzheimers_classification/adni.h` |
| `TCGA` | The Cancer Genome Atlas (TCGA) dataset, containing genomic and clinical data for cancer research. | `biomedical_data/cancer_genomics_classification/tcga.h` |

### Recommendation Systems

These datasets contain user-item interaction data (e.g., ratings, reviews) and are used to train recommender models.

| Dataset Class | Description | Header File |
|---|---|---|
| `MovieLens` | A classic dataset family containing movie ratings from users. Different versions (e.g., 100K, 1M, 20M) are available. | `recommendation_systems/recommendation/movie_lens.h` |
| `AmazonProductReviews` | A large dataset of product reviews from Amazon, useful for training recommendation and sentiment analysis models. | `recommendation_systems/recommendation/amazon_product_reviews.h` |

### Reinforcement Learning

These are not traditional datasets but rather environments or collections of recorded experiences used to train reinforcement learning agents.

| Dataset Class | Description | Header File |
|---|---|---|
| `Atari2600ALE` | Provides an interface to the Arcade Learning Environment (ALE), allowing agents to be trained on a wide variety of Atari 2600 games. | `reinforcement_learning/reinforcement_learning/atari_2600_ale.h` |
| `MuJoCoGym` | Provides an interface to continuous control environments from OpenAI Gym powered by the MuJoCo physics engine (e.g., Hopper, Walker, Humanoid). | `reinforcement_learning/continuous_control/mu_jo_co_gym.h` |
