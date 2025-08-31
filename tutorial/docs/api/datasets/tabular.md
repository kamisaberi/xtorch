# Tabular Datasets

Tabular datasets are one of the most common forms of data, organized in a table-like structure with rows representing individual samples and columns representing features. These datasets are foundational to "classic" machine learning tasks like classification and regression.

xTorch provides a collection of well-known, small-scale tabular datasets, which are extremely useful for educational purposes, debugging models, and experimenting with algorithms on well-understood data.

All tabular datasets are located under the `xt::datasets` namespace and can be found in the `<xtorch/datasets/tabular_data/>` header directory.

## General Usage

Tabular datasets in xTorch are typically pre-processed and do not require complex transformations like image or audio data. The features are often numerical, and the targets are provided as class indices or continuous values.

The workflow is straightforward: instantiate the dataset and pass it to a `DataLoader`.

```cpp
#include <xtorch/xtorch.h>
#include <iostream>

int main() {
    // 1. Instantiate a dataset for the Iris flower classification task.
    // These datasets typically only require a root path and whether to download.
    auto dataset = xt::datasets::Iris(
        "./data",
        xt::datasets::DataMode::TRAIN, // Mode might not be used if the dataset isn't split
        /*download=*/true
    );

    std::cout << "Iris dataset size: " << *dataset.size() << std::endl;
    // Iris has 4 features per sample
    std::cout << "Sample feature shape: " << dataset.get(0).data.sizes() << std::endl;

    // 2. Pass the dataset to a DataLoader
    xt::dataloaders::ExtendedDataLoader data_loader(dataset, 16, true);

    // The data loader is now ready for use in a training loop
    for (auto& batch : data_loader) {
        auto features = batch.first;  // Tensor of shape
        auto labels = batch.second;   // Tensor of shape
        // ... training step with a simple Multi-Layer Perceptron (MLP) ...
    }
}
```

!!! info "Simplified Constructors"
Many tabular datasets do not have official train/test splits and are not large enough to warrant complex transforms. As such, their constructors are often simpler than those for other domains, sometimes only requiring a `root` path.

---

## Available Datasets by Task

### Binary Classification

These datasets involve predicting one of two possible outcomes.

| Dataset Class | Description | Header File |
|---|---|---|
| `AdultCensusIncome` | Predict whether income exceeds $50K/yr based on census data. | `binary_classification/adult_census_income.h` |
| `BanknoteAuthentication`| Predict whether a banknote is genuine or forged from image features. | `binary_classification/banknote_authentication.h` |
| `BreastCancerWisconsin`| Predict whether a breast mass is benign or malignant from digitized image features. | `binary_classification/breast_cancer_wisconsin.h` |
| `HabermansSurvival`| Predict the survival status of patients who had undergone surgery for breast cancer. | `binary_classification/habermans_survival.h` |
| `Ionosphere` | Classify radar returns from the ionosphere as "good" or "bad". | `binary_classification/ionosphere.h` |
| `Mushroom` | Predict whether a mushroom is edible or poisonous based on its characteristics. | `binary_classification/mushroom_dataset.h` |
| `PimaIndiansDiabetes`| Predict the onset of diabetes based on diagnostic measures. | `binary_classification/pima_indians_diabetes.h` |
| `SonarMinesVsRocks` | Discriminate between sonar signals bounced off a metal cylinder and those bounced off a rock. | `binary_classification/sonar_mines_vs_rocks.h` |
| `Titanic` | Predict survival on the Titanic. | `binary_classification/titanic_dataset.h` |

### Multi-Class Classification

These datasets involve predicting one of more than two possible outcomes.

| Dataset Class | Description | Header File |
|---|---|---|
| `Iris` | The classic dataset for classifying iris flowers into one of three species. | `classification/iris.h` |
| `CarEvaluation` | Evaluate the acceptability of a car based on six input attributes. | `classification/car_evaluation.h` |
| `Ecoli` | Classify the localization site of proteins in E. coli bacteria. | `classification/ecoli.h` |
| `GlassIdentification`| Classify types of glass based on their chemical composition. | `classification/glass_identification.h`|
| `PalmerPenguin` | A modern alternative to Iris for data exploration and classification. | `classification/palmer_penguin.h` |
| `VertebralColumn`| Classify patients into 3 classes (normal, disk hernia, spondylolisthesis) based on orthopedic features. | `classification/vertebral_column.h`|
| `WheatSeeds` | Classify kernels belonging to three different varieties of wheat. | `classification/wheat_seeds.h` |
| `Wine` | Classify wines into one of three cultivars using chemical analysis. | `classification/wine_dataset.h` |
| `Yeast` | Predict the cellular localization sites of proteins in yeast. | `classification/yeast.h` |
| `Zoo` | Classify animals into one of seven types based on their attributes. | `classification/zoo_dataset.h` |

### Regression

These datasets involve predicting a continuous numerical value.

| Dataset Class | Description | Header File |
|---|---|---|
| `BostonHousing` | Predict the median value of owner-occupied homes in Boston suburbs. | `regression/boston_housing.h` |
| `Abalone` | Predict the age of abalone from physical measurements. | `regression_classification/abalone.h`|
