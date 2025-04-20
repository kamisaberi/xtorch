

# Class xt::data::datasets::WNLI



[**ClassList**](annotated.md) **>** [**xt**](namespacext.md) **>** [**data**](namespacext_1_1data.md) **>** [**datasets**](namespacext_1_1data_1_1datasets.md) **>** [**WNLI**](classxt_1_1data_1_1datasets_1_1WNLI.md)








Inherits the following classes: [xt::data::datasets::BaseDataset](classxt_1_1data_1_1datasets_1_1BaseDataset.md)
















## Public Types inherited from xt::data::datasets::BaseDataset

See [xt::data::datasets::BaseDataset](classxt_1_1data_1_1datasets_1_1BaseDataset.md)

| Type | Name |
| ---: | :--- |
| typedef vector&lt; std::function&lt; torch::Tensor(torch::Tensor)&gt; &gt; | [**TransformType**](classxt_1_1data_1_1datasets_1_1BaseDataset.md#typedef-transformtype)  <br> |








## Public Attributes inherited from xt::data::datasets::BaseDataset

See [xt::data::datasets::BaseDataset](classxt_1_1data_1_1datasets_1_1BaseDataset.md)

| Type | Name |
| ---: | :--- |
|  [**xt::data::transforms::Compose**](classxt_1_1data_1_1transforms_1_1Compose.md) | [**compose**](classxt_1_1data_1_1datasets_1_1BaseDataset.md#variable-compose)  <br> |
|  std::vector&lt; torch::Tensor &gt; | [**data**](classxt_1_1data_1_1datasets_1_1BaseDataset.md#variable-data)  <br> |
|  fs::path | [**dataset\_path**](classxt_1_1data_1_1datasets_1_1BaseDataset.md#variable-dataset_path)  <br> |
|  bool | [**download**](classxt_1_1data_1_1datasets_1_1BaseDataset.md#variable-download)   = `false`<br> |
|  std::vector&lt; uint8\_t &gt; | [**labels**](classxt_1_1data_1_1datasets_1_1BaseDataset.md#variable-labels)  <br> |
|  DataMode | [**mode**](classxt_1_1data_1_1datasets_1_1BaseDataset.md#variable-mode)   = `DataMode::TRAIN`<br> |
|  fs::path | [**root**](classxt_1_1data_1_1datasets_1_1BaseDataset.md#variable-root)  <br> |
|  vector&lt; std::function&lt; torch::Tensor(torch::Tensor)&gt; &gt; | [**transforms**](classxt_1_1data_1_1datasets_1_1BaseDataset.md#variable-transforms)   = `{}`<br> |






























## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**WNLI**](#function-wnli-14) (const std::string & root) <br> |
|   | [**WNLI**](#function-wnli-24) (const std::string & root, DataMode mode) <br> |
|   | [**WNLI**](#function-wnli-34) (const std::string & root, DataMode mode, bool download) <br> |
|   | [**WNLI**](#function-wnli-44) (const std::string & root, DataMode mode, bool download, TransformType transforms) <br> |


## Public Functions inherited from xt::data::datasets::BaseDataset

See [xt::data::datasets::BaseDataset](classxt_1_1data_1_1datasets_1_1BaseDataset.md)

| Type | Name |
| ---: | :--- |
|   | [**BaseDataset**](classxt_1_1data_1_1datasets_1_1BaseDataset.md#function-basedataset-14) (const std::string & root) <br> |
|   | [**BaseDataset**](classxt_1_1data_1_1datasets_1_1BaseDataset.md#function-basedataset-24) (const std::string & root, DataMode mode) <br> |
|   | [**BaseDataset**](classxt_1_1data_1_1datasets_1_1BaseDataset.md#function-basedataset-34) (const std::string & root, DataMode mode, bool download) <br> |
|   | [**BaseDataset**](classxt_1_1data_1_1datasets_1_1BaseDataset.md#function-basedataset-44) (const std::string & root, DataMode mode, bool download, vector&lt; std::function&lt; torch::Tensor(torch::Tensor)&gt; &gt; transforms) <br> |
|  torch::data::Example | [**get**](classxt_1_1data_1_1datasets_1_1BaseDataset.md#function-get) (size\_t index) override<br> |
|  torch::optional&lt; size\_t &gt; | [**size**](classxt_1_1data_1_1datasets_1_1BaseDataset.md#function-size) () override const<br> |






















































## Public Functions Documentation




### function WNLI [1/4]

```C++
explicit xt::data::datasets::WNLI::WNLI (
    const std::string & root
) 
```




<hr>



### function WNLI [2/4]

```C++
xt::data::datasets::WNLI::WNLI (
    const std::string & root,
    DataMode mode
) 
```




<hr>



### function WNLI [3/4]

```C++
xt::data::datasets::WNLI::WNLI (
    const std::string & root,
    DataMode mode,
    bool download
) 
```




<hr>



### function WNLI [4/4]

```C++
xt::data::datasets::WNLI::WNLI (
    const std::string & root,
    DataMode mode,
    bool download,
    TransformType transforms
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/kami/Documents/cpp/models/include/datasets/natural-language-inference/wnli.h`

