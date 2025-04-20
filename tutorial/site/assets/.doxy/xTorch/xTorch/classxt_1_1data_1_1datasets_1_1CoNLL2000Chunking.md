

# Class xt::data::datasets::CoNLL2000Chunking



[**ClassList**](annotated.md) **>** [**xt**](namespacext.md) **>** [**data**](namespacext_1_1data.md) **>** [**datasets**](namespacext_1_1data_1_1datasets.md) **>** [**CoNLL2000Chunking**](classxt_1_1data_1_1datasets_1_1CoNLL2000Chunking.md)








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
|   | [**CoNLL2000Chunking**](#function-conll2000chunking-14) (const std::string & root) <br> |
|   | [**CoNLL2000Chunking**](#function-conll2000chunking-24) (const std::string & root, DataMode mode) <br> |
|   | [**CoNLL2000Chunking**](#function-conll2000chunking-34) (const std::string & root, DataMode mode, bool download) <br> |
|   | [**CoNLL2000Chunking**](#function-conll2000chunking-44) (const std::string & root, DataMode mode, bool download, TransformType transforms) <br> |


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




### function CoNLL2000Chunking [1/4]

```C++
explicit xt::data::datasets::CoNLL2000Chunking::CoNLL2000Chunking (
    const std::string & root
) 
```




<hr>



### function CoNLL2000Chunking [2/4]

```C++
xt::data::datasets::CoNLL2000Chunking::CoNLL2000Chunking (
    const std::string & root,
    DataMode mode
) 
```




<hr>



### function CoNLL2000Chunking [3/4]

```C++
xt::data::datasets::CoNLL2000Chunking::CoNLL2000Chunking (
    const std::string & root,
    DataMode mode,
    bool download
) 
```




<hr>



### function CoNLL2000Chunking [4/4]

```C++
xt::data::datasets::CoNLL2000Chunking::CoNLL2000Chunking (
    const std::string & root,
    DataMode mode,
    bool download,
    TransformType transforms
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/kami/Documents/cpp/models/include/datasets/sequence-tagging/con-ll-2000-chunking.h`

