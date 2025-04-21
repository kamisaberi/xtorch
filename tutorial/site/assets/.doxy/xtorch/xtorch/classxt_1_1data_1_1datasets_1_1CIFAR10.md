

# Class xt::data::datasets::CIFAR10



[**ClassList**](annotated.md) **>** [**xt**](namespacext.md) **>** [**data**](namespacext_1_1data.md) **>** [**datasets**](namespacext_1_1data_1_1datasets.md) **>** [**CIFAR10**](classxt_1_1data_1_1datasets_1_1CIFAR10.md)








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
|   | [**CIFAR10**](#function-cifar10-14) (const std::string & root) <br> |
|   | [**CIFAR10**](#function-cifar10-24) (const std::string & root, DataMode mode) <br> |
|   | [**CIFAR10**](#function-cifar10-34) (const std::string & root, DataMode mode, bool download) <br> |
|   | [**CIFAR10**](#function-cifar10-44) (const std::string & root, DataMode mode, bool download, TransformType transforms) <br> |
|  torch::data::Example | [**get**](#function-get) (size\_t index) override<br> |
|  torch::optional&lt; size\_t &gt; | [**size**](#function-size) () override const<br> |


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




### function CIFAR10 [1/4]

```C++
explicit xt::data::datasets::CIFAR10::CIFAR10 (
    const std::string & root
) 
```




<hr>



### function CIFAR10 [2/4]

```C++
xt::data::datasets::CIFAR10::CIFAR10 (
    const std::string & root,
    DataMode mode
) 
```




<hr>



### function CIFAR10 [3/4]

```C++
xt::data::datasets::CIFAR10::CIFAR10 (
    const std::string & root,
    DataMode mode,
    bool download
) 
```




<hr>



### function CIFAR10 [4/4]

```C++
xt::data::datasets::CIFAR10::CIFAR10 (
    const std::string & root,
    DataMode mode,
    bool download,
    TransformType transforms
) 
```




<hr>



### function get 

```C++
torch::data::Example xt::data::datasets::CIFAR10::get (
    size_t index
) override
```




<hr>



### function size 

```C++
torch::optional< size_t > xt::data::datasets::CIFAR10::size () override const
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/kami/Documents/cpp/models/include/datasets/image-classification/cifar.h`

