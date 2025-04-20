

# Class xt::data::datasets::MNISTBase



[**ClassList**](annotated.md) **>** [**xt**](namespacext.md) **>** [**data**](namespacext_1_1data.md) **>** [**datasets**](namespacext_1_1data_1_1datasets.md) **>** [**MNISTBase**](classxt_1_1data_1_1datasets_1_1MNISTBase.md)








Inherits the following classes: [xt::data::datasets::BaseDataset](classxt_1_1data_1_1datasets_1_1BaseDataset.md)


Inherited by the following classes: [xt::data::datasets::EMNIST](classxt_1_1data_1_1datasets_1_1EMNIST.md),  [xt::data::datasets::FashionMNIST](classxt_1_1data_1_1datasets_1_1FashionMNIST.md),  [xt::data::datasets::KMNIST](classxt_1_1data_1_1datasets_1_1KMNIST.md),  [xt::data::datasets::MNIST](classxt_1_1data_1_1datasets_1_1MNIST.md),  [xt::data::datasets::QMNIST](classxt_1_1data_1_1datasets_1_1QMNIST.md)














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
|   | [**MNISTBase**](#function-mnistbase-14) (const std::string & root) <br> |
|   | [**MNISTBase**](#function-mnistbase-24) (const std::string & root, DataMode mode) <br> |
|   | [**MNISTBase**](#function-mnistbase-34) (const std::string & root, DataMode mode, bool download) <br> |
|   | [**MNISTBase**](#function-mnistbase-44) (const std::string & root, DataMode mode, bool download, vector&lt; std::function&lt; torch::Tensor(torch::Tensor)&gt; &gt; transforms) <br> |
|  void | [**read\_images**](#function-read_images) (const std::string & file\_path, int num\_images) <br> |
|  void | [**read\_labels**](#function-read_labels) (const std::string & file\_path, int num\_labels) <br> |


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




### function MNISTBase [1/4]

```C++
explicit xt::data::datasets::MNISTBase::MNISTBase (
    const std::string & root
) 
```




<hr>



### function MNISTBase [2/4]

```C++
xt::data::datasets::MNISTBase::MNISTBase (
    const std::string & root,
    DataMode mode
) 
```




<hr>



### function MNISTBase [3/4]

```C++
xt::data::datasets::MNISTBase::MNISTBase (
    const std::string & root,
    DataMode mode,
    bool download
) 
```




<hr>



### function MNISTBase [4/4]

```C++
xt::data::datasets::MNISTBase::MNISTBase (
    const std::string & root,
    DataMode mode,
    bool download,
    vector< std::function< torch::Tensor(torch::Tensor)> > transforms
) 
```




<hr>



### function read\_images 

```C++
void xt::data::datasets::MNISTBase::read_images (
    const std::string & file_path,
    int num_images
) 
```




<hr>



### function read\_labels 

```C++
void xt::data::datasets::MNISTBase::read_labels (
    const std::string & file_path,
    int num_labels
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/kami/Documents/cpp/models/include/datasets/image-classification/mnist.h`

