

# Class xt::data::datasets::CSVDataset



[**ClassList**](annotated.md) **>** [**xt**](namespacext.md) **>** [**data**](namespacext_1_1data.md) **>** [**datasets**](namespacext_1_1data_1_1datasets.md) **>** [**CSVDataset**](classxt_1_1data_1_1datasets_1_1CSVDataset.md)








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
|   | [**CSVDataset**](#function-csvdataset-111) (const std::string & file\_path) <br> |
|   | [**CSVDataset**](#function-csvdataset-211) (const std::string & file\_path, DataMode mode) <br> |
|   | [**CSVDataset**](#function-csvdataset-311) (const std::string & file\_path, DataMode mode, vector&lt; int &gt; x\_indices, int y\_index) <br> |
|   | [**CSVDataset**](#function-csvdataset-411) (const std::string & file\_path, DataMode mode, vector&lt; int &gt; x\_indices, vector&lt; int &gt; y\_indices) <br> |
|   | [**CSVDataset**](#function-csvdataset-511) (const std::string & file\_path, DataMode mode, vector&lt; string &gt; x\_titles, string y\_title) <br> |
|   | [**CSVDataset**](#function-csvdataset-611) (const std::string & file\_path, DataMode mode, vector&lt; string &gt; x\_titles, vector&lt; string &gt; y\_titles) <br> |
|   | [**CSVDataset**](#function-csvdataset-711) (const std::string & file\_path, DataMode mode, TransformType transforms) <br> |
|   | [**CSVDataset**](#function-csvdataset-811) (const std::string & file\_path, DataMode mode, vector&lt; int &gt; x\_indices, int y\_index, TransformType transforms) <br> |
|   | [**CSVDataset**](#function-csvdataset-911) (const std::string & file\_path, DataMode mode, vector&lt; int &gt; x\_indices, vector&lt; int &gt; y\_indices, TransformType transforms) <br> |
|   | [**CSVDataset**](#function-csvdataset-1011) (const std::string & file\_path, DataMode mode, vector&lt; string &gt; x\_titles, string y\_title, TransformType transforms) <br> |
|   | [**CSVDataset**](#function-csvdataset-1111) (const std::string & file\_path, DataMode mode, vector&lt; string &gt; x\_titles, vector&lt; string &gt; y\_titles, TransformType transforms) <br> |


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




### function CSVDataset [1/11]

```C++
xt::data::datasets::CSVDataset::CSVDataset (
    const std::string & file_path
) 
```




<hr>



### function CSVDataset [2/11]

```C++
xt::data::datasets::CSVDataset::CSVDataset (
    const std::string & file_path,
    DataMode mode
) 
```




<hr>



### function CSVDataset [3/11]

```C++
xt::data::datasets::CSVDataset::CSVDataset (
    const std::string & file_path,
    DataMode mode,
    vector< int > x_indices,
    int y_index
) 
```




<hr>



### function CSVDataset [4/11]

```C++
xt::data::datasets::CSVDataset::CSVDataset (
    const std::string & file_path,
    DataMode mode,
    vector< int > x_indices,
    vector< int > y_indices
) 
```




<hr>



### function CSVDataset [5/11]

```C++
xt::data::datasets::CSVDataset::CSVDataset (
    const std::string & file_path,
    DataMode mode,
    vector< string > x_titles,
    string y_title
) 
```




<hr>



### function CSVDataset [6/11]

```C++
xt::data::datasets::CSVDataset::CSVDataset (
    const std::string & file_path,
    DataMode mode,
    vector< string > x_titles,
    vector< string > y_titles
) 
```




<hr>



### function CSVDataset [7/11]

```C++
xt::data::datasets::CSVDataset::CSVDataset (
    const std::string & file_path,
    DataMode mode,
    TransformType transforms
) 
```




<hr>



### function CSVDataset [8/11]

```C++
xt::data::datasets::CSVDataset::CSVDataset (
    const std::string & file_path,
    DataMode mode,
    vector< int > x_indices,
    int y_index,
    TransformType transforms
) 
```




<hr>



### function CSVDataset [9/11]

```C++
xt::data::datasets::CSVDataset::CSVDataset (
    const std::string & file_path,
    DataMode mode,
    vector< int > x_indices,
    vector< int > y_indices,
    TransformType transforms
) 
```




<hr>



### function CSVDataset [10/11]

```C++
xt::data::datasets::CSVDataset::CSVDataset (
    const std::string & file_path,
    DataMode mode,
    vector< string > x_titles,
    string y_title,
    TransformType transforms
) 
```




<hr>



### function CSVDataset [11/11]

```C++
xt::data::datasets::CSVDataset::CSVDataset (
    const std::string & file_path,
    DataMode mode,
    vector< string > x_titles,
    vector< string > y_titles,
    TransformType transforms
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/kami/Documents/cpp/models/include/datasets/general/csv-dataset.h`

