

# Class xt::DataLoader

**template &lt;typename Dataset&gt;**



[**ClassList**](annotated.md) **>** [**xt**](namespacext.md) **>** [**DataLoader**](classxt_1_1DataLoader.md)








Inherits the following classes: torch::data::DataLoaderBase< Dataset, Dataset::BatchType, std::vector< size_t > >














## Public Types

| Type | Name |
| ---: | :--- |
| typedef torch::data::DataLoaderBase&lt; Dataset, BatchType, BatchRequestType &gt; | [**Base**](#typedef-base)  <br> |
| typedef std::vector&lt; size\_t &gt; | [**BatchRequestType**](#typedef-batchrequesttype)  <br> |
| typedef typename Dataset::BatchType | [**BatchType**](#typedef-batchtype)  <br> |




















## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**DataLoader**](#function-dataloader) (Dataset dataset, const torch::data::DataLoaderOptions & options, bool shuffle=false) <br> |
























## Protected Functions

| Type | Name |
| ---: | :--- |
|  std::optional&lt; BatchRequestType &gt; | [**get\_batch\_request**](#function-get_batch_request) () override<br> |
|  void | [**reset**](#function-reset) () override<br> |
|  void | [**reset\_indices**](#function-reset_indices) () <br> |




## Public Types Documentation




### typedef Base 

```C++
using xt::DataLoader< Dataset >::Base =  torch::data::DataLoaderBase<Dataset, BatchType, BatchRequestType>;
```




<hr>



### typedef BatchRequestType 

```C++
using xt::DataLoader< Dataset >::BatchRequestType =  std::vector<size_t>;
```




<hr>



### typedef BatchType 

```C++
using xt::DataLoader< Dataset >::BatchType =  typename Dataset::BatchType;
```




<hr>
## Public Functions Documentation




### function DataLoader 

```C++
xt::DataLoader::DataLoader (
    Dataset dataset,
    const torch::data::DataLoaderOptions & options,
    bool shuffle=false
) 
```




<hr>
## Protected Functions Documentation




### function get\_batch\_request 

```C++
std::optional< BatchRequestType > xt::DataLoader::get_batch_request () override
```




<hr>



### function reset 

```C++
void xt::DataLoader::reset () override
```




<hr>



### function reset\_indices 

```C++
void xt::DataLoader::reset_indices () 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/kami/Documents/cpp/models/include/data-loaders/data-loader.h`

