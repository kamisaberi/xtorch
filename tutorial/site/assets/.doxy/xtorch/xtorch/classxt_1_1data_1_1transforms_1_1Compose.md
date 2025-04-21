

# Class xt::data::transforms::Compose



[**ClassList**](annotated.md) **>** [**xt**](namespacext.md) **>** [**data**](namespacext_1_1data.md) **>** [**transforms**](namespacext_1_1data_1_1transforms.md) **>** [**Compose**](classxt_1_1data_1_1transforms_1_1Compose.md)



_A class to compose multiple tensor transformations into a single callable pipeline._ [More...](#detailed-description)

* `#include <compose.h>`

















## Public Types

| Type | Name |
| ---: | :--- |
| typedef std::function&lt; torch::Tensor(torch::Tensor)&gt; | [**TransformFunc**](#typedef-transformfunc)  <br>_Alias for a transformation function that takes a tensor and returns a tensor._  |




















## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**Compose**](#function-compose-12) () <br>_Default constructor, initializing an empty transformation pipeline._  |
|   | [**Compose**](#function-compose-22) (std::vector&lt; [**TransformFunc**](classxt_1_1data_1_1transforms_1_1Compose.md#typedef-transformfunc) &gt; transforms) <br>_Constructs a_ [_**Compose**_](classxt_1_1data_1_1transforms_1_1Compose.md) _object with a vector of transformation functions._ |
|  torch::Tensor | [**operator()**](#function-operator()) (torch::Tensor input) const<br>_Applies the sequence of transformations to the input tensor._  |




























## Detailed Description


The [**Compose**](classxt_1_1data_1_1transforms_1_1Compose.md) class allows chaining of multiple transformation functions, each operating on a `torch::Tensor`, into a single operation. It is designed to facilitate preprocessing or augmentation of tensor data (e.g., images) by applying a sequence of transforms in the order they are provided. The transformations are stored as a vector of function objects and applied via the call operator. 


    
## Public Types Documentation




### typedef TransformFunc 

_Alias for a transformation function that takes a tensor and returns a tensor._ 
```C++
using xt::data::transforms::Compose::TransformFunc =  std::function<torch::Tensor(torch::Tensor)>;
```



This type alias defines a function signature for transformations that operate on `torch::Tensor` objects, enabling flexible composition of operations. 


        

<hr>
## Public Functions Documentation




### function Compose [1/2]

_Default constructor, initializing an empty transformation pipeline._ 
```C++
xt::data::transforms::Compose::Compose () 
```



Creates a [**Compose**](classxt_1_1data_1_1transforms_1_1Compose.md) object with no transformations, allowing subsequent addition of transforms if needed. 


        

<hr>



### function Compose [2/2]

_Constructs a_ [_**Compose**_](classxt_1_1data_1_1transforms_1_1Compose.md) _object with a vector of transformation functions._
```C++
xt::data::transforms::Compose::Compose (
    std::vector< TransformFunc > transforms
) 
```





**Parameters:**


* `transforms` A vector of [**TransformFunc**](classxt_1_1data_1_1transforms_1_1Compose.md#typedef-transformfunc) objects specifying the sequence of transformations.

Initializes the [**Compose**](classxt_1_1data_1_1transforms_1_1Compose.md) object with a predefined set of transformations to be applied in order. 


        

<hr>



### function operator() 

_Applies the sequence of transformations to the input tensor._ 
```C++
torch::Tensor xt::data::transforms::Compose::operator() (
    torch::Tensor input
) const
```





**Parameters:**


* `input` The input tensor to be transformed. 



**Returns:**

A tensor resulting from applying all transformations in sequence.


This operator applies each transformation in the `transforms` vector to the input tensor, passing the output of one transformation as the input to the next, and returns the final result. 


        

<hr>

------------------------------
The documentation for this class was generated from the following file `/home/kami/Documents/cpp/models/include/transforms/compose.h`

