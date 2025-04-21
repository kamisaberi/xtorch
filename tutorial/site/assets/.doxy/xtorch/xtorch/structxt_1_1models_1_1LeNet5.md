

# Struct xt::models::LeNet5



[**ClassList**](annotated.md) **>** [**xt**](namespacext.md) **>** [**models**](namespacext_1_1models.md) **>** [**LeNet5**](structxt_1_1models_1_1LeNet5.md)








Inherits the following classes: [xt::models::BaseModel](classxt_1_1models_1_1BaseModel.md)






















































## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**LeNet5**](#function-lenet5-12) (int num\_classes, int in\_channels=1) <br> |
|   | [**LeNet5**](#function-lenet5-22) (int num\_classes, int in\_channels, std::vector&lt; int64\_t &gt; input\_shape) <br> |
| virtual torch::Tensor | [**forward**](#function-forward) (torch::Tensor x) override const<br> |


## Public Functions inherited from xt::models::BaseModel

See [xt::models::BaseModel](classxt_1_1models_1_1BaseModel.md)

| Type | Name |
| ---: | :--- |
|   | [**BaseModel**](classxt_1_1models_1_1BaseModel.md#function-basemodel) () <br> |
| virtual torch::Tensor | [**forward**](classxt_1_1models_1_1BaseModel.md#function-forward) (torch::Tensor input) const = 0<br> |














## Protected Attributes

| Type | Name |
| ---: | :--- |
|  torch::nn::Linear | [**fc1**](#variable-fc1)   = `nullptr`<br> |
|  torch::nn::Linear | [**fc2**](#variable-fc2)   = `nullptr`<br> |
|  torch::nn::Linear | [**fc3**](#variable-fc3)   = `nullptr`<br> |
|  torch::nn::Sequential | [**layer1**](#variable-layer1)   = `nullptr`<br> |
|  torch::nn::Sequential | [**layer2**](#variable-layer2)   = `nullptr`<br> |








































## Public Functions Documentation




### function LeNet5 [1/2]

```C++
xt::models::LeNet5::LeNet5 (
    int num_classes,
    int in_channels=1
) 
```




<hr>



### function LeNet5 [2/2]

```C++
xt::models::LeNet5::LeNet5 (
    int num_classes,
    int in_channels,
    std::vector< int64_t > input_shape
) 
```




<hr>



### function forward 

```C++
virtual torch::Tensor xt::models::LeNet5::forward (
    torch::Tensor x
) override const
```



Implements [*xt::models::BaseModel::forward*](classxt_1_1models_1_1BaseModel.md#function-forward)


<hr>
## Protected Attributes Documentation




### variable fc1 

```C++
torch::nn::Linear xt::models::LeNet5::fc1;
```




<hr>



### variable fc2 

```C++
torch::nn::Linear xt::models::LeNet5::fc2;
```




<hr>



### variable fc3 

```C++
torch::nn::Linear xt::models::LeNet5::fc3;
```




<hr>



### variable layer1 

```C++
torch::nn::Sequential xt::models::LeNet5::layer1;
```




<hr>



### variable layer2 

```C++
torch::nn::Sequential xt::models::LeNet5::layer2;
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/kami/Documents/cpp/models/include/models/cnn/lenet/lenet5.h`

