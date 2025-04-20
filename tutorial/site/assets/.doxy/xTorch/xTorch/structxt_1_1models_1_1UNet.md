

# Struct xt::models::UNet



[**ClassList**](annotated.md) **>** [**xt**](namespacext.md) **>** [**models**](namespacext_1_1models.md) **>** [**UNet**](structxt_1_1models_1_1UNet.md)








Inherits the following classes: [xt::models::BaseModel](classxt_1_1models_1_1BaseModel.md)






















































## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**UNet**](#function-unet) (int num\_classes, int in\_channels=1) <br> |
| virtual torch::Tensor | [**forward**](#function-forward) (torch::Tensor input) override const<br> |


## Public Functions inherited from xt::models::BaseModel

See [xt::models::BaseModel](classxt_1_1models_1_1BaseModel.md)

| Type | Name |
| ---: | :--- |
|   | [**BaseModel**](classxt_1_1models_1_1BaseModel.md#function-basemodel) () <br> |
| virtual torch::Tensor | [**forward**](classxt_1_1models_1_1BaseModel.md#function-forward) (torch::Tensor input) const = 0<br> |






















































## Public Functions Documentation




### function UNet 

```C++
xt::models::UNet::UNet (
    int num_classes,
    int in_channels=1
) 
```




<hr>



### function forward 

```C++
virtual torch::Tensor xt::models::UNet::forward (
    torch::Tensor input
) override const
```



Implements [*xt::models::BaseModel::forward*](classxt_1_1models_1_1BaseModel.md#function-forward)


<hr>

------------------------------
The documentation for this class was generated from the following file `/home/kami/Documents/cpp/models/include/models/cnn/unet/unet.h`

