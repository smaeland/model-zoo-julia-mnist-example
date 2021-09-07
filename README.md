# Julia MNIST example

Simple MNIST classification example, based on the [Flux.jl tutorial](https://github.com/FluxML/model-zoo/blob/master/vision/conv_mnist/conv_mnist.jl).

### Setup

```julia
pkg> activate .
```

### Use the pretrained model

```julia
include("mnist.jl")
evaluate("cnn-model.bson")
```

### Train 

```julia
include("mnist.jl")
train(false, "cnn-model.bson")  # false -> cpu, true -> gpu
```
