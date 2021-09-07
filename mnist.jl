

using Flux
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser
using Flux: onehotbatch, onecold
using Flux.Losses: logitcrossentropy
import MLDatasets
using ProgressMeter: @showprogress
import BSON


"""
MNIST data from MLDatasets are already in the range [0, 1], but we need to add
the channel dimension
"""
function preprocess(input_data)

    img_size = size(input_data)[1:2]
    
    return reshape(input_data, img_size[1], img_size[2], 1, :)
end



function create_model(input_shape, num_classes)

    @info "input_shape:" input_shape

    #out_conv_size = (input_shape[1]÷4 - 3, input_shape[2]÷4 - 3, 16)

    model = Chain(
        Conv((3, 3), input_shape[end]=>6, relu),
        MaxPool((2, 2)),
        Conv((3, 3), 6=>16, relu),
        flatten,
        Dense(11*11*16, num_classes)
    )
    return model

    return Chain(
            Conv((3, 5), input_shape[end]=>6, relu),
            MaxPool((2, 2)),
            Conv((5, 5), 6=>16, relu),
            MaxPool((2, 2)),
            flatten,
            Dense(prod(out_conv_size), 120, relu), 
            Dense(120, 84, relu), 
            Dense(84, num_classes)
          )

end

loss(ŷ, y) = logitcrossentropy(ŷ, y)
round4(x) = round(x, digits=4)

function eval_loss_accuracy(loader, model, device)
    l = 0f0
    acc = 0
    ntot = 0
    for (x, y) in loader
        x, y = x |> device, y |> device
        ŷ = model(x)
        l += loss(ŷ, y) * size(x)[end]        
        acc += sum(onecold(ŷ |> cpu) .== onecold(y |> cpu))
        ntot += size(x)[end]
    end
    return (loss = l/ntot |> round4, acc = acc/ntot*100 |> round4)
end


function train(use_cuda, model_save_file)

    # Get data
    x_train, y_train = MLDatasets.MNIST.traindata(Float32)
    x_test, y_test = MLDatasets.MNIST.testdata(Float32)

    x_train = preprocess(x_train)
    x_test = preprocess(x_test)

    y_train, y_test = onehotbatch(y_train, 0:9), onehotbatch(y_test, 0:9)

    batchsize = 128
    epochs = 1
    train_loader = DataLoader((x_train, y_train), batchsize=batchsize, shuffle=true)
    test_loader = DataLoader((x_test, y_test),  batchsize=batchsize)   


    if use_cuda
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    model = create_model(size(x_train)[1:3], 10) |> device

    ps = Flux.params(model)
    opt = ADAM(3e-4)

    function report(epoch)
        train = eval_loss_accuracy(train_loader, model, device)
        test = eval_loss_accuracy(test_loader, model, device)        
        println("Epoch: $epoch   Train: $(train)   Test: $(test)")
    end

    @info "Start Training"
    report(0)
    for epoch in 1:epochs
        @showprogress for (x, y) in train_loader
            x, y = x |> device, y |> device
            gs = Flux.gradient(ps) do
                    ŷ = model(x)
                    loss(ŷ, y)
                end

            Flux.Optimise.update!(opt, ps, gs)
        end
        
        report(epoch)
    end

    let model = cpu(model) #return model to cpu before serialization
        BSON.@save model_save_file model
    end

end


function evaluate(saved_model)

    BSON.@load "cnn-model.bson" model

    x_test, y_test = MLDatasets.MNIST.testdata(Float32)
    x_test = preprocess(x_test)
    y_test = onehotbatch(y_test, 0:9)

    test_loader = DataLoader((x_test, y_test),  batchsize=128)   
    test = eval_loss_accuracy(test_loader, model, cpu)
    println("Evaluation on test set: $test")

    
end


if abspath(PROGRAM_FILE) == @__FILE__
    model_path = "cnn-model.bson"
    train(false, model_path)
    evaluate(model_path)
end