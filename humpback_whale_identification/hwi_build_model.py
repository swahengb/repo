import mxnet as mx

class HWIResNet:
    # ResNet module function - this can be used to construct stack of resnet module.
    @staticmethod
    def buildResNet(num_classes):
        stages = [3, 4, 6, 3]
        filters = [64, 128, 256, 512]
        return HWIResNet.buildModel(num_classes, stages, filters, 64)

    @staticmethod
    def resModule(data, filters, stride, reduceSize = False, eps = 2e-5):
        shortcut = data
        # First layer of ResNet module is 1x1 CONV
        bn1 = mx.sym.BatchNorm(data = data, fix_gamma = False, eps = eps)
        act1 = mx.sym.Activation(data = bn1, act_type = "relu")
        conv1 = mx.sym.Convolution(data = act1, pad=(0, 0),kernel = (1, 1), 
                stride = (1, 1), num_filter = filters, no_bias = True)

        # Second layer of ResNet module is 3x3 CONV
        bn2 = mx.sym.BatchNorm(data = conv1, fix_gamma = False, eps = eps)
        act2 = mx.sym.Activation(data = bn2, act_type = "relu")
        conv2 = mx.sym.Convolution(data = act2, pad = (1, 1), kernel = (3, 3), 
                stride = stride, num_filter = filters, no_bias = True)

        # Third layer of ResNet module is 1x1 CONV
        bn3 = mx.sym.BatchNorm(data = conv2, fix_gamma = False, eps = eps)
        act3 = mx.sym.Activation(data = bn3, act_type = "relu")
        conv3 = mx.sym.Convolution(data = act3, pad = (0, 0), kernel = (1, 1), 
                stride = (1, 1), num_filter = (filters * 4), no_bias = True)

        # To reduce the spatial size, apply a CONV layer to the shortcut
        if reduceSize:
            shortcut = mx.sym.Convolution(data = act1, pad = (0, 0), kernel = (1, 1), 
                       stride = stride, num_filter = (filters * 4), no_bias = True)

        # Add shortcut to the final CONV
        add = conv3 + shortcut
        return add

    @staticmethod 
    def buildModel(classes, stages, filters, firstFilter, eps = 2e-5): 
        data = mx.sym.Variable("data")
        # Initial common layers for all resnet architecture  
        bn1 = mx.sym.BatchNorm(data = data, fix_gamma = True, eps = eps) 
        conv1 = mx.sym.Convolution(data = bn1, pad = (3, 3), kernel = (7, 7), 
                  stride = (2, 2), num_filter = firstFilter, no_bias = True) 
        bn2 = mx.sym.BatchNorm(data = conv1, fix_gamma = False, eps = eps) 
        act1 = mx.sym.Activation(data = bn2, act_type = "relu") 
        body = mx.sym.Pooling(data = act1, pool_type = "max", pad = (1, 1), 
                  kernel = (3, 3), stride = (2, 2)) 

        # Construct the four stages of the resnet, each having (3, 4, 6, 3) modules
        for i in range(0, len(stages)): 
            # initialize the stride, then apply a residual module 
            stride = (1, 1) if i == 0 else (2, 2) 
            body = HWIResNet.resModule(body, filters[i], stride, reduceSize = True, eps = eps) 
            # loop over the number of layers in the stage 
            for j in range(0, stages[i] - 1): 
                # apply a ResNet module 
                body = HWIResNet.resModule(body, filters[i], (1, 1), eps = eps)

        # Final layers of CONV 
        bn3 = mx.sym.BatchNorm(data = body, fix_gamma = False, eps = eps) 
        act2 = mx.sym.Activation(data = bn3, act_type = "relu") 
        pool2 = mx.sym.Pooling(data = act2, pool_type = "avg", global_pool = True, kernel = (7, 7))

        # softmax classifier 
        flatten = mx.sym.Flatten(data = pool2) 
        fc1 = mx.sym.FullyConnected(data = flatten, num_hidden = classes)
        model = mx.sym.SoftmaxOutput(data = fc1, name = "softmax") 

        return model

