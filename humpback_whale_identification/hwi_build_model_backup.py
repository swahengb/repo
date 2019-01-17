import mxnet as mx
import numpy as np

class HWIResNet:
    # ResNet module function - this can be used to construct stack of resnet module.
    @staticmethod
    def resModule(data, num_filter, stride, reduceDim, name, bn_mom = 0.9):
        bn1 = mx.sym.BatchNorm(data = data, fix_gamma = False, eps = 2e-5, momentum = bn_mom, name=name + '_bn1')
        act1 = mx.sym.Activation(data = bn1, act_type = 'relu', name = name + '_relu1')
        conv1 = mx.sym.Convolution(data = act1, num_filter = int(num_filter * 0.25), kernel = (1,1), stride = (1,1), 
                                   pad = (0,0), no_bias = True, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data = conv1, fix_gamma = False, eps = 2e-5, momentum = bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(data = bn2, act_type = 'relu', name = name + '_relu2')
        conv2 = mx.sym.Convolution(data = act2, num_filter = int(num_filter * 0.25), kernel = (3,3), stride = stride, 
                                   pad = (1,1), no_bias = True, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data = conv2, fix_gamma = False, eps = 2e-5, momentum = bn_mom, name = name + '_bn3')
        act3 = mx.sym.Activation(data = bn3, act_type = 'relu', name = name + '_relu3')
        conv3 = mx.sym.Convolution(data = act3, num_filter = num_filter, kernel = (1,1), stride = (1,1), pad = (0,0), 
                                   no_bias = True, name = name + '_conv3')
        shortcut = data
        if reduceDim:
            shortcut = mx.sym.Convolution(data = act1, num_filter = num_filter, kernel = (1,1), stride = stride, 
                                   no_bias = True, name = name+'_sc')
        return conv3 + shortcut


    @staticmethod
    def resnetModel(units, num_stages, filter_list, num_classes, bn_mom=0.9):
        num_unit = len(units)
        assert(num_unit == num_stages)
        data = mx.sym.Variable('data')
        data = mx.sym.BatchNorm(data = data, fix_gamma = True, eps = 2e-5, momentum = bn_mom, name ='bn_data')
        body = mx.sym.Convolution(data = data, num_filter = filter_list[0], kernel = (7, 7), stride = (2,2), 
                                  pad = (3, 3), no_bias = True, name = "conv0")
        body = mx.sym.BatchNorm(data = body, fix_gamma = False, eps = 2e-5, momentum = bn_mom, name = 'bn0')
        body = mx.sym.Activation(data = body, act_type = 'relu', name = 'relu0')
        body = mx.sym.Pooling(data = body, kernel = (3, 3), stride = (2,2), pad = (1,1), pool_type = 'max')

        for i in range(num_stages):
            body = HWIResNet.resModule(body, filter_list[i + 1], (1 if i == 0 else 2, 1 if i == 0 else 2), False,
                             name = 'stage%d_unit%d' % (i + 1, 1))
            for j in range(units[i]-1):
                body = HWIResNet.resModule(body, filter_list[i + 1], (1, 1), True, name = 'stage%d_unit%d' % (i + 1, j + 2))
        bn1 = mx.sym.BatchNorm(data = body, fix_gamma = False, eps = 2e-5, momentum = bn_mom, name = 'bn1')
        relu1 = mx.sym.Activation(data = bn1, act_type = 'relu', name = 'relu1')
        pool1 = mx.sym.Pooling(data = relu1, global_pool = True, kernel = (7, 7), pool_type = 'avg', name = 'pool1')
        flat = mx.sym.Flatten(data = pool1)
        fc1 = mx.sym.FullyConnected(data = flat, num_hidden = num_classes, name = 'fc1')

        return mx.sym.SoftmaxOutput(data = fc1, name = 'softmax')

    @staticmethod
    def buildResNet(num_classes):
        units = [3, 4, 6, 3]
        num_stages = 4
        filter_list = [64, 256, 512, 1024, 2048]

        return HWIResNet.resnetModel(units = units, num_stages  = num_stages, 
                                     filter_list = filter_list, num_classes = num_classes)
        '''
        stages = [3, 4, 6, 3]
        filters = [64, 128, 256, 512]
        return HWIResNet.buildModel(num_classes, stages, filters, 64)
        '''

    '''
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

    '''
