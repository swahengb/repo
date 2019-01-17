from hwi_build_model import HWIResNet
import hwi_config as conf
import mxnet as mx
import argparse
import json
import logging
import os


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", help = "Path to save model checkpoints", required = True)
ap.add_argument("-p", "--prefix", help = "Prefix to be used in checkpoint filename", required = True)
ap.add_argument("-e", "--epochcount", help = "Epoch count to restart", default = 0, type = int)
args = vars(ap.parse_args())

logging.basicConfig(filename = "training_updated{}.log".format(args["epochcount"]), filemode = "w", level = logging.DEBUG)
batch_size = conf.BATCH_SIZE
RGBMean = json.loads(open(conf.MEAN_PATH).read())

trainIter = mx.io.ImageRecordIter(path_imgrec = conf.TRAIN_REC, data_shape = (3, 224, 224), 
                                  batch_size = batch_size, rand_crop = True, rotate = 15, 
                                  mean_r = RGBMean["R"], mean_g = RGBMean["G"], mean_b = RGBMean["B"], 
                                  rand_mirror = True, max_shear_ratio = 0.1, 
                                  preprocess_threads = conf.NUM_DEVICE * 2) 

valIter = mx.io.ImageRecordIter(path_imgrec = conf.VAL_REC, data_shape = (3, 224, 224), 
                                batch_size = batch_size, mean_r = RGBMean["R"], mean_g = RGBMean["G"],
                                mean_b = RGBMean["B"])

opt = mx.optimizer.SGD(learning_rate = 1e-3, momentum = 0.9, wd = 0.0001, rescale_grad = 1.0 / batch_size)
# prepare checkpoints file path - this is the path, 
# where partially train model will be save through callback or reload the model to resume the training.
model_path = os.path.sep.join([args["checkpoints"], args["prefix"]])

# create the model - 1. if epoch is equal to 0; build new model 
#                    2. if epoch is greater than 0; load it using epoch 
if args["epochcount"] <= 0:
    print("Building the model ...")
    model = HWIResNet.buildResNet(conf.NUM_CLASSES)
    print("Compiling the model ...")
    model =  mx.mod.Module(context = mx.gpu(0), symbol = model)
else:
    print("Loading the model ...")
    model, argParams, auxParams = mx.model.load_checkpoint(model_path, args["epochcount"])
    model = mx.mod.Module(symbol = model, context = mx.gpu(0))
    model.bind(data_shapes = trainIter.provide_data, label_shapes = trainIter.provide_label)
    #model.set_params(argParams, auxParams, allow_missing = True)

# prepare callback for checkpoints and metrics to be used for fit the model
epochEndCallback = [mx.callback.do_checkpoint(model_path)]
batchEndCallback = [mx.callback.Speedometer(batch_size, 250)]
evaluationMetrics = [mx.metric.Accuracy(), mx.metric.TopKAccuracy(top_k = 5)] 
# fit model
print("Start training the model ...")
model.fit(trainIter, eval_data = valIter, eval_metric = evaluationMetrics, 
          epoch_end_callback = epochEndCallback, batch_end_callback = batchEndCallback,  
          begin_epoch = args["epochcount"], num_epoch = conf.NUM_EPOCH, optimizer = opt,
          initializer = mx.initializer.MSRAPrelu())

