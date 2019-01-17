import hwi_config as config
import mxnet as mx
import numpy as np
import argparse
import json
import os
from collections import namedtuple
import cv2
from hwi_util import HWIDataBuilder
from sklearn.preprocessing import LabelEncoder



ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", help = "Path to save model checkpoints", required = True)
ap.add_argument("-p", "--prefix", help = "Prefix to be used in checkpoint filename", required = True)
ap.add_argument("-e", "--epochcount", help = "Epoch count to restart", default = 0, type = int)
ap.add_argument("-f", "--fileName", help = "test file name", required = True)
args = vars(ap.parse_args())

builder = HWIDataBuilder(config)
labelMap = builder.generateCodeClassMap()

# This portion of the code need to be removed
##############################################
(trainImgList, trainLabel) = builder.getTrainNameLabel()
lb = LabelEncoder()
encodedLabel = lb.fit_transform(trainLabel)
##############################################

Batch = namedtuple('Batch', ['data'])
RGBMean = json.loads(open(config.MEAN_PATH).read())
testIter = mx.io.ImageRecordIter(path_imgrec = config.VAL_REC, data_shape = (3, 224, 224), 
           batch_size = config.BATCH_SIZE, mean_r = RGBMean["R"], 
           mean_g = RGBMean["G"], mean_b = RGBMean["B"])

# load the model from disk 
print("Loading model...") 

modelPath = os.path.sep.join([args["checkpoints"], args["prefix"]]) 
sym, arg_params, aux_params = mx.model.load_checkpoint(modelPath, args["epochcount"])
mod = mx.mod.Module(symbol=sym, context=mx.gpu(0), label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)

#submitFile = open("submission.csv", "w")
#submitFile.write("Image,Id\n")
#testImages = builder.getTestName()
#for i in range(len(testImages)):
#for i in range(5):
name = args["fileName"]
img = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
cv2.imshow("test", img)
cv2.waitKey(0)
img = np.swapaxes(img, 0, 2)
img = np.swapaxes(img, 1, 2)
img = img[np.newaxis, :]
mod.forward(Batch([mx.nd.array(img)]))
prob = mod.get_outputs()[0].asnumpy()
#prob = mod.predict(mx.nd.array(img), always_output_list = True)
prob = np.squeeze(prob)

index = np.argsort(prob)[:: -1]
#name = name.split(os.path.sep)[-1]
labels = ""
probs = ""
for j in index[0 : 5]:
    labels = " ".join([labels, labelMap[j]])
    probs = " ".join([probs, str(prob[j])])
#submitFile.write("{},{}\n".format(name, (labels.strip())))
print("name = {} :: label = {}".format(name, labels.strip()))
print("Probalities : {}".format(probs))
#submitFile.close()


