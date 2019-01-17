from hwi_util import HWIDataBuilder
import hwi_config as conf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
import progressbar
import json
import numpy as np

builder = HWIDataBuilder(conf)
# No need to call it as images are already generated
#totalImage = builder.generateImage()
#print("Total generated images ", totalImage)
#builder.getSampleDistributions()
(trainImgList, trainLabel) = builder.getTrainNameLabel()
lb = LabelEncoder()
encodedLabel = lb.fit_transform(trainLabel)
X_train, X_val, y_train, y_val = train_test_split(trainImgList, encodedLabel, test_size = conf.NUM_TEST_IMAGES, 
                                   random_state = 42, stratify = encodedLabel)
# To generate .lst file for train, test and validation
datasets = [("train", X_train, y_train, conf.TRAIN_LIST),
        ("val", X_val, y_val, conf.VAL_LIST)]

# To store R, G, B mean for all images
RGBMean = { "R" : [], "G" : [], "B" : [] }

for datatype, dataPath, labels, fileName in datasets:
    outfile = open(fileName, "w")
    widget = ["Building {} list".format(datatype), progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval = len(dataPath), widgets = widget).start()
    for i, (path, label) in enumerate(zip(dataPath, labels)):
        row = " ".join([str(i), str(label), path])
        outfile.write("{}\n".format(row))
        # Calculate R, G, B mean if it is for training data
        if datatype == "train":
            img = cv2.imread(path)
            (b, g, r) = cv2.mean(img)[:3]
            RGBMean["R"].append(r)
            RGBMean["G"].append(g)
            RGBMean["B"].append(b)
        pbar.update(i)
    pbar.finish()
    outfile.close()
print("Serializing the RGB mean data ...")
outfile = open(conf.MEAN_PATH, "w")
M = { "R" : np.mean(RGBMean["R"]), "G" : np.mean(RGBMean["G"]), "B" : np.mean(RGBMean["B"]) }
outfile.write(json.dumps(M))
outfile.close()
