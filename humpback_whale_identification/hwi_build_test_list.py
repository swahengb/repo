import hwi_config as config
from hwi_util import HWIDataBuilder
from keras.preprocessing.image import array_to_img
import mxnet as mx
import cv2

builder = HWIDataBuilder(config)
imgList = builder.getTestName()
'''
# Generate test list file
lstFile = open(config.TEST_LIST, "w")
for i, name in enumerate(imgList):
    lstFile.write("{} {} {}\n".format(i, i, name))
lstFile.close()
'''
# Generate test rec file
record = mx.recordio.MXRecordIO(config.TEST_REC, "w")
for name in imgList:
    img = cv2.imread(name)
    #img = mx.img.imdecode(img)
    record.write(bytes(img))
record.close()
'''
testIter = mx.io.ImageRecordIter(path_imgrec = "/media3/kaggle/humpback_whale_identification/output/record/test1.rec", data_shape = (3, 224, 224), batch_size = 10)
batchImg = testIter.next()
for i in range(len(batchImg.data)):
    img = array_to_img(batchImg.data[i])
    cv2.imshow("test", img)
    cv2.waitKey(0)
cv2.destroyAllWindows()
'''

