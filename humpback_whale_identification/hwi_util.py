import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img, array_to_img
from collections import Counter
import os


class HWIDataBuilder:
    def __init__(self, config):
        self.config = config
        self.classLabel = self.buildClassLabel()

    # Return a dict with imageID as key and image name as value
    # This is to understand total number of classes and no of images per classes
    def buildClassLabel(self):
        f = open(self.config.IMAGE_ID_FILE)
        next(f)
        rows = f.read().strip().split("\n")
        classLabel = {}
        for row in rows:
            (name, ID) = row.split(",")
            if ID in classLabel.keys():
                classLabel[ID].append(name)
            else:
                classLabel[ID] = [name]
        f.close()

        return classLabel


    # Print the number of classes per samples
    def getSampleDistributions(self):
        d = {}
        for key in self.classLabel.keys():
            d[key] = len(self.classLabel[key])
        counter = Counter(d.values())
        totalImages = 0
        factor = 1
        for key in counter.keys():
            print("Sample count = {} : class count {} : class pc {} : sample pc {}".format(
                   key, counter[key], (counter[key]/self.config.NUM_CLASSES) * 100,
                   ((key * counter[key])/25363) * 100))
            if key > self.config.NO_GEN_NUMBER:
                factor = 1
            elif key < len(self.config.GEN_IMAGES_FACTORS):
                factor = self.config.GEN_IMAGES_FACTORS[key - 1]
            else:
                factor = self.config.MAX_GEN_FACTOR
            totalImages += (key * factor * counter[key])
        print("Total generated images = ", totalImages)



    # Using classLabel dict, depending on the number of images available for each class
    # generate the images in the specified dir and construct the full image path
    # for all the generated images and original images along with their labels.
    # Save all the file names in datasets folder for futher processing.
    def generateImage(self):
        imageGen = ImageDataGenerator(rotation_range = 30, width_shift_range = 0.2,
                                      height_shift_range = 0.2, shear_range = 0.2,
                                      zoom_range = 0.2, horizontal_flip = True,
                                      fill_mode = "nearest")
        totalGen = 0
        for key in self.classLabel.keys():
            imgCount = len(self.classLabel[key])
            for idx in range(imgCount):
                oriImg = load_img(os.path.sep.join([self.config.TRAIN_IMAGES_PATH, 
                                                    self.classLabel[key][idx]]))
                # Copy the original train image to generated location.
                # Rename image with class label + idx as prefix.
                genPrefix = "-".join([key, str(idx), self.classLabel[key][idx]])
                oriImg.save(os.path.sep.join([self.config.GENERATED_TRAIN_IMAGES_PATH, 
                                              genPrefix]))
                imgGenCount = 0
                if imgCount > self.config.NO_GEN_NUMBER:
                    # No generation of images for class having more than 100 samples
                    continue
                elif imgCount < len(self.config.GEN_IMAGES_FACTORS):
                    imgGenCount = self.config.GEN_IMAGES_FACTORS[imgCount - 1]
                else:
                    imgGenCount = self.config.MAX_GEN_FACTOR
                img = img_to_array(oriImg)
                # flow required rank 4
                img = np.expand_dims(img, axis = 0)
                print("Generating for {} : image count {} : imageGenCount {}".
                        format(genPrefix, imgCount, imgGenCount))
                genPrefix = genPrefix.split(".")[:-1][0]
                i = 0
                for _ in imageGen.flow(img, batch_size = 1, 
                                       save_to_dir = self.config.GENERATED_TRAIN_IMAGES_PATH, 
                                       save_prefix = genPrefix, save_format = "jpg"):
                    i += 1
                    totalGen += 1
                    if i > imgGenCount:
                        break

        return totalGen
 

    # Get all the image file name from the generated folder
    def getAllFileNames(self, path):
        for dirPath, dirNames, fileNames in os.walk(path):
            for filename in fileNames:
                if filename.split(".")[-1] == "jpg":
                    imgPath = os.path.sep.join([dirPath, filename])
                    yield imgPath

    # Build training image list and its corresponding labels
    def getTrainNameLabel(self):
        imgPath = []
        imgLabel = []
        for name in self.getAllFileNames(self.config.GENERATED_TRAIN_IMAGES_PATH):
            imgLabel.append(name.split(os.path.sep)[-1].split("-")[0])
            imgPath.append(name)

        return (np.array(imgPath), np.array(imgLabel))

    # Build all testing image list
    def getTestName(self):
        imgPath = []
        for name in self.getAllFileNames(self.config.TEST_IMAGES_PATH):
            imgPath.append(name)

        return np.array(imgPath)

    def generateCodeClassMap(self):
        codeClass = {}
        if not os.path.exists(self.config.ENC_CODE_CLASS_MAP):
            fout = open(self.config.ENC_CODE_CLASS_MAP, "w")
            fin = open(self.config.TRAIN_LIST, "r")
            rows = fin.read().strip().split("\n")
            for row in rows:
                words = row.split(" ")
                key = int(words[1])
                value = words[2].split(os.path.sep)[-1].split("-")[0]
                if key in codeClass.keys():
                    continue
                else:
                    codeClass[key] = value
            fin.close()
            for key in codeClass.keys():
                fout.write("{} {}\n".format(str(key), codeClass[key]))
            fout.close()
        else:
            fin = open(self.config.ENC_CODE_CLASS_MAP, "r")
            rows = fin.read().strip().split("\n")
            for row in rows:
                words = row.split(" ")
                codeClass[int(words[0])] = words[1]
            fin.close()

        return codeClass

