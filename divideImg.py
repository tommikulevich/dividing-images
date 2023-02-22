# Dividing images into folders according to features
# - Author: Tomash Mikulevich
# - Created with: PyCharm 2022.2.1 (Professional Edition - Student Pack)

import os
import shutil
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
from scipy.spatial import distance
import pickle
from operator import itemgetter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"


def dirFileList(dirPath):
    fileList = list()

    for path, _, dirFiles in os.walk(dirPath):
        for dirFile in dirFiles:
            fileList.append(os.path.join(path, dirFile))

    return fileList


def extractFeatures(imagePath, imageShape, featureModel):
    imgEdited = Image.open(imagePath).convert('L').resize(imageShape)
    imgEdited = np.stack((imgEdited,) * 3, axis=-1)
    imgEdited = np.array(imgEdited) / 255.0

    emb = featureModel.predict(imgEdited[np.newaxis, ...])
    imageFeatures = np.array(emb)
    flattenedFeatures = imageFeatures.flatten()

    return flattenedFeatures


imgPath = "img"
imgGroupsPath = "imgGroups"
imgWithoutGroupsPath = "_noGroup"
featuresBinPath = "imgFeatures.bin"
imgShape = (512, 512)
loadFeatures = True                     # You can change loadFeatures to False if you want to extract new features
compareCoeff = 0.455                    # You can change compareCoeff to get more accurate results

shutil.rmtree(imgGroupsPath, ignore_errors=True)
os.makedirs(imgGroupsPath, exist_ok=True)

print("Initializing model ...")
layer = hub.KerasLayer("https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_xl/feature_vector/2")
model = tf.keras.Sequential([layer])

if loadFeatures:
    print("Loading features ...")

    file = open(featuresBinPath, 'rb')
    features = pickle.load(file)
    file.close()
else:
    print("Extracting features ...")

    features = {}
    imgList = dirFileList(imgPath)
    imgList.sort()

    for img in imgList:
        features[img] = extractFeatures(img, imgShape, model)

    newFeatures = "./" + featuresBinPath
    with open(newFeatures, 'wb') as file:
        pickle.dump(features, file)

featuresWithKey = {}
images = list()
for f in features:
    images.append([f, features[f]])
    featuresWithKey[os.path.basename(f)] = features[f]

print("Comparing features and creating groups ...")
tempCount = 1
for img in images:
    image = img[0]
    imageBasename1 = os.path.basename(image)

    if tempCount == 1:
        subFol = imgGroupsPath + "/" + str(tempCount)
        shutil.rmtree(subFol, ignore_errors=True)
        os.makedirs(subFol, exist_ok=True)

        destination1 = subFol + "/" + imageBasename1
        shutil.copy(image, destination1)

        tempCount += 1
    else:
        subImages = dirFileList(imgGroupsPath)
        temp = list()

        for subImage in subImages:
            imgBasename2 = os.path.basename(subImage)
            similarity = distance.cdist([featuresWithKey[imageBasename1]], [featuresWithKey[imgBasename2]], 'cosine')[0]
            temp.append((similarity, subImage))

        temp = sorted(temp, key=itemgetter(0))

        if temp[0][0] < compareCoeff:                   # You can change compareCoeff to get more accurate results
            subDir = os.path.dirname(temp[0][1])
            destination2 = subDir + "/" + imageBasename1
            shutil.copy(image, destination2)
        else:
            subFol = imgGroupsPath + "/" + str(tempCount)
            shutil.rmtree(subFol, ignore_errors=True)
            os.makedirs(subFol, exist_ok=True)

            destination = subFol + "/" + os.path.basename(image)
            shutil.copy(image, destination)

        tempCount += 1

numOfGroups = 0
folders = list()
for f in os.listdir(imgGroupsPath):
    directory = os.path.join(imgGroupsPath, f)

    if os.path.isdir(directory):
        folders.append(directory)
        numOfGroups += 1

shutil.rmtree(imgGroupsPath + "/" + imgWithoutGroupsPath, ignore_errors=True)
os.makedirs(imgGroupsPath + "/" + imgWithoutGroupsPath, exist_ok=True)

for folder in folders:
    files = dirFileList(folder)

    if len(files) == 1:
        source = files[0]
        destination = imgGroupsPath + "/" + imgWithoutGroupsPath + "/" + os.path.basename(source)
        shutil.copy(source, destination)
        shutil.rmtree(folder, ignore_errors=True)

        numOfGroups -= 1

print("Done. Num of groups: ", numOfGroups)
