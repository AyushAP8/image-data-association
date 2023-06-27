import random
import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans, MeanShift

number_of_clusters = 3
data_path = "Data"

paths = os.listdir(data_path)
n_imgs = len(paths)
n_clusters = 4
data_path = data_path
random.shuffle(paths)
image_paths = paths[:n_imgs]
del paths 
try:
	shutil.rmtree("output")
except FileExistsError:
	pass
os.makedirs("output")
for i in range(n_clusters):
	os.makedirs("output\\cluster" + str(i))



images = []
for image in image_paths:
	images.append(cv2.cvtColor(cv2.resize(cv2.imread(data_path + "\\" + image), (224,224)), cv2.COLOR_BGR2RGB))
images = np.float32(images)#.reshape(len(self.images), -1)
images /= 255
print("\n\n " + str(n_imgs) + " images from the \"" + data_path + "\" folder has been loaded")


model1 = tf.keras.applications.vgg19.VGG19(include_top=False, weights="imagenet", input_shape=(224,224,3))
pred = model1.predict(images)
print("pred: ", len(pred))
print(images.shape)
images_new = pred.reshape(images.shape[0], -1)
print("pred: ", len(pred))



#model = KMeans(n_clusters=n_clusters, random_state=728)
#from sklearn.cluster import  estimate_bandwidth
#bandwidth = estimate_bandwidth(images_new, quantile=0.2, n_samples=500)
#print(bandwidth)
bandwidth = []
for b in range(10000):
	a=30
	b+=1
	b*=0.1
	model = MeanShift(bandwidth = a+b)
	model.fit(images_new)
	print(a+b," - ", len(model.cluster_centers_))
	if len(model.cluster_centers_) == number_of_clusters:
		bandwidth.append(a+b)
	if len(model.cluster_centers_) < number_of_clusters:
		break
#print(predictions)

for b in bandwidth:
	model = MeanShift(bandwidth = b)
	model.fit(images_new)
	predictions_ms = model.predict(images_new)
	print("cluster centre from mean shift : ",b , " - ", model.cluster_centers_)

model = KMeans(n_clusters=n_clusters, random_state=728)
model.fit(images_new)
predictions_km = model.predict(images_new)
print("cluster centre from kmeans : ",model.cluster_centers_)

for i in range(n_imgs):
	shutil.copy2(data_path+"\\"+image_paths[i], "output\cluster"+str(predictions_ms[i]))
print("\nCheck Output Directory\n")
