import os
import cv2 as cv
import numpy as np
import gc
import caer
import canaro
import matplotlib.pyplot as plt


IMG_SIZE = (80,80) 
channels = 1
train_folder_path = r'U:\vscode\opencv\deepLearning\train'


# Grab the top 10 folders with max images

# Create empty dict
# Store the count of images
# sort it in descending order

persons_image_count = {}
for person in os.listdir(train_folder_path) :
    persons_image_count[person] = len( os.listdir(os.path.join(train_folder_path, person)) )

persons_image_count = caer.sort_dict( persons_image_count, descending=True )
print(persons_image_count)

print('\n\n\n')

# Store top 10 in list
persons = []
for i in range(10) :
    persons.append(persons_image_count[i][0])

print(persons)


print('\n\n\n')


# Creating the Training Data
# verbose = 0, if we don't want to see the processing of train
train = caer.preprocess_from_dir( train_folder_path, persons, channels=channels, IMG_SIZE= IMG_SIZE, isShuffle= True, verbose=False )
print(len(train))

plt.figure(figsize=(50,50))
plt.imshow(train[0][0], cmap='gray')
plt.show()




featureSet, labels = caer.sep_train( train, IMG_SIZE=IMG_SIZE )


# Normalize the Feature Set in the range (0,1)
featureSet = caer.normalize(featureSet)

# Convert numerical int to binary class vectors for labels
from keras.utils import to_categorical

labels = to_categorical(labels, len(persons))



# split the featureSet and labels into validation set and feature set
# 0.2 means 20% goes to validationSet and 80% goes to the training set
x_train, x_val, y_train, y_val = caer.train_val_split( featureSet, labels, val_ratio= 0.2 )


# delete the variables we are not going to use
del train
del labels
del featureSet
gc.collect()




# Image Data generator : synthesis new images from already available images 
# for better processing of the images

BATCH_SIZE = 32
EPOCHS = 10

datagen = canaro.generators.imageDataGenerator()
train_gen = datagen.flow( x_train, y_train, batch_size = BATCH_SIZE )



# cREATE OUR MODEL
model = canaro.models.createSimpsonsModel( learning_rate=0.001,
                                         momentum=0.9, nesterov=True,
                                         IMG_SIZE=IMG_SIZE, channels=channels, output_dim=len(persons),
                                         loss='binary_crossentropy', decay=1e-6,  )

model.summary()



# Schedule learning rate so that our program train better
from keras.callbacks import LearningRateScheduler
callbacks_list = [LearningRateScheduler(canaro.lr_schedule)]










# #### Training
# training =  model.fit( train_gen,
#                     steps_per_epoch = len(x_train)//BATCH_SIZE,
#                     epochs = EPOCHS,
#                     validation_data = (x_val, y_val),
#                     validation_steps = len(y_val)//BATCH_SIZE,
#                     callbacks = callbacks_list )


print(persons)

test_path = r'U:\vscode\opencv\deepLearning\train\Sachin Tendulkar\Sachin Tendulkar192.jpg'

img = cv.imread(test_path)
cv.imshow( 'Testing Image', img )

# Preparing the image according to the properties we have defined during the training like shape, size, color

def prepare(img):
    img = cv.cvtColor( img, cv.COLOR_BGR2GRAY )
    img = cv.resize(img, IMG_SIZE)
    img = caer.reshape( img, IMG_SIZE, 1 )
    return img


predictions = model.predict( prepare(img) )
print( persons[np.argmax(predictions[0])] )
cv.waitKey(0)
