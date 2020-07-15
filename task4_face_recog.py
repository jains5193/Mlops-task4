#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.applications import VGG16

model = VGG16(weights = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', 
                 include_top = False, 
                 input_shape = (224, 224, 3))

model.summary()


# In[2]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model



for layer in model.layers:
    layer.trainable = False

top_model = model.output
top_model = Flatten(name = "flatten")(top_model)
top_model = Dense(526, activation = "relu")(top_model)
top_model = Dense(263, activation = "relu")(top_model)
top_model = Dense(2 , activation = "softmax")(top_model)

newmodel = Model(inputs=model.input , outputs=top_model)

newmodel.summary()


# In[5]:


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=45,
      width_shift_range=0.3,
      height_shift_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest')
 
validation_datagen = ImageDataGenerator(rescale=1./255)

train_batchsize = 20
val_batchsize = 20

train_generator = train_datagen.flow_from_directory(
        'data/train_set',
        target_size=(224, 224),
        batch_size=train_batchsize,
        class_mode='categorical')
 
validation_generator = validation_datagen.flow_from_directory(
        'data/test_set',
        target_size=(224, 224 ),
        batch_size=val_batchsize,
        class_mode='categorical')


# In[6]:


from keras.optimizers import RMSprop

newmodel.compile(loss = 'categorical_crossentropy',
              optimizer = RMSprop( lr = 0.001 ),
              metrics = ['accuracy'])

nb_train_samples = 1000
nb_validation_samples = 370
##epochs = 3
batch_size = 16
history = newmodel.fit_generator(
    train_generator,
    steps_per_epoch = 25,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size,
    epochs = 1)


# In[7]:


result_accuracy = history.history['accuracy']

newmodel.save('facial_recog.h5')

newmodel.save('facial_recog.xml')


# In[8]:


from keras.models import load_model


# In[9]:


classifier = load_model('facial_recog.h5')


# In[10]:


import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join


# In[20]:


human_dict = {"[0]": "men", 
              "[1]": "women"}


# In[21]:


human_dict_n = {"n0": "men", 
                "n1": "women"}


# In[ ]:





# In[27]:


def draw_test(name, pred, im):
    human = human_dict[str(pred)]
    BLACK = [0,0,0]
    expanded_image = cv2.copyMakeBorder(im, 80, 0, 0, 100 ,cv2.BORDER_CONSTANT,value=BLACK)
    cv2.putText(expanded_image, human, (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
    cv2.imshow(name, expanded_image)


# In[28]:


def getRandomImage(path):
    """function loads a random images from a random folder in our test path """
    folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))
    random_directory = np.random.randint(0,len(folders))
    path_class = folders[random_directory]
    print("Class - " + human_dict_n[str(path_class)])
    file_path = path + path_class
    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    random_file_index = np.random.randint(0,len(file_names))
    image_name = file_names[random_file_index]
    return cv2.imread(file_path+"/"+image_name)  


# In[29]:


for i in range(0,10):
    input_im = getRandomImage("data/test_set/")
    input_original = input_im.copy()
    input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    
    input_im = cv2.resize(input_im, (224, 224), interpolation = cv2.INTER_LINEAR)
    input_im = input_im / 255.
    input_im = input_im.reshape(1,224,224,3)
    
    res = np.argmax(classifier.predict(input_im, 1, verbose = 0), axis=1)
    draw_test("Prediction", res, input_original) 
    cv2.waitKey(0)

cv2.destroyAllWindows()


# In[ ]:


# Get Prediction
    


# In[ ]:




