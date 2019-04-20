
# Artificial Intelligence Nanodegree

## Convolutional Neural Networks

## Project: Write an Algorithm for a Dog Identification App 

---

In this notebook, some template code has already been provided for you, and you will need to implement additional functionality to successfully complete this project. You will not need to modify the included code beyond what is requested. Sections that begin with **'(IMPLEMENTATION)'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section, and the specifics of the implementation are marked in the code block with a 'TODO' statement. Please be sure to read the instructions carefully! 

> **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.

>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut.  Markdown cells can be edited by double-clicking the cell to enter edit mode.

The rubric contains _optional_ "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. If you decide to pursue the "Stand Out Suggestions", you should include the code in this IPython notebook.



---
### Why We're Here 

In this notebook, you will make the first steps towards developing an algorithm that could be used as part of a mobile or web app.  At the end of this project, your code will accept any user-supplied image as input.  If a dog is detected in the image, it will provide an estimate of the dog's breed.  If a human is detected, it will provide an estimate of the dog breed that is most resembling.  The image below displays potential sample output of your finished project (... but we expect that each student's algorithm will behave differently!). 

![Sample Dog Output](images/sample_dog_output.png)

In this real-world setting, you will need to piece together a series of models to perform different tasks; for instance, the algorithm that detects humans in an image will be different from the CNN that infers dog breed.  There are many points of possible failure, and no perfect algorithm exists.  Your imperfect solution will nonetheless create a fun user experience!

### The Road Ahead

We break the notebook into separate steps.  Feel free to use the links below to navigate the notebook.

* [Step 0](#step0): Import Datasets
* [Step 1](#step1): Detect Humans
* [Step 2](#step2): Detect Dogs
* [Step 3](#step3): Create a CNN to Classify Dog Breeds (from Scratch)
* [Step 4](#step4): Use a CNN to Classify Dog Breeds (using Transfer Learning)
* [Step 5](#step5): Create a CNN to Classify Dog Breeds (using Transfer Learning)
* [Step 6](#step6): Write your Algorithm
* [Step 7](#step7): Test Your Algorithm

---
<a id='step0'></a>
## Step 0: Import Datasets

### Import Dog Dataset

In the code cell below, we import a dataset of dog images.  We populate a few variables through the use of the `load_files` function from the scikit-learn library:
- `train_files`, `valid_files`, `test_files` - numpy arrays containing file paths to images
- `train_targets`, `valid_targets`, `test_targets` - numpy arrays containing onehot-encoded classification labels 
- `dog_names` - list of string-valued dog breed names for translating labels


```python
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('/data/dog_images/train')
valid_files, valid_targets = load_dataset('/data/dog_images/valid')
test_files, test_targets = load_dataset('/data/dog_images/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("/data/dog_images/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))
```

    Using TensorFlow backend.
    

    There are 133 total dog categories.
    There are 8351 total dog images.
    
    There are 6680 training dog images.
    There are 835 validation dog images.
    There are 836 test dog images.
    

### Import Human Dataset

In the code cell below, we import a dataset of human images, where the file paths are stored in the numpy array `human_files`.


```python
import random
random.seed(8675309)

# load filenames in shuffled human dataset
human_files = np.array(glob("/data/lfw/*/*"))
random.shuffle(human_files)

# print statistics about the dataset
print('There are %d total human images.' % len(human_files))
```

    There are 13233 total human images.
    

---
<a id='step1'></a>
## Step 1: Detect Humans

We use OpenCV's implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) to detect human faces in images.  OpenCV provides many pre-trained face detectors, stored as XML files on [github](https://github.com/opencv/opencv/tree/master/data/haarcascades).  We have downloaded one of these detectors and stored it in the `haarcascades` directory.

In the next code cell, we demonstrate how to use this detector to find human faces in a sample image.


```python
import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline                               

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[1])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()
```

    Number of faces detected: 1
    


![png](https://github.com/RajNikhil/dataBlog/blob/gh-pages/assets/DogApp/dogApp_5_1.png?raw=true)


Before using any of the face detectors, it is standard procedure to convert the images to grayscale.  The `detectMultiScale` function executes the classifier stored in `face_cascade` and takes the grayscale image as a parameter.  

In the above code, `faces` is a numpy array of detected faces, where each row corresponds to a detected face.  Each detected face is a 1D array with four entries that specifies the bounding box of the detected face.  The first two entries in the array (extracted in the above code as `x` and `y`) specify the horizontal and vertical positions of the top left corner of the bounding box.  The last two entries in the array (extracted here as `w` and `h`) specify the width and height of the box.

### Write a Human Face Detector

We can use this procedure to write a function that returns `True` if a human face is detected in an image and `False` otherwise.  This function, aptly named `face_detector`, takes a string-valued file path to an image as input and appears in the code block below.


```python
# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
```

### (IMPLEMENTATION) Assess the Human Face Detector

__Question 1:__ Use the code cell below to test the performance of the `face_detector` function.  
- What percentage of the first 100 images in `human_files` have a detected human face?  
- What percentage of the first 100 images in `dog_files` have a detected human face? 

Ideally, we would like 100% of human images with a detected face and 0% of dog images with a detected face.  You will see that our algorithm falls short of this goal, but still gives acceptable performance.  We extract the file paths for the first 100 images from each of the datasets and store them in the numpy arrays `human_files_short` and `dog_files_short`.

__Answer:__ 
The Haar Cascade classifier for face detecion was able to detect 100% of the human faces in the fist 100 images of the human faces. The same classifier also detected human faces in 11% of the first 100 images of dog faces.


```python
human_files_short = human_files[:100]
dog_files_short = train_files[:100]
# Do NOT modify the code above this line.
```


```python
## TODO: Test the performance of the face_detector algorithm 
## on the images in human_files_short and dog_files_short.
hcount=0
for hface in human_files_short:
    if face_detector(hface)==True:
        hcount += 1
        
dcount=0
for dface in dog_files_short:
    if face_detector(dface)==True:
        dcount += 1
        
print('Percentage of the first 100 images in human_files detected as human face is {:0.1f}%'
      .format((hcount/len(human_files_short))*100))
print('Percentage of the first 100 images in dog_files detected as human face is {:0.1f}%'
      .format((dcount/len(dog_files_short))*100))
```

    Percentage of the first 100 images in human_files detected as human face is 100.0%
    Percentage of the first 100 images in dog_files detected as human face is 11.0%
    

__Question 2:__ This algorithmic choice necessitates that we communicate to the user that we accept human images only when they provide a clear view of a face (otherwise, we risk having unneccessarily frustrated users!). In your opinion, is this a reasonable expectation to pose on the user? If not, can you think of a way to detect humans in images that does not necessitate an image with a clearly presented face?

__Answer:__
With modern day technology progress, any limitation on what an app like the one we are developing can do is annoying to the users. Therefore our algorithm should be able to detect a face not matter what the angle of the face, i.e. tilted or side ways is in the image.

To do this we should train CNN models with images that have human faces in many other angles/positions than just frontal face.

We suggest the face detector from OpenCV as a potential way to detect human images in your algorithm, but you are free to explore other approaches, especially approaches that make use of deep learning :).  Please use the code cell below to design and test your own face detection algorithm.  If you decide to pursue this _optional_ task, report performance on each of the datasets.


```python
## (Optional) TODO: Report the performance of another  
## face detection algorithm on the LFW dataset
### Feel free to use as many code cells as needed.
```

---
<a id='step2'></a>
## Step 2: Detect Dogs

In this section, we use a pre-trained [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) model to detect dogs in images.  Our first line of code downloads the ResNet-50 model, along with weights that have been trained on [ImageNet](http://www.image-net.org/), a very large, very popular dataset used for image classification and other vision tasks.  ImageNet contains over 10 million URLs, each linking to an image containing an object from one of [1000 categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).  Given an image, this pre-trained ResNet-50 model returns a prediction (derived from the available categories in ImageNet) for the object that is contained in the image.


```python
from keras.applications.resnet50 import ResNet50

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')
```

    Downloading data from https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5
    102858752/102853048 [==============================] - 1s 0us/step
    

### Pre-process the Data

When using TensorFlow as backend, Keras CNNs require a 4D array (which we'll also refer to as a 4D tensor) as input, with shape

$$
(\text{nb_samples}, \text{rows}, \text{columns}, \text{channels}),
$$

where `nb_samples` corresponds to the total number of images (or samples), and `rows`, `columns`, and `channels` correspond to the number of rows, columns, and channels for each image, respectively.  

The `path_to_tensor` function below takes a string-valued file path to a color image as input and returns a 4D tensor suitable for supplying to a Keras CNN.  The function first loads the image and resizes it to a square image that is $224 \times 224$ pixels.  Next, the image is converted to an array, which is then resized to a 4D tensor.  In this case, since we are working with color images, each image has three channels.  Likewise, since we are processing a single image (or sample), the returned tensor will always have shape

$$
(1, 224, 224, 3).
$$

The `paths_to_tensor` function takes a numpy array of string-valued image paths as input and returns a 4D tensor with shape 

$$
(\text{nb_samples}, 224, 224, 3).
$$

Here, `nb_samples` is the number of samples, or number of images, in the supplied array of image paths.  It is best to think of `nb_samples` as the number of 3D tensors (where each 3D tensor corresponds to a different image) in your dataset!


```python
from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
```

### Making Predictions with ResNet-50

Getting the 4D tensor ready for ResNet-50, and for any other pre-trained model in Keras, requires some additional processing.  First, the RGB image is converted to BGR by reordering the channels.  All pre-trained models have the additional normalization step that the mean pixel (expressed in RGB as $[103.939, 116.779, 123.68]$ and calculated from all pixels in all images in ImageNet) must be subtracted from every pixel in each image.  This is implemented in the imported function `preprocess_input`.  If you're curious, you can check the code for `preprocess_input` [here](https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py).

Now that we have a way to format our image for supplying to ResNet-50, we are now ready to use the model to extract the predictions.  This is accomplished with the `predict` method, which returns an array whose $i$-th entry is the model's predicted probability that the image belongs to the $i$-th ImageNet category.  This is implemented in the `ResNet50_predict_labels` function below.

By taking the argmax of the predicted probability vector, we obtain an integer corresponding to the model's predicted object class, which we can identify with an object category through the use of this [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a). 


```python
from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))
```

### Write a Dog Detector

While looking at the [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a), you will notice that the categories corresponding to dogs appear in an uninterrupted sequence and correspond to dictionary keys 151-268, inclusive, to include all categories from `'Chihuahua'` to `'Mexican hairless'`.  Thus, in order to check to see if an image is predicted to contain a dog by the pre-trained ResNet-50 model, we need only check if the `ResNet50_predict_labels` function above returns a value between 151 and 268 (inclusive).

We use these ideas to complete the `dog_detector` function below, which returns `True` if a dog is detected in an image (and `False` if not).


```python
### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 
```

### (IMPLEMENTATION) Assess the Dog Detector

__Question 3:__ Use the code cell below to test the performance of your `dog_detector` function.  
- What percentage of the images in `human_files_short` have a detected dog?  
- What percentage of the images in `dog_files_short` have a detected dog?

__Answer:__ 
Using a pre-trained CNN model, ResNet-50, the algorithm seems to have become extremely efficient. The algorithm was successful in detecting a dog face in dog images 100% of the times and was also successful in rejecting human faces as not dog faces in human face images as it detected dog face in them 0% of the times.


```python
### TODO: Test the performance of the dog_detector function
### on the images in human_files_short and dog_files_short.
hfcount=0
dfcount=0
for hface in human_files_short:
    if dog_detector(hface) == True:
        hfcount += 1
for dface in dog_files_short:
    if dog_detector(dface) == True:
        dfcount += 1
```


```python
print("Human faces identified as dog: {:0.1f}%".format((hfcount/100)*100))
print("Dog faces identified as dog: {:0.1f}%".format((dfcount/100)*100))
```

    Human faces identified as dog: 0.0%
    Dog faces identified as dog: 100.0%
    

---
<a id='step3'></a>
## Step 3: Create a CNN to Classify Dog Breeds (from Scratch)

Now that we have functions for detecting humans and dogs in images, we need a way to predict breed from images.  In this step, you will create a CNN that classifies dog breeds.  You must create your CNN _from scratch_ (so, you can't use transfer learning _yet_!), and you must attain a test accuracy of at least 1%.  In Step 5 of this notebook, you will have the opportunity to use transfer learning to create a CNN that attains greatly improved accuracy.

Be careful with adding too many trainable layers!  More parameters means longer training, which means you are more likely to need a GPU to accelerate the training process.  Thankfully, Keras provides a handy estimate of the time that each epoch is likely to take; you can extrapolate this estimate to figure out how long it will take for your algorithm to train. 

We mention that the task of assigning breed to dogs from images is considered exceptionally challenging.  To see why, consider that *even a human* would have great difficulty in distinguishing between a Brittany and a Welsh Springer Spaniel.  

Brittany | Welsh Springer Spaniel
- | - 
<img src="images/Brittany_02625.jpg" width="100"> | <img src="images/Welsh_springer_spaniel_08203.jpg" width="200">

It is not difficult to find other dog breed pairs with minimal inter-class variation (for instance, Curly-Coated Retrievers and American Water Spaniels).  

Curly-Coated Retriever | American Water Spaniel
- | -
<img src="images/Curly-coated_retriever_03896.jpg" width="200"> | <img src="images/American_water_spaniel_00648.jpg" width="200">


Likewise, recall that labradors come in yellow, chocolate, and black.  Your vision-based algorithm will have to conquer this high intra-class variation to determine how to classify all of these different shades as the same breed.  

Yellow Labrador | Chocolate Labrador | Black Labrador
- | -
<img src="images/Labrador_retriever_06457.jpg" width="150"> | <img src="images/Labrador_retriever_06455.jpg" width="240"> | <img src="images/Labrador_retriever_06449.jpg" width="220">

We also mention that random chance presents an exceptionally low bar: setting aside the fact that the classes are slightly imabalanced, a random guess will provide a correct answer roughly 1 in 133 times, which corresponds to an accuracy of less than 1%.  

Remember that the practice is far ahead of the theory in deep learning.  Experiment with many different architectures, and trust your intuition.  And, of course, have fun! 

### Pre-process the Data

We rescale the images by dividing every pixel in every image by 255.


```python
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255
```

    100%|██████████| 6680/6680 [01:12<00:00, 92.49it/s] 
    100%|██████████| 835/835 [00:08<00:00, 100.54it/s]
    100%|██████████| 836/836 [00:08<00:00, 101.13it/s]
    

### (IMPLEMENTATION) Model Architecture

Create a CNN to classify dog breed.  At the end of your code cell block, summarize the layers of your model by executing the line:
    
        model.summary()

We have imported some Python modules to get you started, but feel free to import as many modules as you need.  If you end up getting stuck, here's a hint that specifies a model that trains relatively fast on CPU and attains >1% test accuracy in 5 epochs:

![Sample CNN](images/sample_cnn.png)
           
__Question 4:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  If you chose to use the hinted architecture above, describe why you think that CNN architecture should work well for the image classification task.

__Answer:__ 
I did start out with above mentioned model at least with the number of filters. I also thought of these filter numbers in each of the layers as appropriate since similar layers in the prior excercises had given me decent results. However I did reduce the number of layers since they were resulting in poorer results. I used pool size of 2 in max pooling layers as this would help halve the size of the subsequent layers and improve processing times. I added in two dense layers at the end along with dropout layers to avoid overfitting. In all of the convolutional and dense layers, except the last dense layer, I used the relu activation function since we are looking at pixel in the image in this problem. The last dense layer has 133 nodes, one node for each of the dog breed and used the softmax activation function to determine the probability of image beloging to each of the 133 dog breeds.


```python
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

model = Sequential()

### TODO: Define your architecture.
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
#model.add(GlobalAveragePooling2D(dim_ordering='default'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(133, activation='softmax'))
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 224, 224, 16)      208       
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 112, 112, 16)      0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 112, 112, 32)      2080      
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 56, 56, 32)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 56, 56, 64)        8256      
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 28, 28, 64)        0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 28, 28, 64)        0         
    _________________________________________________________________
    flatten_2 (Flatten)          (None, 50176)             0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 500)               25088500  
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 500)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 133)               66633     
    =================================================================
    Total params: 25,165,677
    Trainable params: 25,165,677
    Non-trainable params: 0
    _________________________________________________________________
    

### Compile the Model


```python
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
```

### (IMPLEMENTATION) Train the Model

Train your model in the code cell below.  Use model checkpointing to save the model that attains the best validation loss.

You are welcome to [augment the training data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html), but this is not a requirement. 


```python
from keras.callbacks import ModelCheckpoint  

### TODO: specify the number of epochs that you would like to use to train the model.

epochs = 10

### Do NOT modify the code below this line.

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)
```

    Train on 6680 samples, validate on 835 samples
    Epoch 1/10
    6660/6680 [============================>.] - ETA: 0s - loss: 4.8644 - acc: 0.0186Epoch 00001: val_loss improved from inf to 4.59748, saving model to saved_models/weights.best.from_scratch.hdf5
    6680/6680 [==============================] - 31s 5ms/step - loss: 4.8635 - acc: 0.0187 - val_loss: 4.5975 - val_acc: 0.0287
    Epoch 2/10
    6660/6680 [============================>.] - ETA: 0s - loss: 4.3756 - acc: 0.0565Epoch 00002: val_loss improved from 4.59748 to 4.26038, saving model to saved_models/weights.best.from_scratch.hdf5
    6680/6680 [==============================] - 30s 5ms/step - loss: 4.3757 - acc: 0.0563 - val_loss: 4.2604 - val_acc: 0.0647
    Epoch 3/10
    6660/6680 [============================>.] - ETA: 0s - loss: 3.7742 - acc: 0.1404Epoch 00003: val_loss improved from 4.26038 to 4.11764, saving model to saved_models/weights.best.from_scratch.hdf5
    6680/6680 [==============================] - 30s 5ms/step - loss: 3.7747 - acc: 0.1407 - val_loss: 4.1176 - val_acc: 0.0886
    Epoch 4/10
    6660/6680 [============================>.] - ETA: 0s - loss: 2.8060 - acc: 0.3252Epoch 00004: val_loss did not improve
    6680/6680 [==============================] - 30s 5ms/step - loss: 2.8048 - acc: 0.3250 - val_loss: 4.6643 - val_acc: 0.0886
    Epoch 5/10
    6660/6680 [============================>.] - ETA: 0s - loss: 1.6290 - acc: 0.5908Epoch 00005: val_loss did not improve
    6680/6680 [==============================] - 30s 5ms/step - loss: 1.6298 - acc: 0.5906 - val_loss: 5.1281 - val_acc: 0.1006
    Epoch 6/10
    6660/6680 [============================>.] - ETA: 0s - loss: 0.7666 - acc: 0.7992Epoch 00006: val_loss did not improve
    6680/6680 [==============================] - 30s 5ms/step - loss: 0.7653 - acc: 0.7996 - val_loss: 6.3567 - val_acc: 0.0826
    Epoch 7/10
    6660/6680 [============================>.] - ETA: 0s - loss: 0.3859 - acc: 0.8956Epoch 00007: val_loss did not improve
    6680/6680 [==============================] - 30s 5ms/step - loss: 0.3865 - acc: 0.8954 - val_loss: 7.1536 - val_acc: 0.0778
    Epoch 8/10
    6660/6680 [============================>.] - ETA: 0s - loss: 0.2522 - acc: 0.9311Epoch 00008: val_loss did not improve
    6680/6680 [==============================] - 30s 5ms/step - loss: 0.2521 - acc: 0.9310 - val_loss: 7.8638 - val_acc: 0.0647
    Epoch 9/10
    6660/6680 [============================>.] - ETA: 0s - loss: 0.1934 - acc: 0.9468Epoch 00009: val_loss did not improve
    6680/6680 [==============================] - 30s 5ms/step - loss: 0.1930 - acc: 0.9470 - val_loss: 8.1073 - val_acc: 0.0790
    Epoch 10/10
    6660/6680 [============================>.] - ETA: 0s - loss: 0.1518 - acc: 0.9613Epoch 00010: val_loss did not improve
    6680/6680 [==============================] - 30s 4ms/step - loss: 0.1521 - acc: 0.9611 - val_loss: 8.0485 - val_acc: 0.0695
    




    <keras.callbacks.History at 0x7f9719d72358>



### Load the Model with the Best Validation Loss


```python
model.load_weights('saved_models/weights.best.from_scratch.hdf5')
```

### Test the Model

Try out your model on the test dataset of dog images.  Ensure that your test accuracy is greater than 1%.


```python
# get index of predicted dog breed for each image in test set
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 10.6459%
    

---
<a id='step4'></a>
## Step 4: Use a CNN to Classify Dog Breeds

To reduce training time without sacrificing accuracy, we show you how to train a CNN using transfer learning.  In the following step, you will get a chance to use transfer learning to train your own CNN.

### Obtain Bottleneck Features


```python
bottleneck_features = np.load('/data/bottleneck_features/DogVGG16Data.npz')
train_VGG16 = bottleneck_features['train']
valid_VGG16 = bottleneck_features['valid']
test_VGG16 = bottleneck_features['test']
```

### Model Architecture

The model uses the the pre-trained VGG-16 model as a fixed feature extractor, where the last convolutional output of VGG-16 is fed as input to our model.  We only add a global average pooling layer and a fully connected layer, where the latter contains one node for each dog category and is equipped with a softmax.


```python
VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
VGG16_model.add(Dense(133, activation='softmax'))

VGG16_model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    global_average_pooling2d_1 ( (None, 512)               0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 133)               68229     
    =================================================================
    Total params: 68,229
    Trainable params: 68,229
    Non-trainable params: 0
    _________________________________________________________________
    

### Compile the Model


```python
VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```

### Train the Model


```python
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5', 
                               verbose=1, save_best_only=True)

VGG16_model.fit(train_VGG16, train_targets, 
          validation_data=(valid_VGG16, valid_targets),
          epochs=60, batch_size=20, callbacks=[checkpointer], verbose=1)
```

    Train on 6680 samples, validate on 835 samples
    Epoch 1/60
    6660/6680 [============================>.] - ETA: 0s - loss: 12.2477 - acc: 0.1153Epoch 00001: val_loss improved from inf to 10.70885, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 294us/step - loss: 12.2441 - acc: 0.1157 - val_loss: 10.7089 - val_acc: 0.2144
    Epoch 2/60
    6540/6680 [============================>.] - ETA: 0s - loss: 9.8490 - acc: 0.2957Epoch 00002: val_loss improved from 10.70885 to 9.82497, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 246us/step - loss: 9.8359 - acc: 0.2964 - val_loss: 9.8250 - val_acc: 0.2922
    Epoch 3/60
    6440/6680 [===========================>..] - ETA: 0s - loss: 9.2401 - acc: 0.3641Epoch 00003: val_loss improved from 9.82497 to 9.58216, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 243us/step - loss: 9.2150 - acc: 0.3659 - val_loss: 9.5822 - val_acc: 0.3126
    Epoch 4/60
    6580/6680 [============================>.] - ETA: 0s - loss: 9.0275 - acc: 0.3948Epoch 00004: val_loss improved from 9.58216 to 9.49187, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 246us/step - loss: 9.0283 - acc: 0.3951 - val_loss: 9.4919 - val_acc: 0.3401
    Epoch 5/60
    6640/6680 [============================>.] - ETA: 0s - loss: 8.9181 - acc: 0.4140Epoch 00005: val_loss improved from 9.49187 to 9.40123, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 244us/step - loss: 8.9182 - acc: 0.4139 - val_loss: 9.4012 - val_acc: 0.3545
    Epoch 6/60
    6500/6680 [============================>.] - ETA: 0s - loss: 8.7214 - acc: 0.4289Epoch 00006: val_loss improved from 9.40123 to 9.21989, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 246us/step - loss: 8.7282 - acc: 0.4287 - val_loss: 9.2199 - val_acc: 0.3557
    Epoch 7/60
    6640/6680 [============================>.] - ETA: 0s - loss: 8.5608 - acc: 0.4434Epoch 00007: val_loss improved from 9.21989 to 9.04633, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 245us/step - loss: 8.5517 - acc: 0.4439 - val_loss: 9.0463 - val_acc: 0.3689
    Epoch 8/60
    6460/6680 [============================>.] - ETA: 0s - loss: 8.4680 - acc: 0.4571Epoch 00008: val_loss improved from 9.04633 to 9.02663, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 243us/step - loss: 8.4542 - acc: 0.4573 - val_loss: 9.0266 - val_acc: 0.3737
    Epoch 9/60
    6660/6680 [============================>.] - ETA: 0s - loss: 8.3390 - acc: 0.4683Epoch 00009: val_loss improved from 9.02663 to 8.94437, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 241us/step - loss: 8.3430 - acc: 0.4681 - val_loss: 8.9444 - val_acc: 0.3856
    Epoch 10/60
    6480/6680 [============================>.] - ETA: 0s - loss: 8.1901 - acc: 0.4727Epoch 00010: val_loss improved from 8.94437 to 8.60743, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 243us/step - loss: 8.1798 - acc: 0.4735 - val_loss: 8.6074 - val_acc: 0.3952
    Epoch 11/60
    6620/6680 [============================>.] - ETA: 0s - loss: 7.9362 - acc: 0.4931Epoch 00011: val_loss improved from 8.60743 to 8.57547, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 244us/step - loss: 7.9324 - acc: 0.4934 - val_loss: 8.5755 - val_acc: 0.4060
    Epoch 12/60
    6480/6680 [============================>.] - ETA: 0s - loss: 7.8592 - acc: 0.5039Epoch 00012: val_loss improved from 8.57547 to 8.46237, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 237us/step - loss: 7.8856 - acc: 0.5018 - val_loss: 8.4624 - val_acc: 0.3988
    Epoch 13/60
    6440/6680 [===========================>..] - ETA: 0s - loss: 7.7715 - acc: 0.5067Epoch 00013: val_loss improved from 8.46237 to 8.40649, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 242us/step - loss: 7.7978 - acc: 0.5049 - val_loss: 8.4065 - val_acc: 0.4132
    Epoch 14/60
    6620/6680 [============================>.] - ETA: 0s - loss: 7.7587 - acc: 0.5098Epoch 00014: val_loss improved from 8.40649 to 8.30692, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 245us/step - loss: 7.7615 - acc: 0.5096 - val_loss: 8.3069 - val_acc: 0.4275
    Epoch 15/60
    6640/6680 [============================>.] - ETA: 0s - loss: 7.6683 - acc: 0.5107Epoch 00015: val_loss improved from 8.30692 to 8.23403, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 244us/step - loss: 7.6755 - acc: 0.5103 - val_loss: 8.2340 - val_acc: 0.4347
    Epoch 16/60
    6440/6680 [===========================>..] - ETA: 0s - loss: 7.5632 - acc: 0.5211Epoch 00016: val_loss did not improve
    6680/6680 [==============================] - 2s 241us/step - loss: 7.5918 - acc: 0.5195 - val_loss: 8.2996 - val_acc: 0.4192
    Epoch 17/60
    6460/6680 [============================>.] - ETA: 0s - loss: 7.5860 - acc: 0.5231Epoch 00017: val_loss improved from 8.23403 to 8.21154, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 244us/step - loss: 7.5582 - acc: 0.5247 - val_loss: 8.2115 - val_acc: 0.4287
    Epoch 18/60
    6660/6680 [============================>.] - ETA: 0s - loss: 7.5289 - acc: 0.5266Epoch 00018: val_loss did not improve
    6680/6680 [==============================] - 2s 242us/step - loss: 7.5402 - acc: 0.5259 - val_loss: 8.2421 - val_acc: 0.4275
    Epoch 19/60
    6460/6680 [============================>.] - ETA: 0s - loss: 7.5518 - acc: 0.5260Epoch 00019: val_loss improved from 8.21154 to 8.18331, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 243us/step - loss: 7.5299 - acc: 0.5275 - val_loss: 8.1833 - val_acc: 0.4299
    Epoch 20/60
    6460/6680 [============================>.] - ETA: 0s - loss: 7.5290 - acc: 0.5296Epoch 00020: val_loss did not improve
    6680/6680 [==============================] - 2s 241us/step - loss: 7.5128 - acc: 0.5307 - val_loss: 8.2822 - val_acc: 0.4251
    Epoch 21/60
    6600/6680 [============================>.] - ETA: 0s - loss: 7.4973 - acc: 0.5298Epoch 00021: val_loss did not improve
    6680/6680 [==============================] - 2s 243us/step - loss: 7.5113 - acc: 0.5290 - val_loss: 8.2701 - val_acc: 0.4311
    Epoch 22/60
    6660/6680 [============================>.] - ETA: 0s - loss: 7.4445 - acc: 0.5311Epoch 00022: val_loss did not improve
    6680/6680 [==============================] - 2s 242us/step - loss: 7.4464 - acc: 0.5310 - val_loss: 8.2725 - val_acc: 0.4275
    Epoch 23/60
    6620/6680 [============================>.] - ETA: 0s - loss: 7.4091 - acc: 0.5343Epoch 00023: val_loss did not improve
    6680/6680 [==============================] - 2s 243us/step - loss: 7.4101 - acc: 0.5343 - val_loss: 8.2545 - val_acc: 0.4311
    Epoch 24/60
    6640/6680 [============================>.] - ETA: 0s - loss: 7.3734 - acc: 0.5364Epoch 00024: val_loss did not improve
    6680/6680 [==============================] - 2s 243us/step - loss: 7.3728 - acc: 0.5364 - val_loss: 8.2190 - val_acc: 0.4168
    Epoch 25/60
    6640/6680 [============================>.] - ETA: 0s - loss: 7.2437 - acc: 0.5410Epoch 00025: val_loss improved from 8.18331 to 8.02106, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 243us/step - loss: 7.2510 - acc: 0.5406 - val_loss: 8.0211 - val_acc: 0.4323
    Epoch 26/60
    6640/6680 [============================>.] - ETA: 0s - loss: 7.2023 - acc: 0.5482Epoch 00026: val_loss improved from 8.02106 to 7.98235, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 244us/step - loss: 7.1930 - acc: 0.5488 - val_loss: 7.9823 - val_acc: 0.4371
    Epoch 27/60
    6660/6680 [============================>.] - ETA: 0s - loss: 7.1607 - acc: 0.5520Epoch 00027: val_loss did not improve
    6680/6680 [==============================] - 2s 243us/step - loss: 7.1756 - acc: 0.5509 - val_loss: 8.0126 - val_acc: 0.4323
    Epoch 28/60
    6600/6680 [============================>.] - ETA: 0s - loss: 7.1646 - acc: 0.5527Epoch 00028: val_loss improved from 7.98235 to 7.95055, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 245us/step - loss: 7.1706 - acc: 0.5522 - val_loss: 7.9505 - val_acc: 0.4467
    Epoch 29/60
    6620/6680 [============================>.] - ETA: 0s - loss: 7.1558 - acc: 0.5533Epoch 00029: val_loss did not improve
    6680/6680 [==============================] - 2s 243us/step - loss: 7.1569 - acc: 0.5531 - val_loss: 8.0814 - val_acc: 0.4395
    Epoch 30/60
    6580/6680 [============================>.] - ETA: 0s - loss: 7.1144 - acc: 0.5555Epoch 00030: val_loss improved from 7.95055 to 7.86744, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 246us/step - loss: 7.1214 - acc: 0.5551 - val_loss: 7.8674 - val_acc: 0.4467
    Epoch 31/60
    6580/6680 [============================>.] - ETA: 0s - loss: 7.0943 - acc: 0.5558Epoch 00031: val_loss did not improve
    6680/6680 [==============================] - 2s 244us/step - loss: 7.0895 - acc: 0.5561 - val_loss: 7.9516 - val_acc: 0.4479
    Epoch 32/60
    6500/6680 [============================>.] - ETA: 0s - loss: 7.0842 - acc: 0.5583Epoch 00032: val_loss did not improve
    6680/6680 [==============================] - 2s 244us/step - loss: 7.0695 - acc: 0.5593 - val_loss: 7.9309 - val_acc: 0.4491
    Epoch 33/60
    6560/6680 [============================>.] - ETA: 0s - loss: 7.0481 - acc: 0.5613Epoch 00033: val_loss improved from 7.86744 to 7.83809, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 245us/step - loss: 7.0663 - acc: 0.5602 - val_loss: 7.8381 - val_acc: 0.4575
    Epoch 34/60
    6640/6680 [============================>.] - ETA: 0s - loss: 7.0665 - acc: 0.5602Epoch 00034: val_loss did not improve
    6680/6680 [==============================] - 2s 242us/step - loss: 7.0652 - acc: 0.5603 - val_loss: 7.8950 - val_acc: 0.4551
    Epoch 35/60
    6660/6680 [============================>.] - ETA: 0s - loss: 7.0663 - acc: 0.5607Epoch 00035: val_loss did not improve
    6680/6680 [==============================] - 2s 241us/step - loss: 7.0644 - acc: 0.5608 - val_loss: 7.9139 - val_acc: 0.4551
    Epoch 36/60
    6440/6680 [===========================>..] - ETA: 0s - loss: 7.0688 - acc: 0.5607Epoch 00036: val_loss did not improve
    6680/6680 [==============================] - 2s 242us/step - loss: 7.0634 - acc: 0.5611 - val_loss: 7.9583 - val_acc: 0.4563
    Epoch 37/60
    6580/6680 [============================>.] - ETA: 0s - loss: 7.0610 - acc: 0.5616Epoch 00037: val_loss did not improve
    6680/6680 [==============================] - 2s 243us/step - loss: 7.0615 - acc: 0.5615 - val_loss: 7.8608 - val_acc: 0.4599
    Epoch 38/60
    6440/6680 [===========================>..] - ETA: 0s - loss: 7.0923 - acc: 0.5596Epoch 00038: val_loss did not improve
    6680/6680 [==============================] - 2s 242us/step - loss: 7.0619 - acc: 0.5615 - val_loss: 7.8470 - val_acc: 0.4599
    Epoch 39/60
    6620/6680 [============================>.] - ETA: 0s - loss: 7.0418 - acc: 0.5619Epoch 00039: val_loss improved from 7.83809 to 7.82195, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 245us/step - loss: 7.0606 - acc: 0.5608 - val_loss: 7.8219 - val_acc: 0.4611
    Epoch 40/60
    6440/6680 [===========================>..] - ETA: 0s - loss: 7.0587 - acc: 0.5618Epoch 00040: val_loss did not improve
    6680/6680 [==============================] - 2s 242us/step - loss: 7.0616 - acc: 0.5615 - val_loss: 7.8466 - val_acc: 0.4635
    Epoch 41/60
    6560/6680 [============================>.] - ETA: 0s - loss: 7.0811 - acc: 0.5605Epoch 00041: val_loss did not improve
    6680/6680 [==============================] - 2s 244us/step - loss: 7.0601 - acc: 0.5618 - val_loss: 7.9122 - val_acc: 0.4563
    Epoch 42/60
    6560/6680 [============================>.] - ETA: 0s - loss: 7.0581 - acc: 0.5605Epoch 00042: val_loss did not improve
    6680/6680 [==============================] - 2s 244us/step - loss: 7.0522 - acc: 0.5608 - val_loss: 7.8610 - val_acc: 0.4563
    Epoch 43/60
    6600/6680 [============================>.] - ETA: 0s - loss: 6.9805 - acc: 0.5641Epoch 00043: val_loss did not improve
    6680/6680 [==============================] - 2s 244us/step - loss: 6.9889 - acc: 0.5635 - val_loss: 7.8468 - val_acc: 0.4707
    Epoch 44/60
    6660/6680 [============================>.] - ETA: 0s - loss: 6.9598 - acc: 0.5656Epoch 00044: val_loss improved from 7.82195 to 7.78889, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 244us/step - loss: 6.9655 - acc: 0.5653 - val_loss: 7.7889 - val_acc: 0.4731
    Epoch 45/60
    6440/6680 [===========================>..] - ETA: 0s - loss: 6.8683 - acc: 0.5660Epoch 00045: val_loss improved from 7.78889 to 7.75296, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 243us/step - loss: 6.8774 - acc: 0.5657 - val_loss: 7.7530 - val_acc: 0.4635
    Epoch 46/60
    6620/6680 [============================>.] - ETA: 0s - loss: 6.7794 - acc: 0.5725Epoch 00046: val_loss improved from 7.75296 to 7.74380, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 245us/step - loss: 6.7891 - acc: 0.5719 - val_loss: 7.7438 - val_acc: 0.4683
    Epoch 47/60
    6640/6680 [============================>.] - ETA: 0s - loss: 6.7225 - acc: 0.5786Epoch 00047: val_loss improved from 7.74380 to 7.71658, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 245us/step - loss: 6.7235 - acc: 0.5784 - val_loss: 7.7166 - val_acc: 0.4695
    Epoch 48/60
    6560/6680 [============================>.] - ETA: 0s - loss: 6.6914 - acc: 0.5809Epoch 00048: val_loss improved from 7.71658 to 7.59487, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 245us/step - loss: 6.6871 - acc: 0.5813 - val_loss: 7.5949 - val_acc: 0.4731
    Epoch 49/60
    6620/6680 [============================>.] - ETA: 0s - loss: 6.5936 - acc: 0.5847Epoch 00049: val_loss improved from 7.59487 to 7.52395, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 244us/step - loss: 6.5972 - acc: 0.5846 - val_loss: 7.5240 - val_acc: 0.4754
    Epoch 50/60
    6660/6680 [============================>.] - ETA: 0s - loss: 6.5156 - acc: 0.5910Epoch 00050: val_loss improved from 7.52395 to 7.47643, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 243us/step - loss: 6.5106 - acc: 0.5913 - val_loss: 7.4764 - val_acc: 0.4850
    Epoch 51/60
    6620/6680 [============================>.] - ETA: 0s - loss: 6.4834 - acc: 0.5923Epoch 00051: val_loss improved from 7.47643 to 7.41760, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 244us/step - loss: 6.4785 - acc: 0.5925 - val_loss: 7.4176 - val_acc: 0.4886
    Epoch 52/60
    6660/6680 [============================>.] - ETA: 0s - loss: 6.4563 - acc: 0.5949Epoch 00052: val_loss improved from 7.41760 to 7.36099, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 243us/step - loss: 6.4515 - acc: 0.5952 - val_loss: 7.3610 - val_acc: 0.4850
    Epoch 53/60
    6440/6680 [===========================>..] - ETA: 0s - loss: 6.4535 - acc: 0.5966Epoch 00053: val_loss did not improve
    6680/6680 [==============================] - 2s 241us/step - loss: 6.4424 - acc: 0.5972 - val_loss: 7.3781 - val_acc: 0.4970
    Epoch 54/60
    6600/6680 [============================>.] - ETA: 0s - loss: 6.4016 - acc: 0.5974Epoch 00054: val_loss improved from 7.36099 to 7.31766, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 244us/step - loss: 6.4070 - acc: 0.5972 - val_loss: 7.3177 - val_acc: 0.4886
    Epoch 55/60
    6640/6680 [============================>.] - ETA: 0s - loss: 6.3649 - acc: 0.6002Epoch 00055: val_loss did not improve
    6680/6680 [==============================] - 2s 242us/step - loss: 6.3606 - acc: 0.6004 - val_loss: 7.3207 - val_acc: 0.4898
    Epoch 56/60
    6600/6680 [============================>.] - ETA: 0s - loss: 6.3734 - acc: 0.6000Epoch 00056: val_loss improved from 7.31766 to 7.27047, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 245us/step - loss: 6.3540 - acc: 0.6012 - val_loss: 7.2705 - val_acc: 0.4886
    Epoch 57/60
    6460/6680 [============================>.] - ETA: 0s - loss: 6.3057 - acc: 0.6059Epoch 00057: val_loss did not improve
    6680/6680 [==============================] - 2s 242us/step - loss: 6.3334 - acc: 0.6042 - val_loss: 7.3215 - val_acc: 0.4922
    Epoch 58/60
    6660/6680 [============================>.] - ETA: 0s - loss: 6.3144 - acc: 0.6038Epoch 00058: val_loss did not improve
    6680/6680 [==============================] - 2s 242us/step - loss: 6.3221 - acc: 0.6031 - val_loss: 7.4054 - val_acc: 0.4719
    Epoch 59/60
    6560/6680 [============================>.] - ETA: 0s - loss: 6.2411 - acc: 0.6020Epoch 00059: val_loss improved from 7.27047 to 7.20366, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 245us/step - loss: 6.2262 - acc: 0.6027 - val_loss: 7.2037 - val_acc: 0.4898
    Epoch 60/60
    6640/6680 [============================>.] - ETA: 0s - loss: 6.1041 - acc: 0.6127Epoch 00060: val_loss improved from 7.20366 to 7.12338, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 244us/step - loss: 6.1241 - acc: 0.6114 - val_loss: 7.1234 - val_acc: 0.4898
    




    <keras.callbacks.History at 0x7f9709e50198>



### Load the Model with the Best Validation Loss


```python
VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')
```

### Test the Model

Now, we can use the CNN to test how well it identifies breed within our test dataset of dog images.  We print the test accuracy below.


```python
# get index of predicted dog breed for each image in test set
VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]

# report test accuracy
test_accuracy = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 50.1196%
    

### Predict Dog Breed with the Model


```python
from extract_bottleneck_features import *

def VGG16_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_VGG16(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = VGG16_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]
```

---
<a id='step5'></a>
## Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)

You will now use transfer learning to create a CNN that can identify dog breed from images.  Your CNN must attain at least 60% accuracy on the test set.

In Step 4, we used transfer learning to create a CNN using VGG-16 bottleneck features.  In this section, you must use the bottleneck features from a different pre-trained model.  To make things easier for you, we have pre-computed the features for all of the networks that are currently available in Keras.  These are already in the workspace, at /data/bottleneck_features.  If you wish to download them on a different machine, they can be found at:
- [VGG-19](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz) bottleneck features
- [ResNet-50](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) bottleneck features
- [Inception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz) bottleneck features
- [Xception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz) bottleneck features

The files are encoded as such:

    Dog{network}Data.npz
    
where `{network}`, in the above filename, can be one of `VGG19`, `Resnet50`, `InceptionV3`, or `Xception`.  

The above architectures are downloaded and stored for you in the `/data/bottleneck_features/` folder.

This means the following will be in the `/data/bottleneck_features/` folder:

`DogVGG19Data.npz`
`DogResnet50Data.npz`
`DogInceptionV3Data.npz`
`DogXceptionData.npz`



### (IMPLEMENTATION) Obtain Bottleneck Features

In the code block below, extract the bottleneck features corresponding to the train, test, and validation sets by running the following:

    bottleneck_features = np.load('/data/bottleneck_features/Dog{network}Data.npz')
    train_{network} = bottleneck_features['train']
    valid_{network} = bottleneck_features['valid']
    test_{network} = bottleneck_features['test']


```python
### TODO: Obtain bottleneck features from another pre-trained CNN.
bottleneck_features = np.load('/data/bottleneck_features/DogResnet50Data.npz')
train_ResNet50 = bottleneck_features['train']
valid_ResNet50 = bottleneck_features['valid']
test_ResNet50 = bottleneck_features['test']
```

### (IMPLEMENTATION) Model Architecture

Create a CNN to classify dog breed.  At the end of your code cell block, summarize the layers of your model by executing the line:
    
        <your model's name>.summary()
   
__Question 5:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  Describe why you think the architecture is suitable for the current problem.

__Answer:__ 
I used the ResNet-50 algorithm to predict the breed the dog in this app. Since the new dataset is going to be smaller than the originial ResNet-50 dataset but the categories in the new dataset are also present in the original dataset we only need to add a final layer depending on the new dataset. Here we have 133 breed of dogs that's why I added a final dense layer with 133 nodes and Softmax activation function since we need probability of image belonging to breed of dog.


```python
### TODO: Define your architecture.
Resnet50_model = Sequential()
Resnet50_model.add(GlobalAveragePooling2D(input_shape=train_ResNet50.shape[1:]))
Resnet50_model.add(Dense(133, activation='softmax'))

Resnet50_model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    global_average_pooling2d_2 ( (None, 2048)              0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 133)               272517    
    =================================================================
    Total params: 272,517
    Trainable params: 272,517
    Non-trainable params: 0
    _________________________________________________________________
    

### (IMPLEMENTATION) Compile the Model


```python
### TODO: Compile the model.
Resnet50_model.compile(loss='categorical_crossentropy', 
                       optimizer='rmsprop', metrics=['accuracy'])
```

### (IMPLEMENTATION) Train the Model

Train your model in the code cell below.  Use model checkpointing to save the model that attains the best validation loss.  

You are welcome to [augment the training data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html), but this is not a requirement. 


```python
### TODO: Train the model.
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.Resnet50.hdf5',
                              verbose=1, save_best_only=True)

Resnet50_model.fit(train_ResNet50, train_targets,
                  validation_data=(valid_ResNet50, valid_targets),
                  epochs=5, batch_size=20, callbacks=[checkpointer],
                  verbose=1)
```

    Train on 6680 samples, validate on 835 samples
    Epoch 1/5
    6500/6680 [============================>.] - ETA: 0s - loss: 1.6358 - acc: 0.5983Epoch 00001: val_loss improved from inf to 0.79978, saving model to saved_models/weights.best.Resnet50.hdf5
    6680/6680 [==============================] - 2s 265us/step - loss: 1.6130 - acc: 0.6027 - val_loss: 0.7998 - val_acc: 0.7485
    Epoch 2/5
    6500/6680 [============================>.] - ETA: 0s - loss: 0.4358 - acc: 0.8620Epoch 00002: val_loss improved from 0.79978 to 0.66511, saving model to saved_models/weights.best.Resnet50.hdf5
    6680/6680 [==============================] - 1s 218us/step - loss: 0.4375 - acc: 0.8618 - val_loss: 0.6651 - val_acc: 0.7976
    Epoch 3/5
    6520/6680 [============================>.] - ETA: 0s - loss: 0.2600 - acc: 0.9127Epoch 00003: val_loss did not improve
    6680/6680 [==============================] - 1s 216us/step - loss: 0.2580 - acc: 0.9132 - val_loss: 0.6684 - val_acc: 0.8096
    Epoch 4/5
    6500/6680 [============================>.] - ETA: 0s - loss: 0.1774 - acc: 0.9469Epoch 00004: val_loss did not improve
    6680/6680 [==============================] - 1s 216us/step - loss: 0.1751 - acc: 0.9478 - val_loss: 0.6771 - val_acc: 0.8048
    Epoch 5/5
    6480/6680 [============================>.] - ETA: 0s - loss: 0.1199 - acc: 0.9634Epoch 00005: val_loss did not improve
    6680/6680 [==============================] - 1s 217us/step - loss: 0.1205 - acc: 0.9632 - val_loss: 0.6716 - val_acc: 0.8120
    




    <keras.callbacks.History at 0x7f970a02f9b0>



### (IMPLEMENTATION) Load the Model with the Best Validation Loss


```python
### TODO: Load the model weights with the best validation loss.
Resnet50_model.load_weights('saved_models/weights.best.Resnet50.hdf5')
```

### (IMPLEMENTATION) Test the Model

Try out your model on the test dataset of dog images. Ensure that your test accuracy is greater than 60%.


```python
### TODO: Calculate classification accuracy on the test dataset.
Resnet_50_predictions = [np.argmax(Resnet50_model.predict(
    np.expand_dims(feature, axis=0))) for feature in test_ResNet50]

# report test accuracy
test_accuracy = 100*np.sum(np.array(Resnet_50_predictions)==np.argmax(
    test_targets, axis=1))/len(Resnet_50_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 79.3062%
    

### (IMPLEMENTATION) Predict Dog Breed with the Model

Write a function that takes an image path as input and returns the dog breed (`Affenpinscher`, `Afghan_hound`, etc) that is predicted by your model.  

Similar to the analogous function in Step 5, your function should have three steps:
1. Extract the bottleneck features corresponding to the chosen CNN model.
2. Supply the bottleneck features as input to the model to return the predicted vector.  Note that the argmax of this prediction vector gives the index of the predicted dog breed.
3. Use the `dog_names` array defined in Step 0 of this notebook to return the corresponding breed.

The functions to extract the bottleneck features can be found in `extract_bottleneck_features.py`, and they have been imported in an earlier code cell.  To obtain the bottleneck features corresponding to your chosen CNN architecture, you need to use the function

    extract_{network}
    
where `{network}`, in the above filename, should be one of `VGG19`, `Resnet50`, `InceptionV3`, or `Xception`.


```python
### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.
from extract_bottleneck_features import *

def Resnet50_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = Resnet50_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]
```

---
<a id='step6'></a>
## Step 6: Write your Algorithm

Write an algorithm that accepts a file path to an image and first determines whether the image contains a human, dog, or neither.  Then,
- if a __dog__ is detected in the image, return the predicted breed.
- if a __human__ is detected in the image, return the resembling dog breed.
- if __neither__ is detected in the image, provide output that indicates an error.

You are welcome to write your own functions for detecting humans and dogs in images, but feel free to use the `face_detector` and `dog_detector` functions developed above.  You are __required__ to use your CNN from Step 5 to predict dog breed.  

Some sample output for our algorithm is provided below, but feel free to design your own user experience!

![Sample Human Output](images/sample_human_output.png)


### (IMPLEMENTATION) Write your Algorithm


```python
### TODO: Write your algorithm.
### Feel free to use as many code cells as needed.
def identify_dog_breed(img_path):
    breed = Resnet50_predict_breed(img_path)
    
    img = cv2.imread(img_path)
    # convert BGR image to RGB for plotting
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # display the image, along with bounding box
    plt.imshow(cv_rgb)
    plt.show()
    
    if dog_detector(img_path):
        print("The dog detected in image seem to of breed: "+ str(breed))
    elif face_detector(img_path):
        print("A human was detected in image and is similar dog breed: "+
             str(breed))
    else:
        print("Nothing that looks like dog was detected in the image :(")
```

---
<a id='step7'></a>
## Step 7: Test Your Algorithm

In this section, you will take your new algorithm for a spin!  What kind of dog does the algorithm think that __you__ look like?  If you have a dog, does it predict your dog's breed accurately?  If you have a cat, does it mistakenly think that your cat is a dog?

### (IMPLEMENTATION) Test Your Algorithm on Sample Images!

Test your algorithm at least six images on your computer.  Feel free to use any images you like.  Use at least two human and two dog images.  

__Question 6:__ Is the output better than you expected :) ?  Or worse :( ?  Provide at least three possible points of improvement for your algorithm.

__Answer:__ 
After several runs I observed the prediction of the algorithm is around 80%. This is great prediction compared to prediction of the CNN I created above from scratch. I do think there is lot of room for improvement since ResNet-50 architecture can achieve predictions in 90s.

1. In the second photo, the algorithm predicts the image to be that of Alaskan Malamute however that is actually an image of a Husky. I haven't actually seen all the pictures in the dataset but I believe it would definitely be helpful if the dataset was balanced with images of all the dog breed and not be only towards some popular breeds.

2. It would also be helpful if the size of the dataset were increased as neural networks do tend to get better with more data. This may also address the problem mentioned in the previous point.

3. I haven't tested any of the other algorithms that were available for transfer learning above but it may be a worth while exercise to test those to see if any of them can delivery better accuracy and/or speed as both of these factors are taken for granted in today's apps.



```python
## TODO: Execute your algorithm from Step 6 on
## at least 6 images on your computer.
## Feel free to use as many code cells as needed.
for pic in sorted(glob("app_test_images/*")):
    identify_dog_breed(pic)
```

    Downloading data from https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
    94658560/94653016 [==============================] - 1s 0us/step
    


![png](https://github.com/RajNikhil/dataBlog/blob/gh-pages/assets/DogApp/dogApp_67_1.png?raw=true)


    The dog detected in image seem to of breed: in/076.Golden_retriever
    


![png](https://github.com/RajNikhil/dataBlog/blob/gh-pages/assets/DogApp/dogApp_67_3.png?raw=true)


    The dog detected in image seem to of breed: in/005.Alaskan_malamute
    


![png](https://github.com/RajNikhil/dataBlog/blob/gh-pages/assets/DogApp/dogApp_67_5.png?raw=true)


    Nothing that looks like dog was detected in the image :(
    


![png](https://github.com/RajNikhil/dataBlog/blob/gh-pages/assets/DogApp/dogApp_67_7.png?raw=true)


    A human was detected in image and is similar dog breed: in/016.Beagle
    


![png](https://github.com/RajNikhil/dataBlog/blob/gh-pages/assets/DogApp/dogApp_67_9.png?raw=true)


    Nothing that looks like dog was detected in the image :(
    


![png](https://github.com/RajNikhil/dataBlog/blob/gh-pages/assets/DogApp/dogApp_67_11.png?raw=true)


    A human was detected in image and is similar dog breed: in/026.Black_russian_terrier
    

```python

```
