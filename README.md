# Tomato-leaf-disease


This model predicts Tomato diseases.This model prediction accuracy is `98%`(test data)
## Data preprocessing
The data_augmentation model uses for dataset preprocessing.
* Flip (horizontal)
* Roation (0.2)
* Zoom (0.2)
* Height (0.2)
* Width (0.2)
* Rescaling (0-255)-(0-1)

## Images
![screenshot](https://github.com/HSAkash/Tomato-leaf-disease/raw/main/related_images/original.png)

## After augmetation
![augmetation_image](https://github.com/HSAkash/Tomato-leaf-disease/raw/main/related_images/augmented_image.png)

## CNN model
CNN model is used to train the network.<br>
Layer parameters:<br>
* Input size (112,112,3)
* Conv2D with 32 filters
* MaxPool2D (pool_size=2)
* Conv2D with 16 filters
* MaxPool2D (pool_size=2)
* Conv2D with 32 filters
* MaxPool2D (pool_size=2)
* GlobalAveragePooling2D
* Output

### Compile model
Compile the model with the following options:
* Loss function (categorical_crossentropy)
* optimizer (Adam lr=0.001)
* metrics (accuracy)

### Fit model
Then fit the model with the following parameters:
* train_data
* epochs (3000)
* validation_data (test data)
* validation_steps (len of test_data)


#### 0.Input image
![layer_0](https://github.com/HSAkash/Tomato-leaf-disease/raw/main/related_images/test_image.png)
#### 1.Conv2D with 32 filters (output)
![layer_0](https://github.com/HSAkash/Tomato-leaf-disease/raw/main/related_images/layer_0.png)
#### 2.MaxPool2D (pool_size=2) (output)
![layer_1](https://github.com/HSAkash/Tomato-leaf-disease/raw/main/related_images/layer_1.png)
#### 3.Conv2D with 16 filters (output)
![layer_2](https://github.com/HSAkash/Tomato-leaf-disease/raw/main/related_images/layer_2.png)
#### 4.MaxPool2D (pool_size=2) (output)
![layer_3](https://github.com/HSAkash/Tomato-leaf-disease/raw/main/related_images/layer_3.png)
#### 5.Conv2D with 32 filters (output)
![layer_4](https://github.com/HSAkash/Tomato-leaf-disease/raw/main/related_images/layer_4.png)
#### 6.MaxPool2D (pool_size=2) (output)
![layer_5](https://github.com/HSAkash/Tomato-leaf-disease/raw/main/related_images/layer_5.png)
#### 7.Prediction (final output)
![prediction](https://github.com/HSAkash/Tomato-leaf-disease/raw/main/related_images/predict.png)

## Confusion Matrix
![confusion_matrix](https://github.com/HSAkash/Tomato-leaf-disease/raw/main/related_images/confusion_matrix.png)




# Requirements
* matplotlib 3.5.2
* numpy 1.23.1
* Pillow 9.2.0
* scikit-learn 1.1.1
* scipy 1.8.1
* tensorflow 2.9.1


# Demo
Here is how to run the tomato disease program using the following command line.<br>
```bash
python tomato.py
```

# Directories
<pre>
│  tomato.py
│
├─env
├─tomato
|   ├─train
|   ├─val
|   
</pre>

# Reference
* [Tensorflow](https://www.tensorflow.org/)
* [data_augmentation](https://www.tensorflow.org/tutorials/images/data_augmentation)

# Links (dataset & code)
* [Tomato leaf disease detection](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf)
* [Code](https://www.kaggle.com/code/hsakash/tomato-leaf-disease-val-data-accuracy-98)


# Author
HSAkash
* [Facebook](https://www.facebook.com/hemel.akash.7/)
* [Kaggle](https://www.kaggle.com/hsakash)
