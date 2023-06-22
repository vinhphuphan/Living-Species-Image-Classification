# [Python]Living Species Image Classification

### 1. Introduction
In recent years, image classification has become an increasingly important task in the field of computer vision. The ability to automatically classify images has a wide range of applications, from recognizing objects in photographs to identifying medical conditions in diagnostic images.

In this project, I will be working with a dataset of living things, which includes images of animals and plants. The dataset contains a set of coarse-grained labels, consisting of 8 different categories.

<p align="center"><img style="align: center;" src="https://github.com/vinhphuphan/ImageClassification/blob/main/images/example.png?raw=true" width=600></p>

### 2. Data Preprocessing
-**Bad proportion of data** : Dataset contains **102,465** and **96,383** images for train and validataion data respectively. Thus, I decided to move all images from validation folder into train folder and merge the val.csv to train.csv. After that, I use ImageDataGenerator of tensorflow library to reallocate the data 80% for train set and 20% for validataion set.

-**Unbalanced Design Data** : Regard of Coarse data, the design of dataset is unbalanced.While it has a large amount of **Magnoliopsida, Insecta,  Aves** images, there are very few images of **Reptilia, Pinopsida, Arachnida, Mammalia, Liliopsida**. To tackle this issue, instead of using accuracy alone, F1 score provide a more comprehensive understanding of the model's performance across different classes. Beside that,  **resampling** and **data augmentation** are efficient techniques to create more images for Reptilia, Pinopsida, Arachnida, Mammalia class.

**Before resampling**

<p align="center"><img style="align: center;" src="https://github.com/vinhphuphan/ImageClassification/blob/main/images/distribution1.png?raw=true" width=400></p>
<h4 align="center">Figure 1. The distribution of dataset before resampling</h4>

**After resampling**

<p align="center"><img style="align: center;" src="https://github.com/vinhphuphan/ImageClassification/blob/main/images/distribution2.png?raw=true" width=400></p>
<h4 align="center">Figure 2. The distribution of dataset after resampling</h4>

-**Label Encoding** :  The labels in csv files give the information of image. They are in form of numerical variable which is from 0 to 7. There are several drawbacks when using this scale of labels such as creating bias in the model or ordinality assumptions. Therefore, labels are encoded into the vector of 0 and 1 (onehot encoding). When ImageDataGenerator is used to load data, it will automatically encode the labels.

<p align="center"><img style="align: center;" src="https://github.com/vinhphuphan/ImageClassification/blob/main/images/label_encoding.png?raw=true" width=400></p>
<h4 align="center">Figure 3. Label Encoding</h4>

### 3. Conventional ML Model
The task is **image classification** for color images, so this is **supervised, categorical (classifcation), batch problem**. Therefore, the machine learning classifiers which can be use in this case are Logistic Regression, Decision Trees, Random Forest, Support Vector Machines (SVM) , K-Nearest Neighbors (KNN). In this project, I decide to use **Support Vector Machines (SVM)** and **Random Forest** for the task. 
The final model that produced the best-performing predictions for the Kaggle submission (accuracy 48%) was an **Random Forest** with `max_depth` = 60, `n_estimators` = 1000.
<p align="center"><img style="align: center;" src="https://github.com/vinhphuphan/ImageClassification/blob/main/images/Convolutional_ML.jpg?raw=true" width=600></p>
<h4 align="center">Figure 4. The performance of conventional models</h4>

### 4. Deep Learning Model
In the Deep Learning section, I have chosen transfer learning technique to take advantage of pretrained model.  Instead of training a model from scratch on a new task, transfer learning leverages the pre-existing knowledge captured by a model trained on a large dataset or a related task. The pretrained models for this section are **EfficientNetV2S** and **MobileNetV2**. The best model I use to submit prediction on Kaggle is the model based on pretrained **EfficientNetV2S**. The best accuracy for public test set on Kaggle is **83%** .

Firstly, I freeze all of the base model's layers and then train the model, the weights of the pre-trained network were not updated during training. After that, in the fine-tuning step, I unfreeze base model's layers to adapt these specialized features to work with the new dataset, rather than overwrite the generic learning.

#### Why use EfficientNetV2S?

EfficientNetV2S has achieved state-of-the-art performance on various benchmark datasets, including ImageNet, which demonstrates its effectiveness in image classification tasks.

EfficientNet models are designed to achieve high accuracy while maintaining efficiency in terms of model size and computational requirements.

#### Problems when trainning the model : 
**Overfitting** : When I first tried MobileNetV2 model, after creating and freezing base model, the architecture of model was only base model layer and output layer for prediction. The result was that there is a big gap between trainning and validation accuracy. The trainning accuracy continously increased until 80% , but the val_acc reach the peak at 52% at 3rd epoch. To handle this issue, Dropout layer with dropout rate at 0.2 and L2  kernel_regularizer are added to model to reduce the overfitting problem. 

<p align="center"><img style="align: center;" src="https://github.com/vinhphuphan/ImageClassification/blob/main/images/before_fix_overfit.jpg?raw=true" width=500></p>
<h4 align="center">Figure 5. The big gap between train_accuracy and val_accuracy</h4>

<p align="center"><img style="align: center;" src="https://github.com/vinhphuphan/ImageClassification/blob/main/images/after_fix_overfit.jpg?raw=true" width=400></p>
<h4 align="center">Figure 6. The gap between train_accuracy and val_accuracy becomes smaller</h4>

**Learning rate divergency** : According to the figure below, from epoch 5, the trainning loss and validation loss value become diverse significantly. At the beginning, I tried learning rate at 0.001, the situation did not improve. Therefore, I adjusted it to 0.0001 and applied L2 regularization layer to address the issue. Then, the model has been improved considerably. 
<p align="center"><img style="align: center;" src="https://github.com/vinhphuphan/ImageClassification/blob/main/images/learning_rate.jpg?raw=true" width=400></p>
<h4 align="center">Figure 7 . Learning rate divergency problem with the deep learning model</h4>

### 5. Discussion

The deep learning model I have build perform well with image classification task for coarse-grained data which has 8 classes when model reach 80% accuracy.

The performance comparison between the final conventional ML and deep learning models reveals that the deep learning model outperformed the conventional ML model by a significant margin. The deep learning model achieved an accuracy of 83% on the public test set, surpassing the conventional ML model by 36%. This indicates that the deep learning model was able to capture more complex patterns and representations in the data, leading to improved classification accuracy.

When inspecting the data, it is crucial to analyze the characteristics and quality of the dataset. Understanding the dataset can provide insights into the challenges faced during modeling and help identify areas for improvement. Factors such as class imbalance, data distribution, and potential biases need to be carefully examined to ensure fair and unbiased model performance.

In terms of resource requirements, deep learning models often demand more computational resources compared to conventional ML models. Deep learning models typically require powerful GPUs and longer training times due to their complex architectures and the need to process large amounts of data.

Overall, the implementation of deep learning models for image classification tasks provides promising results, showcasing the capability of neural networks to learn intricate patterns and achieve high accuracy. However, addressing overfitting and ensuring model generalization are important considerations for future improvements. Additionally, optimizing computational resources and adapting models to specific dataset characteristics can further enhance the performance and efficiency of the models.

