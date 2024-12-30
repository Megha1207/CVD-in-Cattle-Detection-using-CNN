# CVD-in-Cattle-Detection-using-CNN

# Problem Statement 
The goal of this project is to develop an image classification model aimed at detecting cardiovascular diseases (CVD) in cattle through the analysis of retina images. CVD detection in cattle is crucial because cardiovascular diseases can significantly impact the health and productivity of livestock, leading to economic losses for farmers. Early and accurate detection of CVD allows for timely intervention and management, improving animal welfare and reducing costs associated with undiagnosed conditions. The project involves identifying or generating a relevant dataset of retina images, followed by preprocessing steps such as resizing and augmentation to prepare the data for model training. A convolutional neural network (CNN) will be utilized to classify the images, distinguishing between healthy and diseased conditions. The model's performance will be evaluated using various metrics, including accuracy, precision, recall, and AUC-ROC. Additionally, the project will include interpretability techniques like Grad-CAM to visualize and explain the model’s predictions, while documenting the entire process, including any challenges faced and how they were addressed.

# Dataset Description
The dataset for this project consists of two parts: real retina images sourced from Kaggle and synthetic retina images generated to simulate both healthy and diseased conditions in cattle. These images will be used to train a deep learning model to classify the presence or absence of cardiovascular disease (CVD) in cattle based on retina scans.


![download](https://github.com/user-attachments/assets/18ec4cb4-3fc9-4234-a93d-6d85abd2a89d)

![download](https://github.com/user-attachments/assets/452703e0-ae7e-4df2-bfbb-c2b93aa4c93b)


1. Real Retina Images from Kaggle
The real retina images were sourced from Kaggle, a platform that hosts various datasets for machine learning tasks. The dataset consists of labeled retina images representing healthy and diseased conditions in cattle. These images contain detailed patterns characteristic of retina health, such as blood vessels, retinal structure, and other features that may indicate the presence of cardiovascular disease.

Healthy Class: Images of cattle with normal retina scans, free from cardiovascular disease.
Diseased Class: Images of cattle exhibiting signs of cardiovascular disease in the retina, such as abnormal blood vessel formation, plaque buildup, or other anomalies indicative of CVD.
Sample Images from Kaggle:

Healthy: These images represent the normal retinal structure of cattle, with clear blood vessels and no signs of disease.
Diseased: These images show abnormalities such as irregular blood vessel patterns or other structural changes that suggest cardiovascular problems.
The dataset from Kaggle includes a total of 100 images (50 healthy, 50 diseased). These images were preprocessed to ensure they are in the correct format (e.g., resized to 224x224 pixels) before use in model training.

2. Synthetic Retina Images
In addition to the Kaggle dataset, synthetic retina images were generated to supplement the dataset. These images are designed to mimic retina scans of cattle, including both healthy and diseased conditions. Using OpenCV, random circular patterns and blood vessel-like lines were added to black background images to simulate retina structures.

Healthy Synthetic Images: These images feature regular, clear circular patterns and blood vessels that resemble the retina of a healthy animal.
Diseased Synthetic Images: These images are modified to include irregularities such as broken blood vessels, abnormal shapes, and other markers indicative of cardiovascular disease.
For each class (healthy and diseased), 100 synthetic images were generated, resulting in a total of 200 synthetic images. These images are resized to the same dimensions as the Kaggle dataset (224x224 pixels) and are labeled accordingly.

3. Data Merging and Stratified Split
The real and synthetic retina images (both healthy and diseased) were merged into a single dataset. The final dataset includes:

50 real healthy images
50 real diseased images
100 synthetic healthy images
100 synthetic diseased images
After merging, a stratified split was performed to ensure that both classes (healthy and diseased) are equally represented in the training and testing sets. Stratified splitting ensures that each subset of the data has the same proportion of samples for each class, which is crucial for avoiding bias in the model training process.

Training Set: 80% of the data (160 images)
Testing Set: 20% of the data (40 images)
The images were split while preserving the distribution of healthy and diseased samples. This resulted in approximately 128 training images and 32 testing images for each class (healthy and diseased).

4. Final Dataset Composition
Total Images: 200 images (100 healthy, 100 diseased)
Training Set: 160 images (80 healthy, 80 diseased)
Testing Set: 40 images (20 healthy, 20 diseased)
The dataset was prepared and organized into two folders: train and test, with subfolders healthy and unhealthy containing the corresponding images.

5. Sample Images
Here are two sample images from each class (healthy and diseased) taken from the merged dataset:

Healthy Image: This image shows a synthetic retina scan with clear, well-formed blood vessels and no visible signs of disease.
Diseased Image: This image represents a synthetic retina scan with irregular blood vessels, indicating potential cardiovascular issues.
6. Preprocessing
The images were resized to 224x224 pixels, normalized, and augmented to improve the robustness of the model. Augmentation techniques such as random rotations, flips, and zooms were applied to enhance the diversity of the dataset, which helps prevent overfitting and improves generalization to unseen data.

# Methodology
The methodology for this image classification task aimed at detecting cardiovascular disease (CVD) in cattle using retina images is outlined in several key steps:

Dataset Collection:

The first step is the acquisition of retina images of cattle that are labeled as healthy or diseased. For this task, we utilized a combination of publicly available datasets, such as those from Kaggle, and supplemented them with synthetically generated retina images. The synthetic data were generated by simulating retina-like structures (such as blood vessels and circular patterns) using random parameters, ensuring that they resembled realistic cattle retina images. These synthetic images were merged with the Kaggle dataset to create a diverse and balanced dataset.
Data Preprocessing:

Resizing: All images were resized to 224x224 pixels to ensure uniformity, as required by most pre-trained models like VGG16.
Augmentation: To increase the robustness of the model and prevent overfitting, data augmentation techniques were applied. These included random rotations, shifts, flips, and zooms.
Stratified Split: The dataset was divided into training and testing sets using a stratified split, ensuring that the class distribution (healthy vs. diseased) remained consistent across both sets. This method is crucial for training models that generalize well on unseen data.
Model Architecture:

![download](https://github.com/user-attachments/assets/270924d6-0d2c-451a-9611-0ff6a338abda)



The VGG16 architecture, a well-established Convolutional Neural Network (CNN), was used as the backbone for feature extraction. VGG16 was pre-trained on ImageNet, and its convolutional layers were utilized to extract features from retina images. The top layers of VGG16 were removed, and custom dense layers were added for classification. These custom layers included:
A Flatten layer to convert the 2D feature maps to a 1D vector.
A Dense layer with 128 units and ReLU activation to capture non-linear relationships.
A Dropout layer with a rate of 0.5 for regularization to reduce overfitting.
A final Dense layer with 1 unit and sigmoid activation for binary classification (healthy vs. diseased).
The model was compiled using the Adam optimizer with binary cross-entropy loss, suitable for binary classification tasks.
Model Training:

The model was trained on the processed retina images, using the training dataset, for a predefined number of epochs with batch processing. The performance was evaluated on the validation set, which was part of the stratified split


# Performance and Evaluation
Final Results Summary:
Training Accuracy: 48.48%
Validation Accuracy: 52.48%
Testing Accuracy: 52.48%
Classification Report:
Precision: 0.43

Precision refers to the proportion of positive predictions that are actually correct. The precision of 0.43 indicates that, out of all the predicted positive instances (both "healthy" and "unhealthy"), only 43% were actually correct.
Recall: 0.39

Recall refers to the proportion of actual positive instances that were correctly identified by the model. The recall of 0.39 shows that the model only correctly identified 39% of the actual positive instances (i.e., the true cases of "healthy" or "unhealthy").
F1-Score: 0.41

The F1-score is the harmonic mean of precision and recall, providing a balance between them. A score of 0.41 suggests that the model is underperforming in both precision and recall.
Support: 50 (for "healthy") and 51 (for "unhealthy")

The support indicates the number of true instances for each class in the test set.
Model Performance Insights:
The accuracy values suggest that the model is not performing well, as it is just slightly better than random chance (around 50%).
The precision and recall are both relatively low (around 0.43 for precision and 0.39 for recall), which indicates that the model struggles to correctly classify the classes and has a high number of false positives and false negatives.
The F1-score of 0.41 shows that the model's classification performance is suboptimal, and there is room for improvement.

![download](https://github.com/user-attachments/assets/4f3580a8-8654-4c74-b606-238bbba74db8)
![download](https://github.com/user-attachments/assets/f0cb59e4-8bfa-42fb-b04c-5044519b5d67)
![download](https://github.com/user-attachments/assets/ce88be47-8172-446f-91ba-b7d6809a3227)

# Challenges and Solutions
1. Dataset Quality and Size
Challenge:

The model's performance may be limited due to insufficient or unbalanced data, especially with small or imbalanced datasets. A low number of labeled samples for both "healthy" and "unhealthy" cattle could lead to overfitting or poor generalization.
Solution:

Data Augmentation: Use data augmentation techniques such as rotations, flips, and zooms to artificially increase the dataset size and introduce variety in the images.
Synthetic Data Generation: For further data enrichment, synthetic images could be generated to mimic real retinal images using computer vision techniques. This increases the diversity of the dataset and helps the model generalize better.

2. Class Imbalance
Challenge:

If the dataset has an imbalance in the number of healthy and unhealthy images, the model may be biased towards the majority class, leading to poor predictions for the minority class.
Solution:

Resampling: Perform oversampling or undersampling to balance the dataset. Techniques such as SMOTE (Synthetic Minority Over-sampling Technique) can be used to generate synthetic samples for the minority class.
Class Weights: Adjust class weights in the loss function so that the model pays more attention to the minority class during training.

3. Model Overfitting
Challenge:

A model that performs well on training data but poorly on validation/test data is likely overfitting, especially with small datasets. This occurs when the model learns the noise or random fluctuations in the training data rather than the underlying patterns.
Solution:

Regularization: Implement dropout layers, L2 regularization, or batch normalization to prevent overfitting and improve generalization.
Early Stopping: Use early stopping during training to halt the process once the validation accuracy stops improving, preventing unnecessary overfitting.
 
 4. Model Complexity and Feature Extraction
Challenge:

A simple model may struggle to capture the complex patterns in retina images, leading to poor performance.
Solution:

Transfer Learning: Use pre-trained models (e.g., VGG16, ResNet, Inception) that have been trained on large image datasets like ImageNet. These models have learned useful features for a wide range of image classification tasks. Fine-tuning these models on our specific dataset can improve performance.
Feature Extraction: Extract more advanced features like texture, shape, or edge patterns that may help distinguish between healthy and unhealthy retina images.




