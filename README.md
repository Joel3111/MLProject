# Eye Disease Classification Project

## Project Overview
This project aims to classify retinal images into four categories: Normal, Diabetic Retinopathy, Cataract, and Glaucoma using a Convolutional Neural Network (CNN) implemented in TensorFlow/Keras.

## Dataset
The dataset consists of approximately 1000 images per class, collected from Kaggle. Each image is resized to 206x206 pixels.

## Project Structure
- `MLProject.ipynb`: Jupyter notebook with the full project code.
- `dataset/`: Directory containing the image dataset organized into subdirectories for each class.
- `images/`: Directory containing the images used in the project (e.g., confusion matrix, training plots).
- `app.py`: Flask application for testing the model.
- `templates/`: Directory containing HTML templates for the Flask app.
- `requirements.txt`: List of dependencies required to run the project.
- `README.md`: Project documentation.
- `.gitignore`: Git ignore file.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Joel3111/MLProject
   cd MLProject
   

## Model Performance

- **Training Accuracy**: 85%
- **Validation Accuracy**: 52.68%
- **Validation Loss**: 1.0663

### Interpretation of Results

#### Confusion Matrix Analysis

- **Normal**: The model correctly identified only 10 out of 207 normal images, with many misclassifications into other categories, especially Glaucoma.
- **Diabetic Retinopathy**: The model correctly classified 48 out of 219 images. While this is better than other classes, the overall precision and recall are still low.
- **Cataract**: Very few cataract images were correctly classified, indicating significant difficulty in identifying this condition.
- **Glaucoma**: The model performed relatively better on Glaucoma, with a recall of 0.71, meaning it correctly identified 152 out of 214 images.

### Classification Report

- **Precision**: Indicates the accuracy of the positive predictions. The precision values are generally low, with the highest for Glaucoma (0.26).
- **Recall**: Indicates the ability of the model to find all relevant cases within a dataset. The recall for Glaucoma is relatively high (0.71), but very low for other classes.
- **F1-Score**: The harmonic mean of precision and recall. The F1-scores are low across all classes, with the highest being 0.38 for Glaucoma.

### Discussion

The model struggles with correctly classifying most of the eye disease categories, particularly Normal, Diabetic Retinopathy, and Cataract. The significant gap between training and validation accuracy suggests overfitting. Overfitting occurs when the model learns the training data, including noise, too well and fails to generalize to new, unseen data. Glaucoma is the only category where the model shows a somewhat reasonable recall.

### Improvement Suggestions

- **Data Augmentation**: Increase the variability of the training data to improve generalization.
- **Regularization**: Implement techniques like dropout and L2 regularization to reduce overfitting.
- **Hyperparameter Tuning**: Optimize model parameters to improve performance.
- **Ensemble Methods**: Combine predictions from multiple models to enhance accuracy.

### Improvements

- Implement cross-validation.
- Experiment with different CNN architectures.
- Use data augmentation and regularization to reduce overfitting.
- Perform hyperparameter tuning.
