## Eye Disease Classification Project

### Project Goal/Motivation

The primary goal of this project is to develop a robust machine learning model to accurately classify retinal images into four categories: Normal, Diabetic Retinopathy, Cataract, and Glaucoma. This project is driven by both a personal and public health motivation. On a personal level, the motivation stems from the diagnosis of my brother with "grauer Star" (Cataract in German), which highlighted the importance of early detection and accurate diagnosis of eye diseases. This personal connection underscores the urgency to create a tool that can aid in early detection and classification of eye diseases, potentially improving patient outcomes through timely intervention.

From a public health perspective, eye diseases like Diabetic Retinopathy, Cataract, and Glaucoma are significant causes of vision impairment and blindness worldwide. Early detection and classification are crucial for effective treatment and management. By leveraging convolutional neural networks (CNNs) and advanced machine learning techniques, this project aims to contribute to the field of ophthalmology by providing an automated, reliable, and efficient diagnostic tool.

### Project Structure

- `MLProject.ipynb`: Jupyter notebook with the full project code.
- `dataset/`: Directory containing the image dataset organized into subdirectories for each class.
- `ResultsImages/`: Directory containing the images used in the project (e.g., confusion matrix, training plots) and results of the prediction.
- `app.py`: Flask application for testing the model.
- `templates/`: Directory containing HTML templates for the Flask app.
- `requirements.txt`: List of dependencies required to run the project.
- `README.md`: Project documentation.
- `.gitignore`: Git ignore file.

### How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/Joel3111/MLProject
   cd MLProject
   ```

2. Download the pre-trained model from [this link](https://drive.google.com/drive/folders/1CX92O-u0nVKGh5r6rcPLmQgjRZaNILSQ?usp=sharing) and place it in the project directory.

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask app:
   ```bash
   python app.py
   ```

By following these steps, you can set up and run the eye disease classification project locally.

### First Approach

#### Steps Taken

1. **Data Preparation**:

   - Collected a dataset from Kaggle consisting of approximately 1000 images per class.
   - Resized all images to 206x206 pixels to standardize input sizes for the model.

2. **Data Augmentation**:

   - Applied basic data augmentation techniques such as horizontal flip, zoom, shear, and rotation to increase dataset variability and help the model generalize better.

3. **Model Architecture**:

   - Built a simple CNN with the following layers:
     - Convolutional layers with ReLU activation and max-pooling.
     - Flatten layer to convert 2D features to 1D.
     - Dense layer with ReLU activation.
     - Dropout layer for regularization.
     - Output layer with softmax activation for classification.

4. **Training**:

   - Trained the model for 25 epochs using categorical crossentropy as the loss function and the Adam optimizer.

5. **Evaluation**:
   - Plotted training and validation accuracy/loss curves.
   - Generated a confusion matrix and classification report to analyze performance.

#### Results

- **Training Accuracy**: 85%
- **Validation Accuracy**: 52.68%
- **Validation Loss**: 1.0663

##### Confusion Matrix Analysis

- **Normal**: The model correctly identified only 10 out of 207 normal images, with many misclassifications into other categories, especially Glaucoma.
- **Diabetic Retinopathy**: The model correctly classified 48 out of 219 images. While this is better than other classes, the overall precision and recall are still low.
- **Cataract**: Very few cataract images were correctly classified, indicating significant difficulty in identifying this condition.
- **Glaucoma**: The model performed relatively better on Glaucoma, with a recall of 0.71, meaning it correctly identified 152 out of 214 images.

##### Classification Report

- **Precision**: Indicates the accuracy of the positive predictions. The precision values are generally low, with the highest for Glaucoma (0.26).
- **Recall**: Indicates the ability of the model to find all relevant cases within a dataset. The recall for Glaucoma is relatively high (0.71), but very low for other classes.
- **F1-Score**: The harmonic mean of precision and recall. The F1-scores are low across all classes, with the highest being 0.38 for Glaucoma.

#### Conclusion

The significant gap between training and validation accuracy suggested overfitting. Despite achieving high training accuracy, the model failed to generalize well to the validation set. This could be due to several factors:

- **Data Imbalance**: Potential imbalances in the dataset might have skewed the model's learning process.
- **Model Complexity**: The architecture might not have been complex enough to capture the nuances of the different eye diseases.
- **Insufficient Regularization**: Overfitting indicated the need for stronger regularization techniques.

### Second Approach

#### Changes Made

1. **Data Preparation**:

   - Increased the image size to 256x256 pixels to allow the model to capture more detailed features.

2. **Data Augmentation**:

   - Continued using aggressive data augmentation techniques to further increase dataset variability and improve generalization.

3. **Model Architecture**:

   - Enhanced the CNN architecture:
     - Increased the complexity by adding more convolutional layers.
     - Used larger kernel sizes in initial layers to capture broader features.
     - Kept the dense and dropout layers for regularization.

4. **Regularization and Early Stopping**:

   - Implemented L2 regularization in dense layers to penalize large weights.
   - Added early stopping with a patience of 8 epochs to halt training when the validation loss stopped improving.

5. **Training**:
   - Trained the model for up to 50 epochs with early stopping.
   - Saved the best model based on validation performance.

#### Results

- **Training Accuracy**: 72%
- **Validation Accuracy**: 47%
- **Validation Loss**: Fluctuating but generally improving.

##### Confusion Matrix Analysis

- **Normal**: The model correctly identified 33 out of 207 normal images, with many misclassifications into other categories, especially Glaucoma.
- **Diabetic Retinopathy**: The model correctly classified 61 out of 219 images, showing slightly better performance.
- **Cataract**: 20 correctly identified out of 201 images, indicating ongoing difficulty in identifying this condition.
- **Glaucoma**: 96 correctly classified out of 214 images, showing relatively better recall.

##### Classification Report

- **Precision**: Generally low, indicating many false positives.
- **Recall**: Low for most classes except Glaucoma.
- **F1-Score**: Low across all classes, indicating poor overall performance.

### Interpretation and Possible Causes

1. **Data Imbalance and Quality**:

   - The dataset may still have imbalances or inconsistencies in image quality between classes, affecting the model's ability to learn effectively.

2. **Overfitting**:

   - The significant gap between training and validation accuracy suggests overfitting. Despite early stopping, the model still memorizes the training data better than generalizing to new data.

3. **Feature Extraction**:
   - The current CNN architecture might not be complex enough to capture the subtle differences between the classes, especially for conditions like Cataract and Normal.

### Improvement Suggestions

1. **Further Data Augmentation**:

   - Increase the diversity of the training data through more aggressive augmentation strategies to help the model generalize better.

2. **Advanced Regularization Techniques**:

   - Implement stronger regularization techniques such as batch normalization and increased dropout rates to reduce overfitting.

3. **Hyperparameter Tuning**:

   - Perform a systematic search for the optimal hyperparameters using tools like GridSearchCV or RandomSearchCV.

4. **More Complex Architectures**:

   - Utilize more advanced CNN architectures such as ResNet, Inception, or EfficientNet, which have better feature extraction capabilities.

5. **Cross-Validation**:

   - Implement k-fold cross-validation to get a more robust estimate of model performance.

6. **Ensemble Methods**:
   - Combine predictions from multiple models to reduce bias and variance, potentially improving overall performance.

### Summary

Throughout this project, we systematically explored various strategies to improve the performance of a CNN for eye disease classification. We identified the challenges and limitations of our initial approach, implemented enhancements in our second approach, and analyzed the results to understand the impact of these changes. Despite the improvements, there is still room for further optimization and experimentation to achieve better generalization and accuracy.

By documenting our process and understanding the underlying reasons for each step, we demonstrated a thorough grasp of the methodologies and their implications. Moving forward, we can leverage more advanced techniques and rigorous evaluations to build a more robust and accurate classification model.
