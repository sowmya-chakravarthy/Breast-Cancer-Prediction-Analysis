# Breast-Cancer-Prediction-Analysis

### INTRODUCTION AND PRE PROCESSING

The code performs a breast cancer classification task using machine learning techniques. The code begins by loading a dataset from a CSV file into a pandas DataFrame, where missing values represented as '?' are treated as NaN (Not a Number). The missing values are then filled with the mean of the respective columns. The dataset is preprocessed to separate the features (columns 1 to 9) from the target variable (column 10). The dataset is split into training and testing sets, with 75% of the data used for training and 25% for testing. The random_state parameter is set to 100 to ensure reproducibility in the data split. Finally, the feature and target sets are converted to NumPy arrays, which are more memory-efficient than Python lists. This preprocessed data can now be used to train and evaluate machine learning models for breast cancer classification. "Wisconsin Breast Cancer" which is the dataset is preprocessed and cleaned is classifed for tumor as 'malignant' or 'benign' and the goal is to find which classifier gives accurate prediction.

Dataset used - https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

**Techniques used:**

- Decision Tree
- Random Forest
- KNN
- Guassian Naive Bayes
- Multinomial Naive Bayes
- SVM

### DECISION TREE CLASSIFIER

A decision tree classifier is a powerful machine learning algorithm used for solving classification tasks. It constructs a tree-like model where each internal node represents a decision based on a particular feature, and each leaf node represents a class label. The algorithm recursively partitions the data into subsets, aiming to maximize the purity of the resulting subsets based on metrics like Gini impurity or entropy. Decision trees are highly interpretable, capable of handling both numerical and categorical data, and capturing nonlinear relationships.

The code performs Decision Tree Classification to the preprocessed breast cancer dataset. First, a DecisionTreeClassifier from the scikit-learn library is instantiated with random_state set to 1 for reproducibility. The classifier is then trained on the training data (X_train and y_train) using the fit method. Next, predictions are made on the test data (X_test) using the predict method, and the accuracy of the classifier is calculated using the metrics.accuracy_score function, which compares the predicted values (y_pred) to the actual target values (y_test). K-fold cross-validation with k set to 10 using the cross_val_score function from scikit-learn has been perfomed. This technique divides the training data into 10 equal-sized folds, where each fold is used as a validation set while the model is trained on the remaining 9 folds. This process is repeated 10 times, ensuring that each fold serves as the validation set once. The mean accuracy of the 10 iterations is calculated and printed as the result of the cross-validation.

### RANDOM FOREST

Random Forest classifier is a popular ensemble learning technique in machine learning used for classification tasks. It constructs multiple decision trees during training and combines their predictions to make a final decision. Each tree is built using a random subset of the data and a random subset of features, which promotes diversity and reduces overfitting. During prediction, the individual tree outputs are aggregated, and the majority class is assigned as the final prediction. Random Forests are robust, handle high-dimensional data well, and can capture complex relationships in the data. They are widely used due to their high accuracy, resistance to overfitting, and interpretability through feature importance measures derived from the ensemble.

The code applies Random Forest Classification to the preprocessed breast cancer dataset. The Random Forest classifier is created with n_estimators set to 10, indicating that the classifier will consist of 10 decision trees. The random_state is set to 1 to ensure reproducibility of the results. The RandomForestClassifier is then trained on the training data (X_train and y_train) using the fit method. After training, predictions are made on the test data (X_test) using the predict method, but the results are not stored in a variable or printed. Instead, the code proceeds to perform k-fold cross-validation with k set to 10 using the cross_val_score function from scikit-learn. The Random Forest classifier (rf) is passed as the estimator to evaluate, and the training data is divided into 10 folds for the cross-validation process. In this project, Random forest accuracy is slightly higher than decision tree.

### K NEAREST NEIGHBOUR

K-Nearest Neighbors (KNN) is a simple and effective machine learning algorithm used for both classification and regression tasks. In KNN, the prediction for a new data point is determined by the majority class (in classification) or the average of the neighboring data points' target values (in regression) among its k nearest neighbors in the feature space. The distance metric (e.g., Euclidean distance) is used to identify these nearest neighbors. KNN's strength lies in its simplicity and ability to adapt to complex decision boundaries in the data. However, it can be computationally expensive for large datasets, and its performance heavily relies on the appropriate choice of the k value and the distance metric. Preprocessing the data, such as scaling the features, is also crucial to avoid the dominance of certain dimensions in the distance calculation.

The code applies K Nearest Neighbors (KNN) Classification to the preprocessed breast cancer dataset. First, a KNeighborsClassifier is instantiated with n_neighbors set to 10, indicating that the model will consider the 10 nearest neighbors to make predictions. The classifier is then trained on the training data (X_train and y_train) using the fit method. Predictions are made on the test data (X_test), and the accuracy of the KNN model is calculated using the metrics.accuracy_score function, which compares the predicted values (y_pred) to the actual target values (y_test).

- The value of K doesn't make significant difference as there is only slight increase/decrease in accuracy as K value changes.
- The best performance from KNN using 10-fold cross validation is 0.9771407837445573.

### NAIVE BAYES

The code applies two different Naive Bayes classifiers, namely GaussianNB and MultinomialNB, to the preprocessed breast cancer dataset.First, it initializes and trains a Gaussian Naive Bayes classifier (GaussianNB) on the training data (X_train and y_train) using the fit method. Predictions are then made on the test data (X_test) using the predict method, and the accuracy of the model is calculated using the metrics.accuracy_score function, which compares the predicted values (sk_pred) to the actual target values (y_test). The obtained accuracy score is printed to show the performance of the GaussianNB classifier on this dataset.

Next, the it performs k-fold cross-validation with k set to 10 using the cross_val_score function from scikit-learn. The GaussianNB classifier is passed as the estimator to evaluate, and the training data is divided into 10 folds for the cross-validation process. The mean accuracy of the 10 iterations is calculated and printed to provide a more robust evaluation of the GaussianNB classifier's performance. Similarly, the code repeats the same process for the Multinomial Naive Bayes classifier (MultinomialNB). The MultinomialNB model is instantiated, trained, and evaluated using accuracy scores in both a single train-test split and 10-fold cross-validation.

The code showcases the use of two different Naive Bayes classifiers, GaussianNB and MultinomialNB, for breast cancer detection. The accuracy results obtained from both the train-test split and 10-fold cross-validation provide insights into how well these classifiers perform on this specific dataset. 

#### GUASSIAN NAIVE BAYES

Gaussian Naive Bayes is a popular variant of the Naive Bayes algorithm used for classification tasks in machine learning. It is particularly suitable for continuous or numerical feature data. The algorithm assumes that the features are normally distributed within each class, hence the term "Gaussian." During training, it estimates the mean and variance of each feature for each class. When making predictions, it applies Bayes' theorem to calculate the probability of a data point belonging to each class, taking into account the prior probabilities of the classes and the likelihood of the features given the class. The final prediction is based on the class with the highest calculated probability. Despite its simplicity and the independence assumption between features (hence "Naive"), Gaussian Naive Bayes often performs surprisingly well in many real-world scenarios, especially when the data follows approximately Gaussian distributions. However, it may not be suitable for datasets with highly correlated features or discrete data, for which other variants of Naive Bayes might be more appropriate.

#### MULTINOMIAL NAIVE BAYES

Multinomial Naive Bayes is a variant of the Naive Bayes algorithm designed specifically for handling discrete or count-based features in machine learning, commonly used for text classification tasks. Unlike Gaussian Naive Bayes, which assumes continuous feature distributions, Multinomial Naive Bayes assumes that the features are generated from a multinomial distribution. In natural language processing tasks, the features often represent word counts or frequencies in a document, and the algorithm calculates probabilities of occurrence for each word in each class during training. When making predictions, it applies Bayes' theorem to compute the likelihood of a data point belonging to each class, considering the prior probabilities of the classes and the probabilities of features given the class. Multinomial Naive Bayes is computationally efficient and performs well in tasks involving text data, especially when the vocabulary is large, but it might not capture more complex relationships between words as compared to other advanced techniques such as deep learning-based models.

-- Conclusion: From all the models , KNN has highest accuracy and MultinomialNB has low accuracy.

### SVM - SUPPORT VECTOR MACHINE

Support Vector Machine (SVM) is a powerful supervised learning algorithm used for both classification and regression tasks in machine learning. In the context of classification, SVM aims to find the hyperplane that best separates the data points belonging to different classes in the feature space. The hyperplane is chosen to maximize the margin, which is the distance between the hyperplane and the nearest data points of each class, resulting in better generalization to unseen data. SVM can handle both linearly separable and non-linearly separable data through the use of kernel functions that map the original feature space into a higher-dimensional space, where the data points become linearly separable. SVM is effective in high-dimensional spaces and is also robust against overfitting when the regularization parameter is properly tuned. Additionally, SVMs have been widely used for tasks like text classification, image recognition, and other complex data classification problems. However, SVM can be computationally expensive for large datasets and can be sensitive to the choice of hyperparameters. Nevertheless, it remains one of the most popular and widely used algorithms in various machine learning applications.

The code uses a for loop to create and train three SVM models with different kernels: linear, sigmoid, and polynomial (poly). For each model, it calculates the accuracy using 10-fold cross-validation and prints the results. The average accuracies for each kernel are stored in a list (accList). Best Kernel Selection: After comparing the average accuracies for different kernels, the code prints the best-performing kernel based on the highest accuracy achieved during the cross-validation process.

-- Linear kernel performs best using 10-fold cross validation with accuracy of 0.9714078374455732.

|   Classifier   |    Accuracy   |
| -------------- | ------------- |
| Decision Tree  |     95.22%    |
| Random Forest  |     96.37%    |
|      KNN       |     97.33%    |
|  Guassian NB   |     96.75%    |
| Multinomial NB |     89.88%    |

