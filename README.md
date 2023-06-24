# ASL_recognition
word level American sign language recognition 

# SVM result
![Untitled video - Made with Clipchamp](https://github.com/marihanraafat/ASL_recognition/assets/91830467/b9e051c3-0594-4555-bb7d-0c76c0bb509a)

# LSTM result
![Untitled video - Made with Clipchamp (1)](https://github.com/marihanraafat/ASL_recognition/assets/91830467/79f30174-775b-441f-b5e7-73040d05dbc3)


# lstm model description
The model architecture consists of three BLSTM layers, each with 64, 128, and 64 units, respectively. The input to the model is a sequence of 20 frames of keypoints, with 1662 dimensions each. The first BLSTM layer is followed by batch normalization and dropout layers to reduce overfitting. The second BLSTM layer is also followed by batch normalization and dropout layers, and the third BLSTM layer is followed by a dense layer with 64 units and a dropout layer. The output of the dropout layer is then passed on to a dense layer with 32 units and another dropout layer.
The output layer depends on the range of labels in the dataset. If the labels are non-negative, the output layer has a ReLU activation function. Otherwise, the output layer has a softmax activation function.
The model is trained using the categorical cross-entropy loss function and the Adam optimizer with a learning rate of 0.0001. The training process is regularized using early stopping and reducing learning rate on plateau techniques to improve generalization performance and prevent overfitting. The model is evaluated using accuracy as a performance metric.

# gru model description
The architecture consists of two GRU layers, each with 128 units. The input to the model is a sequence of 21 frames of keypoints, with 1662 dimensions each. The first GRU layer is followed by batch normalization and dropout layers to reduce overfitting. The second GRU layer is also followed by batch normalization and dropout layers, and a dense layer with num_classes units. The output layer has a softmax activation function.
The model is regularized using L2 regularization with a regularization coefficient of 0.001. The model is trained using the categorical cross-entropy loss function and the Adam optimizer with a learning rate of 0.0001. The learning rate is adjusted based on an epoch-based learning rate schedule, where the learning rate is reduced to 0.00001 after 10 epochs.

# svm model description

To train an SVM model for gesture recognition using hand landmarks, the first step is to collect a dataset of hand gestures and corresponding landmark sequences. The dataset can be divided into a training set and a validation set to optimize the model hyperparameters and prevent overfitting. Once the dataset is prepared, the landmark sequences can be loaded into the SVM model using a suitable feature extraction technique.
The SVM model can be trained using the training set and the hyperparameters can be optimized using the validation set. One important hyperparameter for SVM models is the regularization parameter C, which controls the trade-off between the margin width and the training error. A small value of C results in a wider margin but may lead to underfitting, while a large value of C results in a narrower margin but may lead to overfitting. In this case, a value of C=1 has been specified, which is a common default value.
Once the SVM model has been trained, it can be used to recognize new hand gestures based on their landmark sequences. The model can be evaluated on a test set to estimate its performance on new, unseen data. 
