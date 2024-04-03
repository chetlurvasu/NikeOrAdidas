# NikeOrAdidas
Shoe Brand Classifier
This project involves building a deep learning model to classify images of shoes into two categories: Nike and Adidas. The model is based on transfer learning using the VGG16 architecture pre-trained on the ImageNet dataset. Transfer learning involves taking a pre-trained model and fine-tuning it on a specific dataset, in this case, images of Nike and Adidas shoes.

Dataset
The dataset consists of images of Nike and Adidas shoes collected from various sources. The dataset is divided into training and validation sets to train and evaluate the model's performance.

Model Architecture
The model architecture is based on VGG16 with the fully connected layers replaced with custom layers for classification. The pre-trained VGG16 model's convolutional layers are frozen, and only the custom layers are trained on the shoe dataset.

Training
The model is trained using an image data generator to augment the training images with random transformations like rotation, shear, and flip. Training is performed using the categorical cross-entropy loss function and the RMSprop optimizer.

Evaluation
The model's performance is evaluated on the validation set using accuracy as the metric. Model checkpoints are saved during training to track the best performing model based on validation accuracy.

Prediction
After training, the model can be used to predict the brand of a shoe from a given image. The input image is preprocessed, fed into the trained model, and the predicted brand (Nike or Adidas) is outputted.

Dependencies
1. Python 3.x
2. TensorFlow
3. Keras
3. numpy
4. pandas
5. matplotlib

Usage
1. Clone the repository.
2. Install the dependencies listed in requirements.txt.
3. Train the model using train.py.
4. Evaluate the model using evaluate.py.
5. Make predictions using predict.py.


Feel free to contribute or provide feedback!
