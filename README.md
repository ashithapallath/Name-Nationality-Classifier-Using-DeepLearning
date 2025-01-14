# Nationality Classifier


This repository contains the implementation of a deep learning-based nationality classifier designed to determine whether a name belongs to an Indian or Non-Indian individual. By analyzing name patterns using advanced neural networks, this project aims to provide accurate predictions of nationality, which can be applied in various domains such as demographic analysis, personalized services, and more.

This project was developed as part of an academic experiment to explore the capabilities of deep learning models in linguistic and cultural classification.



## Objective
The main objectives of this project are:

1. **Build a Classifier**:
   - Design a deep learning model capable of distinguishing Indian from Non-Indian names based on training data.

2. **Explore Name Patterns**:
   - Leverage neural network architectures to analyze phonetic, linguistic, and cultural patterns in names.

3. **Test and Evaluate**:
   - Validate the model’s performance on diverse datasets to ensure robustness and accuracy.


## Features

- **Deep Learning Model**: Implements a neural network optimized for text-based classification.
- **Linguistic Analysis**: Extracts features from names to identify patterns.
- **Versatile Applications**: Useful for demographic studies, marketing, and automated services.



## Methodology

1. **Dataset**:
   - The model is trained using datasets from the following sources:
     - [Indian Female Names](https://raw.githubusercontent.com/ashavish/name-nationality/master/data/datasets_70812_149772_Indian-Female-Names.csv)
     - [Indian Male Names](https://raw.githubusercontent.com/ashavish/name-nationality/master/data/datasets_70812_149772_Indian-Male-Names.csv)

2. **Model Architecture**:
   - A neural network with embedding layers for feature extraction.
   - ReLU activation for non-linearity.
   - BatchNormalization and Dropout layers for improved stability and generalization.

3. **Training**:
   - Train the model with a labeled dataset.
   - Use categorical cross-entropy as the loss function.
   - Optimize using Adam optimizer.

4. **Evaluation**:
   - Test the classifier on unseen names.
   - Report accuracy, precision, recall, and F1 scores.



## Requirements

### Software

- Python 3.8+
- TensorFlow/Keras
- Pandas
- NumPy
- Matplotlib

Install the dependencies using pip:

```bash
pip install -r requirements.txt
```


## Usage

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/nationality-classifier.git
   cd nationality-classifier
   ```

2. **Prepare the Dataset**:

   - Place the training and testing datasets in the `data/` directory.

3. **Train the Model**:

   - Run the provided Jupyter Notebook or script to train the classifier.
   
   ```bash
   python train_model.py
   ```

4. **Test the Classifier**:

   - Evaluate the model on test data to analyze its performance.

5. **Make Predictions**:

   - Use the trained model to predict the nationality of new names.



## Project Structure

- `NationalityClassifier.ipynb`: Main notebook for model development and experimentation.
- `data/`: Contains training and testing datasets.
- `models/`: Directory for saving trained models.
- `train_model.py`: Script to train and evaluate the model.



## Results

- The model demonstrates high accuracy in classifying names as Indian or Non-Indian.
- Evaluation metrics such as precision, recall, and F1 scores are detailed in the output logs.



## Contributing

Contributions are welcome! If you have ideas for improving the model or extending its functionality, feel free to fork the repository and submit a pull request.



## License

This project is licensed under the MIT License. See the LICENSE file for details.



