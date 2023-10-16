# Image Classification using Fine-Tuned ResNet101

This project is an example of image classification using a fine-tuned ResNet101 model. It demonstrates how to load a pre-trained ResNet101 model, fine-tune it for a specific classification task, and visualize the model's performance.

## Project Setup

1. **Mount Google Drive**: The project begins with mounting Google Drive to access the dataset and save the trained model. The `drive.mount()` function is used to connect the Colab notebook to Google Drive.

2. **Importing Libraries**: The necessary libraries are imported. These include:
   - `torch` and `torch.nn` for PyTorch-based deep learning.
   - `numpy` for numerical operations.
   - `torchvision` for computer vision-related tasks.
   - `matplotlib` for data visualization.
   - Other utilities for handling data and time.

## Data Preparation

3. **Data Augmentation and Preprocessing**: Data augmentation and preprocessing transformations are defined using the `transforms.Compose` function. These transformations are used to preprocess and augment the data. For instance, it includes resizing, cropping, and normalizing images for training and validation.

4. **Dataset Loading**: The dataset is loaded from Google Drive using `torchvision` and organized into "Train" and "Val" sets. The paths to the datasets are specified in `data_dir`.

5. **Data Loaders**: Data loaders are created to efficiently load and iterate through the dataset during training and validation. They use `torch.utils.data.DataLoader`.

## Training

6. **Model Training**: The `train_model` function is defined to train the ResNet101 model. It runs training and validation epochs, computes losses, and keeps track of the best model weights. The model is trained for a specified number of epochs, and the best model is saved.

## Data Visualization

7. **Data Visualization Functions**: Functions like `imshow` and `visualize_model` are defined for visualizing the data and model performance. These are used to display example images and visualize predictions.

8. **Model Fine-Tuning**: The project demonstrates how to fine-tune a pre-trained ResNet101 model for a specific classification task. The last fully connected layer is replaced to match the number of classes in the new task.

9. **Loss Function and Optimization**: The project defines the loss function (`nn.CrossEntropyLoss`) and sets up the optimization algorithm (`Adam`). Learning rate scheduling is also applied to adjust the learning rate during training.

## Saving and Loading Models

10. **Model Saving**: After training, the best model is saved to Google Drive using `torch.save`.

11. **Model Loading**: The trained model can be loaded and used for inference.

## Usage

This project serves as a template for image classification tasks using fine-tuned models. You can adapt it for your specific classification problem by changing the dataset, number of classes, and other parameters.

The primary purpose of this project is to demonstrate the fine-tuning process of a deep learning model for a specific task, with a focus on image classification.

Feel free to modify and extend this project to suit your own image classification tasks.
