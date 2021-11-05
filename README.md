# temperature-predictor
LSTM model used to analyze a time series of temperatures and make predictions.

## Getting started
This notebook should be executed in an environment running Python.

Dependencies -

Tensorflow: Build the deep learning model and train it.  
Numpy: Organize the data into numpy arrays and split into train/validation subsets.  
Matplotlib: Visualize the results of the model training and overlay the known temperatures with the predictions.  
Csv: Read in the csv file.

## Overview
The code begins by importing the dependencies and defining a helper function to plot data. The data is downloaded to the temp folder and the csv file is read. The relevant information from the dataset is saved in a couple of lists, which are converted to numpy arrays and then plotted. The dataset is split into train and validation arrays.

A couple of important functions are defined: windowed_dataset and model_forecast. The first function, windowed_dataset, prepares the dataset for training with the model. It creates windows 31 values in length: the first 30 values will be the "x" value for training, and the 31st value will be the "y" value. It prepares the data in batches of 32. The second function, model_forecast, creates windows 30 values in length that are fed into the trained model to return predictions.

A basic LSTM model is defined using the Tensorflow library. An LSTM was chosen for this task because it performs well with time series data. The cell state maintained by an LSTM allows all values of the time series to have an effect on the output. A callback is defined to change the learning rate with each epoch, to determine the ideal learning rate for the model. The model is compiled and trained, and the results are plotted to see the ideal learning rate. The model is trained again with a constant learning rate (determined in the previous step) and the results of training are plotted.

The model_forecast function is used to make predictions over the validation data. The predictions are graphed with the actual values, and the mean absolute error is calculated for the predictions. MAE is used to evaluate the model accuracy because it's less punishing toward outliers.

## Results
![time-series-lr](https://user-images.githubusercontent.com/83524079/140448341-767c833e-a4a7-4093-9c73-943206267a7d.png)

The mean average error bottomed out between 10^-6 and 10^-5. I decided to use 5 x 10^-6 as the learning rate for model training.

![time-series-training](https://user-images.githubusercontent.com/83524079/140448542-a601e90b-a807-43a3-9329-bfa552afad2b.png)

The mean average error plotted against the epoch after model training. As you can see, there is some overfitting of the model after 10 epochs.

![time-series-final](https://user-images.githubusercontent.com/83524079/140448702-b52199f5-8682-470c-87c4-a1ea4df5dd6a.png)

The predicted temperatures (orange) plotted with the actual temperatures (blue). The predicted temperatures fail to match the most extreme temperatures, but they do a great job matching the seasonality of the data.

The MAE between the predicted and actual temperatures was calculated to be 1.83.

## Future work
Some ideas for future work to refine the model:
- Experiment with the window size
- Add/drop layers of the model
- Change the loss algorithm
