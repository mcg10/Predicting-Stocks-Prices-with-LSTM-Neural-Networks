# Predicting-Stocks-Prices-with-LSTM-Neural-Networks
In this project, I initially organized stock data, which I imported via Yahoo Fin, with Pandas and NumPy in order to fit a neural network model. Using Tensorflow and Keras, I created a Long-Short-Term-Memory neural network that predicted future stock prices given current data. The predictions of the model were then plotted against the actual prices of stocks using Matplotlib. To run the project, open up run "create_and_train.py" and adjust the "ticker" variable to the company you are concerned with. 

While the program is fairly accurate, I have two areas in which I plan to improve the project:

1. The code for creating/training the model and for plotting the graphs are all currently in the same script. I worked in Jupyter Notebook and had some issues with importing local Python files, but I plan to separate the code into multiple scripts in order to make editing and documenting the code easier overall. 
2. The program is somewhat rigid in terms of its parameters. I plan to make the code more flexible in terms of adjusting parameters such as epochs, lookup steps, batch sizes, and test-data ratios.

Thank you to Abdou Rockikz for the inspiration to pursue this project! 

The following libraries were implemented in the project: 
* Tensorflow
* Keras
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Yahoo Fin
* OS
* Time
