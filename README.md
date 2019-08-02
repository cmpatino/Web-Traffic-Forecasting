# Web-Traffic-Forecasting

## Overview

The objective of this project was to predict the web traffic is around 145.000 Wikipedia pages. The project was a [Kaggle competition](https://www.kaggle.com/c/web-traffic-time-series-forecasting/overview) sponsored by Google. This problem reduces to predicting the 60 next values of 145.000 different time series. 

## Solution 

### Models

We based our approach on the power of Recurrent Neural Networks (RNNs) on sequential data. 

Our first approach was to use a single Long-Short Memory network (LSTM), followed by a dense network. The output of the dense layer was the prediction of the 60 next days of the series.  

Our final approach was to use a sequence to sequence models using to LSTM networks. The encoder received the values of the training series and the additional lagging features. Then, the decoder received the hidden state of the encoder and generated the predictions for the 60 unknown days.

### Data

We used a window of 90 elements for training and 60 for prediction. We split the data accordingly in training and validation using a walk-forward split. 

We calculated lagging features to avoid training the LSTMs with large sequences. We included three lagging features for each element in the series:
+ 360 days
+ 180 days
+ 90 days

We also normalized the data subtracting the mean and diving by the standard deviation. 

## Results

We obtained our best result using the sequence to sequence model trained for 16 epochs with a walk-forward split. With this approach, we got a 43.02 on Kaggle. 