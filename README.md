# -Neural_Network_Charity_Analysis
Alphabet Soup, a philanthropic foundation dedicated to helping organizations that help the environment, improve people's overall wellbeing, and unify the world, has raised and donated over $10 Billion to oganizations in the last 20 years. Organizations have used these donations to invest in life-saving technologies and organize reforestation groups around the world. Alphabet Soup realizes the impact their donations can have and seeks to give to those companies that are most likely to put these donations to good use. Alphabet Soup's CEO has asked their data scientist group to analyze the company's past donation pool (a pool of 34,000 organizations) and develop a mathematical, data driven tool to pinpoint the best companies to provide donations. The data scientist team has determined they will utilize an advanced statistical modeling technique such as the deep learning Neural Network to guide Alphabet Soup in their future donations.

For this project, the data scientist team conducted the following:

>Preprocessed Data for a Neural Network Model

>Compiled, Trained, and Evaluated the Model

>Saved model weights and saved the results to file

>Summary

## Preprocessed Data for a Neural Network Model
Data Prepared for a Neural Network Model The Model Was Computed, Trained, and Evaluated Added model weights and results to a file. model optimization and comparisons with different machine learning models EIN and NAME columns have been removed. Columns with more than 10 unique values were grouped Categorical variables were coded using one-hot coding. Preprocessed data is split into features and target sequences .The preprocessed data is divided into a training data set and a test data set. Numbers were normalized with the StandardScaler() module

We determined that since our strategy is based on charity and funding is mostly dependent on performance and repayment capacity (which are occasionally related), the aim should be success. In order to reduce observation errors, we initially included all data for our report but binned two columns: categorization and application type. Once finished, the process of dividing the data set, scaling it, and then fitting the data was routine. The image below is our first neural network. Every five epochs, data was also saved.

## Compiled, Trained, and Evaluated the Model

Multiple approaches were used to enhance our model. Additional layers with more neurons were then added, 2 columns that would not provide pertinent information, such as "Use case other" and "Affil other," were removed, and finally, part of the hidden layers' activation functions were altered to sigmoid. It's crucial to note that our second model fared the best with an accuracy of 72%, even if each iteration of the new NN represents an overall improvement over the previous model.Number of layers, number of neurons per layer and activation function are defined.An output layer with an activation function is created. I have the output of the structure of the model and have model loss and accuracy issues. Model weights are saved every 5 epochs. Results are saved in HDF5 files

```
 Model 1 optimization

hidden_nodes_layer1_1 = 80
hidden_nodes_layer2_1 = 42

nn = tf.keras.models.Sequential()

First hidden layer
nn.add(
   tf.keras.layers.Dense(units=hidden_nodes_layer1_1, input_dim=number_input_features, activation="relu")
)

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2_1, activation="relu"))

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Compile the model
nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Callback that saves the weights every 5 epochs
fit_model= nn.fit(X_train_scaled, y_train, epochs=50, verbose=2)

```
## Saved model weights and saved the results to file
If the model is optimized and the prediction accuracy is improved to 75% or higher, then I have a working code that makes 3 tries to improve the performance of the model using the following steps:
>>Noisy variables are removed from the function
Additional neurons are added to the hidden layer
An additional hidden layer is added
Activation function for hidden or output layers changed for optimization
To decide on neuron number, arbitrary values were intially chosen and was primarily trial and error but unforunately could not reach 75% although progress is close.

```
Epoch 50/50
804/804 - 1s - loss: 0.5300 - accuracy: 0.7435 - 1s/epoch - 1ms/step
268/268 - 1s - loss: 0.5596 - accuracy: 0.7208 - 536ms/epoch - 2ms/step
Loss: 0.5596364140510559, Accuracy: 0.7208163142204285
```
## Summary
Overall, this session emphasizes one crucial point regarding NNs. Adding too many new adjustments is not usually a smart idea. The sigmoid function's addition significantly decreased our performance. Model 2 appears to have the most promising forecast, thus we need experiment further with it in terms of the amount of neurons in each layer in order to improve this model. We might also think about deleting some of the affiliation and use case columns. Overall, the model is promising, and further improvements can probably assist in reaching the 75% threshold.
```
Epoch 50/50
804/804 [==============================] - 16s 19ms/step - loss: 4352.1768 - accuracy: 0.5910
268/268 - 0s - loss: 1.6285 - accuracy: 0.6929 - 452ms/epoch - 2ms/step
Loss: 1.6284602880477905, Accuracy: 0.6929445862770081
Results were lack luster and only achieved in accuracy score of 69%.
```
Various approaches were attempted to enhance our model. 'Use case other' and 'Affil other' were eliminated as columns that would not include pertinent information. Next, further layers and neurons were created, and finally, some of the hidden layers' sigmoid activation functions were applied. It is crucial to note that our second model fared the best with an accuracy of 72%, even if each iteration of the new NN is an overall improvement from the previous model.


To decide on neuron number, arbitrary values were intially chosen and was primarily trial and error but unforunately could not reach 75% although progress is close.

Summary
With regard to NNs, this module emphasizes 1 key point. There can be a point where there are too many changes. Our performance was significantly lowered by the sigmoid function. We should experiment more with model 2, which appears to have the most promising prediction, in terms of the number of neurons in each layer in order to improve this model. The removal of additional affiliation and use case columns is another option. The model appears promising overall, and further improvements may enable it to pass the 75% threshold.
