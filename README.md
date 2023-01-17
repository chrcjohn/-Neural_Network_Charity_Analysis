# -Neural_Network_Charity_Analysis
Alphabet Soup, a philanthropic foundation dedicated to helping organizations that help the environment, improve people's overall wellbeing, and unify the world, has raised and donated over $10 Billion to oganizations in the last 20 years. Organizations have used these donations to invest in life-saving technologies and organize reforestation groups around the world. Alphabet Soup realizes the impact their donations can have and seeks to give to those companies that are most likely to put these donations to good use. Alphabet Soup's CEO has asked their data scientist group to analyze the company's past donation pool (a pool of 34,000 organizations) and develop a mathematical, data driven tool to pinpoint the best companies to provide donations. The data scientist team has determined they will utilize an advanced statistical modeling technique such as the deep learning Neural Network to guide Alphabet Soup in their future donations.

For this project, the data scientist team conducted the following:

Preprocessed Data for a Neural Network Model
Compiled, Trained, and Evaluated the Model
Saved model weights and saved the results to file
Optimized the Model and made comparisons to other machine learning models

Since funding relies primarily factors such as success and ability to pay back (related sometimes), we decided the target for our model is success since it is charity based. To reach our output, we initially included all data but binned 2 columns: classification, and application type to minimize observation errors. Once completed, it was standard procedure with splitting the data set, scaling, and then fitting the data. Our initial neural network is shown below. Data was also saved every 5 epochs.

Since funding relies primarily factors such as success and ability to pay back (related sometimes), we decided the target for our model is success since it is charity based. To reach our output, we initially included all data but binned 2 columns: classification, and application type to minimize observation errors. Once completed, it was standard procedure with splitting the data set, scaling, and then fitting the data. Our initial neural network is shown below. Data was also saved every 5 epochs.


To improve our model several appraoches were taken. Removed 2 columns that may not provide relevant information such as 'Use case other' and 'Affil other', additional layers were then added with more neurons, and lastly, some of the activation functions of the hidden layers are changed to sigmoid. While each iteration of the new NN is an overall improvement from the first model, its important to note that our 2nd model performed the best with an accuracy of 72%.

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
To decide on neuron number, arbitrary values were intially chosen and was primarily trial and error but unforunately could not reach 75% although progress is close.
```
Epoch 50/50
804/804 - 1s - loss: 0.5300 - accuracy: 0.7435 - 1s/epoch - 1ms/step
268/268 - 1s - loss: 0.5596 - accuracy: 0.7208 - 536ms/epoch - 2ms/step
Loss: 0.5596364140510559, Accuracy: 0.7208163142204285
```
Summary
Overall, this module highlights 1 important note when it comes to NN's. Too many additional changes aren't always good. The addition of sigmoid function greatly reduced our performance. To better improve this model, we should play around more with model 2 in terms of neuron numbers in each layer as it seems to have the most promising prediction. We can also consider removing other columns in affiliation and use case. Overall the model shows good promise and further optimizations can likely help reach the 75% thresh-hold
```
Epoch 50/50
804/804 [==============================] - 16s 19ms/step - loss: 4352.1768 - accuracy: 0.5910
268/268 - 0s - loss: 1.6285 - accuracy: 0.6929 - 452ms/epoch - 2ms/step
Loss: 1.6284602880477905, Accuracy: 0.6929445862770081
Results were lack luster and only achieved in accuracy score of 69%.
```
To improve our model several appraoches were taken. Removed 2 columns that may not provide relevant information such as 'Use case other' and 'Affil other', additional layers were then added with more neurons, and lastly, some of the activation functions of the hidden layers are changed to sigmoid. While each iteration of the new NN is an overall improvement from the first model, its important to note that our 2nd model performed the best with an accuracy of 72%.



To decide on neuron number, arbitrary values were intially chosen and was primarily trial and error but unforunately could not reach 75% although progress is close.

Summary
Overall, this module highlights 1 important note when it comes to NN's. Too many additional changes aren't always good. The addition of sigmoid function greatly reduced our performance. To better improve this model, we should play around more with model 2 in terms of neuron numbers in each layer as it seems to have the most promising prediction. We can also consider removing other columns in affiliation and use case. Overall the model shows good promise and further optimizations can likely help reach the 75% thresh-hold
