import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
#Reads CSV with pokemon data
df = pd.read_csv('C:/Users/Ian/Desktop/Programming stuff/Tensorflow Pokemon Course/TensorFlow-Pokemon-Course/pokemon_alopez247.csv')
print("Done Reading CSV!") # Debugging line

#Creates an array of columns from the dataset
df = df[['isLegendary', 'Generation', 'Type_1', 'Type_2', 'HP', 'Attack', 'Defense',
         'Sp_Atk', 'Sp_Def', 'Speed', 'Color', 'Egg_Group_1', 'Height_m', 'Weight_kg', 'Body_Style']]
#Turns the isLegendary column into a numerical value (1 or 0) 
df['isLegendary'] = df['isLegendary'].astype(int)


#Creates Dummy Variables for each of the columns to be used in the model.... This is to turn the values that would otherwise not be numerical into numerical values
def dummy_creation(df, dummy_categories):
    for i in dummy_categories:
        df_dummy = pd.get_dummies(df[i])
        df = pd.concat([df,df_dummy], axis=1)
        df = df.drop(i, axis=1)
    return(df)
df = dummy_creation(df, ['Egg_Group_1', 'Body_Style', 'Color', 'Type_1', 'Type_2'])


#Splits data into testing and training datasets AND make anything with the generation of 1 part of the training dataset
def train_test_splitter(DataFrame, column):
    df_train = DataFrame.loc[df[column] != 1]
    df_test = DataFrame.loc[df[column] == 1]

    df_train = df_train.drop(column, axis=1)
    df_test = df_test.drop(column, axis=1)
    return(df_train, df_test)
df_train, df_test = train_test_splitter(df,'Generation')

#Separates the labels (the 'islegendary category) from the rest of the dataExtracts data from the dataframe and puts it into arrays that tensorflow can understand with .values
#Remember, this is the answer key to the test the algorithms are trying to solve, and it does no good to have them learn with the answer-key in (metaphorical) hand:
def label_delineator(df_train, df_test, label):
    train_data = df_train.drop(label, axis=1).values
    train_labels = df_train[label].values
    test_data = df_test.drop(label, axis=1).values
    test_labels = df_test[label].values
    return(train_data, train_labels, test_data, test_labels)
train_data, train_labels, test_data, test_labels = label_delineator(df_train, df_test, 'isLegendary')

#Normalize data so that everything is on the same scale
def data_normalizer(train_data, test_data):
    train_data = preprocessing.MinMaxScaler().fit_transform(train_data)
    test_data = preprocessing.MinMaxScaler().fit_transform(test_data)
    return(train_data, test_data)
train_data, test_data = data_normalizer(train_data, test_data)


#Creation of the model using Keras
#The number associated with the layer is the number of neurons in it. The first layer we'll use is a 'ReLU' (Rectified Linear Unit)' activation function. 
# Since this is also the first layer, we need to specify input_size, which is the shape of an entry in our dataset.
#After that, we'll finish with a softmax layer. Softmax is a type of logistic regression done for situations with multiple cases, 
# like our 2 possible groups: 'Legendary' and 'Not Legendary'. 
# With this we delineate the possible identities of the Pokémon into 2 probability groups corresponding to the possible labels:
length = train_data.shape[1]
model = keras.Sequential()
model.add(keras.layers.Dense(500, activation='relu', input_shape=[length,]))
model.add(keras.layers.Dense(2, activation='softmax'))
#print("Test Complete") # Debugging line
#Compiles the model:
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#The optimizer we're using is the Stochastic Gradient Descent (SGD) optimization algorithm, but there are others available. 
# For our loss we're using sparse_categorical_crossentropy. 
# If our values were one-hot encoded, we would want to use "categorial_crossentropy" instead.
#The three parameters model.fit needs are our training data, our training labels, and the number of epochs.
# One epoch is when the model has iterated over every sample once. Essentially the number of epochs is equal to the number of times we want to cycle through the data.
# We'll start with just 1 epoch, and then show that increasing the epoch improves the results.
#Now the model to fit the training:


model.fit(train_data, train_labels, epochs=400)  #Turned off while not training


#Tests current model against testing data
loss_value, accuracy_value = model.evaluate(test_data, test_labels)
print(f'Accuracy: {accuracy_value}')
#model.evaluate will evaluate how strong our model is with the test data, 
# and report that in the form of loss value and accuracy value(since we specified accuracy in our selected_metrics variable when we compiled the model). 
# We'll just focus on our accuracy for now. With an accuracy of ~98%, it's not perfect, but it's very accurate.
#We can also use our model to predict specific Pokémon, or at least have it tell us which status the Pokémon is most likely to have, with model.predict. 
# All it needs to predict a Pokémon is the data for that Pokémon itself. We're providing that by selecting a certain index of test_data:
def predictor(test_data, test_labels, index):
    prediction = model.predict(test_data)
    if np.argmax(prediction[index]) == test_labels[index]:
        print(f'This was correctly predicted to be a \"{test_labels[index]}\"!')
    else:
        print(f'This was incorrectly predicted to be a \"{np.argmax(prediction[index])}\". It was actually a \"{test_labels[index]}\"!')

predictor(test_data, test_labels, 149)