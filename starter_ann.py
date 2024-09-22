import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

def encode_onehot(labels):
    from sklearn.preprocessing import OneHotEncoder
    onehotencoder = OneHotEncoder(sparse_output=False)
    encoded_categories = onehotencoder.fit_transform(labels.reshape(-1, 1))

    return encoded_categories

def get_data(name):
    from ucimlrepo import fetch_ucirepo

    features, targets = np.array([]), np.array([])

    # fetch dataset
    if name == "yeast":
        dataset = fetch_ucirepo(id=110)
        features = dataset.data.features.to_numpy()
        targets = dataset.data.targets.to_numpy()
        targets = targets.ravel()
    elif name == "spambase":
        dataset = fetch_ucirepo(id=94)
        features = dataset.data.features.to_numpy()
        targets = dataset.data.targets.to_numpy()
        targets = targets.ravel()
    elif name == "rice":
        dataset = fetch_ucirepo(id=545)
        features = dataset.data.features.to_numpy()
        targets = dataset.data.targets.to_numpy()
        targets = targets.ravel()
    elif name == "churn":
        dataset = fetch_ucirepo(id=563)
        features = dataset.data.features.to_numpy()
        targets = dataset.data.targets.to_numpy()
        targets = targets.ravel()
    elif name == "health_nutri":
        dataset = fetch_ucirepo(id=887)
        features = dataset.data.features.to_numpy()
        targets = dataset.data.targets.to_numpy()
        targets = targets.ravel()
    elif name == "monk":  # bean 602, creditCard 350, iono52,
        dataset = fetch_ucirepo(id=70)
        features = dataset.data.features.to_numpy()
        targets = dataset.data.targets.to_numpy()
        targets = targets.ravel()
    elif name == "marketing":  # awesome for ANN
        data = pd.read_csv('marketing_campaign.csv', sep="\t")
        # drop missing values
        data = data.dropna()

        features = data.iloc[:, :-1].to_numpy()
        targets = data.iloc[:, -1].to_numpy()
        targets = targets.ravel()

        # Encoding
        from sklearn.preprocessing import OneHotEncoder
        onehotencoder = OneHotEncoder(sparse_output=False)

        encoded = onehotencoder.fit_transform(features[:, 2].reshape(-1, 1))

        ft1 = np.concatenate((features[:, :2], encoded, features[:, 3:]), axis=1)
        # now we encode the marital status
        encoded = onehotencoder.fit_transform(ft1[:, 7].reshape(-1, 1))
        ft2 = np.concatenate((ft1[:, :7], encoded, ft1[:, 8:]), axis=1)

        # removing the date column
        ft2 = np.concatenate((ft2[:, :18], ft2[:, 19:]), axis=1)
        features = ft2


    return features, targets

def run(data_name, encode_type=None):
    import random
    seed_value = 0
    # 1. Set seed for Python's random module
    random.seed(seed_value)
    # 2. Set seed for NumPy
    np.random.seed(seed_value)

    # 3. Set seed for TensorFlow
    tf.random.set_seed(seed_value)


    X, y = get_data(data_name)
    test_size = 0.40

    if encode_type == "onehot":
        y = encode_onehot(y)
    elif encode_type == "label":
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)

    # splitting into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

    # feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)



    # run grid_search for hyperparameter tuning.

    # create the ANN model



    # initialize the ann
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units=X.shape[1], activation='relu')) # input and first hidden layer
    #ann.add(tf.keras.layers.Dense(units=39, activation='relu'))  # 2nd hidden layer
    #ann.add(tf.keras.layers.Dense(units=64, activation='relu'))  # 3rd hidden layer
    # ann.add(tf.keras.layers.Dense(units=64, activation='relu'))  # 2nd hidden layer
    # ann.add(tf.keras.layers.Dense(units=64, activation='relu'))  # 3rd hidden layer

    if y.ndim == 1:
        ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
        ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        ann.add(tf.keras.layers.Dense(units=y.shape[1], activation='softmax'))
        ann.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    ann.fit(X_train, y_train, batch_size=16, epochs=47, shuffle=False)

    # Predictions
    y_pred_train = ann.predict(X_train)

    # Evaluate the model
    loss_train, accuracy_train = ann.evaluate(X_train, y_train)
    loss_test, accuracy_test = ann.evaluate(X_test, y_test)
    print(f"Train Accuracy: {accuracy_train:.2f}")
    print(f"Test Accuracy: {accuracy_test:.2f}")

if __name__ == '__main__':
    print("Running ANN template")
    #run("yeast", "onehot")
    #run("spambase")
    #run("rice", encode_type="label")
    #run("churn")
    #run("health_nutri", encode_type="label")
    #run("marketing")
    run("monk")


# YEAST
    # .6, .61
    # .66, .6 at 64
# SPAMBASE
    # 0.99, 0.94

# RICE
  # .93, .93 epoch 50

# CHURN
  # .95, .94
  # .97, .94 100epcohs 32
  # .97, .95 100epochs, dense64
  # .98, .96 100epochs dense128

# health_nutri
  # .86, .84 100epochs, D64
  # .88, .84 100Ep, D128 x2
  # .90, .82 with 100Ep, D128 x3
  # .91, .81 with 100Ep, D128 x3
  # .91, .84 100ep, d128x3 + d32
  # .90, .85 for 80Ep, d128x3 + d32
  # .95, .81 for 100Ep, d128x4
  # .92, .84 for 90Ep, d128x4

# MONK:
  # ann (81, 69),

# Best trial:
#   Value: 0.7816091775894165
#   Params:
#     units: 14
#     optimizer: rmsprop
#     epochs: 43
#     batch_size: 18