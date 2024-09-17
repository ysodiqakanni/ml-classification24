import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

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
    test_size = 0.30

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
    ann.add(tf.keras.layers.Dense(units=128, activation='relu'))  # 2nd hidden layer
    ann.add(tf.keras.layers.Dense(units=128, activation='relu'))  # 3rd hidden layer
    ann.add(tf.keras.layers.Dense(units=128, activation='relu'))  # 2nd hidden layer
    ann.add(tf.keras.layers.Dense(units=128, activation='relu'))  # 3rd hidden layer

    if y.ndim == 1:
        ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
        ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        ann.add(tf.keras.layers.Dense(units=y.shape[1], activation='softmax'))
        ann.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    ann.fit(X_train, y_train, batch_size=32, epochs=90, shuffle=False)

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
    run("health_nutri", encode_type="label")


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