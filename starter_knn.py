import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd

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
    elif name == "monk":  # bean 602, creditCard 350, 70 for monk
        dataset = fetch_ucirepo(id=70)
        features = dataset.data.features.to_numpy()
        targets = dataset.data.targets.to_numpy()
        targets = targets.ravel()
    elif name == "thyroid":  #useless
        data = pd.read_csv('ann-train.data', header=None)

        # Assuming the last column is the class label
        features = data.iloc[:, :-1]
        labels = data.iloc[:, -1]
        print("Features:\n", features.head())
        print("Labels:\n", labels.head())


    return features, targets

def run_grid_search(xtrain, ytrain):
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier()

    # Define the hyperparameter grid
    param_grid = {
        'n_neighbors': range(1, 31),  # Test different k values
        'weights': ['uniform', 'distance'],  # Test both uniform and distance weighting
        'p': [1, 2],  # Test Manhattan and Euclidean distance
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Test different search algorithms
        'leaf_size': [10, 20, 30, 40, 50]  # Different leaf sizes for tree-based methods
    }

    # Set up the grid search with cross-validation
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Fit the model
    grid_search.fit(xtrain, ytrain)

    return grid_search.best_params_, grid_search.best_score_

def run(data_name):
    X, y = get_data(data_name)
    test_size = 0.40

    # splitting into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

    # feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # run grid_search for hyperparameter tuning.
    # print("Running grid search")
    # best_params, best_scores = run_grid_search(X_train, y_train)
    # print(f"Grid search done. Best params: {best_params} \nBest score: {best_scores}")


    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=1)
    #classifier = KNeighborsClassifier(n_neighbors=5, weights="distance")
    # classifier = KNeighborsClassifier(n_neighbors=30, p=2, algorithm="auto", leaf_size=10, weights="distance") #{'algorithm': 'auto', 'leaf_size': 10, 'n_neighbors': 30, 'p': 1, 'weights': 'distance'}
    classifier.fit(X_train, y_train)

    # predicting on training and test sets
    y_pred_train = classifier.predict(X_train)
    y_pred_test = classifier.predict(X_test)

    print("pridiction done.")
    # cm_train = confusion_matrix(y_train, y_pred_train)
    print(f"Insample prediction for {data_name}= {accuracy_score(y_train, y_pred_train)}")
    print(f"Out-sample prediction for {data_name}= {accuracy_score(y_test, y_pred_test)}")

if __name__ == "__main__":
    print("Running knn template")
    #run("yeast")
    #run("spambase")
    #run("rice")
    #run("pmaintenance")
    run("monk")
    #run("health_nutri")


##
# YEAST (1484x8):
   # .69, .57 with k=5
   # .65, .6 at k=10

# SPAMBASE: .95, .91
# RICE
   # k20: 0.93, 0.94
   # k5: 0.93, 0.92
   # k3: .94, .92

# CHURN
  # .97, .94

# health_nutri:
  # .89, .82 for k3
  # .89, .83 for k5

  # monk (.94, .83) with k=5 and p=1
