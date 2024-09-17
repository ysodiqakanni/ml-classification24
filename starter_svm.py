import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

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

def run_grid_search(xtrain, ytrain):
    from sklearn.model_selection import GridSearchCV
    svc = SVC()

    # Define the hyperparameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],  # Regularization parameter
        'kernel': ['linear', 'rbf', 'poly'],  # Kernel types to test
        'gamma': ['scale', 'auto'],  # Gamma values for rbf/poly kernels
        'degree': [2, 3, 4],  # Degree for 'poly' kernel
        'coef0': [0.0, 0.1, 0.5]  # Coefficient for 'poly' and 'sigmoid'
    }

    # Set up the grid search with cross-validation
    grid_search = GridSearchCV(svc, param_grid, cv=4, scoring='accuracy', n_jobs=-1)

    # Fit the model
    grid_search.fit(xtrain, ytrain)

    return grid_search.best_params_, grid_search.best_score_

### program begins here
def run(data_name):
    X, y = get_data(data_name)
    test_size = 0.20

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


    # training on train set
    classifier = SVC(kernel="rbf", degree=7, random_state=0)
    #classifier = SVC(kernel="poly", degree=4, random_state=0)
    #classifier = SVC(kernel="rbf", degree=3, random_state=0)

    classifier.fit(X_train, y_train)

    # predicting on training and test sets
    y_pred_train = classifier.predict(X_train)
    y_pred_test = classifier.predict(X_test)

    print("pridiction done.")
    #cm_train = confusion_matrix(y_train, y_pred_train)
    print(f"Insample prediction for {data_name}= {accuracy_score(y_train, y_pred_train)}")
    print(f"Out-sample prediction for {data_name}= {accuracy_score(y_test, y_pred_test)}")



if __name__ == "__main__":
    print("Running svm template")
    #run("yeast")
    #run("spambase")
    #run("rice")
    #run("churn")
    run("health_nutri")




# Notes for parameter tuning
# YEAST:
    # grid search returned: Grid search done. Best params: {'C': 1, 'coef0': 0.5, 'degree': 2, 'gamma': 'auto', 'kernel': 'poly'}, Best score: 0.5847898722898722
    # Poly3, ts20: .58, .51
    # rbf, ts30: .64, .63. ts25 good too.

# SPAMBASE:
    # poly2 ts20, .85, .83
    # poly1 ts20: .92, .91
    # rbf ts20: .95, .93

# RICE
  # .93, .94

# CHURN
  # .93, .91 k3
  # 94, .92 poly5

# health_nutri:
  #.84, .84 for linear
  # .85, .85 for rbf