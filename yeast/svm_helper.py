from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def run_grid_search(xtrain, ytrain):
    svc = SVC()

    # Define the hyperparameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],  # Regularization parameter
        'kernel': ['linear', 'rbf', 'poly'],  # Kernel types to test
        'gamma': ['scale', 'auto', 1e-3, 1e-4],  # Gamma values for rbf/poly kernels
        'degree': [2, 3, 4],  # Degree for 'poly' kernel
        'coef0': [0.0, 0.1, 0.5]  # Coefficient for 'poly' and 'sigmoid'
    }

    # Set up the grid search with cross-validation
    grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Fit the model
    grid_search.fit(xtrain, ytrain)

    return grid_search.best_params_, grid_search.best_score_