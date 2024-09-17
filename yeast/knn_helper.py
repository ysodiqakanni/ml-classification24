
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