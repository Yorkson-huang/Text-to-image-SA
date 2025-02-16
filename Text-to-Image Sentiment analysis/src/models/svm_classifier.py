from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import joblib

class SVMClassifier:
    def __init__(self):
        self.model = SVC(kernel='rbf', probability=True)
        
    def train(self, X, y):
        param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
        self.grid_search = GridSearchCV(
            self.model, 
            param_grid, 
            cv=5,
            n_jobs=-1
        )
        self.grid_search.fit(X, y)
        
    def save_model(self, path):
        joblib.dump(self.grid_search.best_estimator_, path)