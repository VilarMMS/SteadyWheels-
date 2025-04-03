import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, precision_recall_curve
import optuna
import joblib
import time
import warnings
warnings.filterwarnings("ignore")        
import lightgbm as lgb
from sklearn.metrics import roc_curve

class LightGBM:
    """
    A class that builds, trains and evaluates a LightGBM classifier
    optimized for a custom cost function using Bayesian optimization (Optuna).
    
    The cost function takes into account the different costs associated with
    false negatives, false positives, and true positives.
    
    This implementation uses fixed training and validation sets provided by the user
    instead of performing cross-validation splits.
    """
    
    def __init__(self, n_trials=100, random_state=42):
        """
        Initialize the CostOptimizedLightGBM.
        
        Parameters:
        -----------
        n_trials : int
            Number of trials for Optuna optimization (default: 100)
        random_state : int
            Random seed for reproducibility (default: 42)
        """
        self.n_trials = n_trials
        self.random_state = random_state
        self.best_params = None
        self.best_model = None
        self.study = None
        self.feature_importances = None
        
    def calculate_cost(self, y_true, y_pred, is_proba=False, dtrain=None):
        """
        Calculate custom cost function based on confusion matrix.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels or probabilities
        is_proba: bool
            if y_pred is probability
        dtrain: lgb.Dataset
            Dataset containing true labels
            
        Returns:
        --------
        float
            Cost value (lower is better)
        """
        w_FN = 500  # Cost of false negative
        w_TP = 25   # Cost of true positive (processing cost)
        w_FP = 10   # Cost of false positive
        
        if is_proba:
            y_pred_binary = (y_pred > 0.5).astype(int)
        else:
            y_pred_binary = y_pred

        if dtrain is not None:
            y_true = dtrain.get_label()
        
        # Calculate confusion matrix elements manually
        tp = np.sum((y_true == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true == 1) & (y_pred_binary == 0))
        
        cost = fp * w_FP + fn * w_FN + tp * w_TP
        return cost
    
    def _objective(self, trial, X_train, y_train, X_val, y_val):
        """
        Objective function for Optuna optimization.
        
        Parameters:
        -----------
        trial : optuna.trial.Trial
            Optuna trial object
        X_train : DataFrame
            Training feature matrix
        y_train : Series
            Training target vector
        X_val : DataFrame
            Validation feature matrix
        y_val : Series
            Validation target vector
            
        Returns:
        --------
        float
            Negative cost on validation set (higher is better for optuna)
        """
        # Define hyperparameters to optimize for LightGBM
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }
                
        # Create LightGBM classifier with suggested hyperparameters
        # Include class weights to reflect cost ratio between FN and FP
        w_FN = 500
        w_FP = 10
        
        # Use LightGBM's native API for training with custom eval
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        # Prepare parameters for training
        lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'seed': self.random_state,
            'scale_pos_weight': w_FN/w_FP,
            'learning_rate': params['learning_rate'],
            'max_depth': params['max_depth'],
            'num_leaves': params['num_leaves'],
            'min_child_samples': params['min_child_samples'],
            'subsample': params['subsample'],
            'colsample_bytree': params['colsample_bytree'],
            'reg_alpha': params['reg_alpha'],
            'reg_lambda': params['reg_lambda'],
        }
        
        # Train with custom eval function
        gbm = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=params['n_estimators'],
            valid_sets=[dval],
            feval=lambda y_pred, dtrain: ('custom_cost', self.calculate_cost(y_val, y_pred, True, dtrain), False)
        )
        
        # Use the model to make predictions directly with the booster
        y_pred_proba = gbm.predict(X_val)
        y_pred = (y_pred_proba > 0.5).astype(int)
        cost = self.calculate_cost(y_val, y_pred)
        
        # Return negative cost (Optuna minimizes, but we want to minimize cost)
        return -cost
    
    def optimize(self, X_train, y_train, X_val, y_val):
        """
        Run hyperparameter optimization using Optuna with fixed validation set.
        
        Parameters:
        -----------
        X_train : DataFrame
            Training feature matrix
        y_train : Series
            Training target vector
        X_val : DataFrame
            Validation feature matrix
        y_val : Series
            Validation target vector
            
        Returns:
        --------
        dict
            Best hyperparameters found
        """
        # Make sure inputs are pandas DataFrame/Series
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
        if not isinstance(y_train, pd.Series):
            y_train = pd.Series(y_train)
        if not isinstance(X_val, pd.DataFrame):
            X_val = pd.DataFrame(X_val)
        if not isinstance(y_val, pd.Series):
            y_val = pd.Series(y_val)
        
        # Create Optuna study
        study = optuna.create_study(direction='maximize')  # Maximize negative cost
        
        # Optimize
        print(f"Starting hyperparameter optimization with {self.n_trials} trials...")
        start_time = time.time()
        
        study.optimize(
            lambda trial: self._objective(trial, X_train, y_train, X_val, y_val),
            n_trials=self.n_trials
        )
        
        elapsed_time = time.time() - start_time
        print(f"Optimization completed in {elapsed_time:.2f} seconds.")
        
        # Store best parameters and study
        self.best_params = study.best_params
        self.study = study
        
        # Print best parameters
        print("\nBest hyperparameters found:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        
        # Build and fit best model
        self._build_best_model()
        self.fit(X_train, y_train, X_val, y_val)
        
        return self.best_params
    
    def _build_best_model(self):
        """Build the model with the best hyperparameters."""
        if self.best_params is None:
            raise ValueError("No best parameters found. Call optimize() first.")
            
        # Set class weights to reflect the cost ratio
        w_FN = 500
        w_FP = 10
        
        self.best_model = lgb.LGBMClassifier(
            random_state=self.random_state,
            class_weight={0: 1, 1: w_FN/w_FP},
            **self.best_params
        )
        
        return self.best_model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Fit the best model on training data with optional validation set for early stopping.
        
        Parameters:
        -----------
        X_train : DataFrame
            Training feature matrix
        y_train : Series
            Training target vector
        X_val : DataFrame, optional
            Validation feature matrix for early stopping
        y_val : Series, optional
            Validation target vector for early stopping
            
        Returns:
        --------
        self
            Fitted model instance
        """
        if self.best_model is None:
            raise ValueError("Model not optimized yet. Call optimize() first.")
        
        # Define eval_set for early stopping if validation data is provided
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        # Fit model with or without early stopping
        if eval_set:
            # Convert to LightGBM dataset format to use custom eval
            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            
            # Use LightGBM's native API for training with custom eval
            lgb_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'verbosity': -1,
                'seed': self.random_state,
                'scale_pos_weight': 500/10,  # FN/FP cost ratio
                'learning_rate': self.best_params['learning_rate'],
                'max_depth': self.best_params['max_depth'],
                'num_leaves': self.best_params['num_leaves'],
                'min_child_samples': self.best_params['min_child_samples'],
                'subsample': self.best_params['subsample'],
                'colsample_bytree': self.best_params['colsample_bytree'],
                'reg_alpha': self.best_params['reg_alpha'],
                'reg_lambda': self.best_params['reg_lambda'],
            }
            
            # Train with custom eval function
            gbm = lgb.train(
                lgb_params,
                dtrain,
                num_boost_round=self.best_params.get('n_estimators', 100),
                valid_sets=[dval],
                feval=lambda y_pred, dtrain: ('custom_cost', self.calculate_cost(y_val, y_pred, True, dtrain), False)
            )
            
            # Create a new classifier
            self.best_model = lgb.LGBMClassifier(**self.best_params)
            
            # Manually fit to make sure the sklearn wrapper is properly initialized
            self.best_model.fit(X_train, y_train)
            
            # Replace the booster with our custom-trained one
            self.best_model._Booster = gbm
        else:
            # Just fit without early stopping
            self.best_model.fit(X_train, y_train)
        
        # Get and store feature importances
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importances = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the best model.
        
        Parameters:
        -----------
        X : DataFrame
            Feature matrix
            
        Returns:
        --------
        array
            Predicted class labels
        """
        if self.best_model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        return self.best_model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the best model.
        
        Parameters:
        -----------
        X : DataFrame
            Feature matrix
            
        Returns:
        --------
        array
            Predicted class probabilities
        """
        if self.best_model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        return self.best_model.predict_proba(X)
    
    def evaluate(self, X_test, y_test, threshold=0.5, plot=True):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        X_test : DataFrame
            Test feature matrix
        y_test : Series
            Test target vector
        threshold : float
            Probability threshold for positive class (default: 0.5)
        plot : bool
            Whether to generate evaluation plots (default: True)
            
        Returns:
        --------
        dict
            Evaluation metrics
        """
        if self.best_model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Get predictions
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        cm = confusion_matrix(y_test, y_pred)
        cost = self.calculate_cost(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Store and print results
        results = {
            'cost': cost,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'classification_report': report,
            'threshold': threshold
        }
        
        print(f"\nModel Evaluation Results (threshold={threshold:3f}):")
        print(f"  Cost function: {cost}")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  ROC AUC: {roc_auc:.3f}")
        print(f"  Precision (Class 1): {report['1']['precision']:.3f}")
        print(f"  Recall (Class 1): {report['1']['recall']:.3f}")
        print(f"  F1 Score (Class 1): {report['1']['f1-score']:.3f}")
        
        # Add confusion matrix plot
        if plot:
            plt.figure(figsize=(10, 8))
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['0', '1'],
                        yticklabels=['0', '1'])
            
            plt.title(f'Confusion Matrix (threshold={threshold:.3f})', fontsize=16)
            plt.ylabel('True Label', fontsize=14)
            plt.xlabel('Predicted Label', fontsize=14)
            plt.tight_layout()
            plt.show()
            
            # Plot ROC curve with requested modifications
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            plt.figure(figsize=(8, 6))
            
            # Remove frame
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            # Plot ROC curve and diagonal reference line
            plt.plot(fpr, tpr, color='blue', lw=2)
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
            
            # Set axis limits and ticks
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xticks([0, 1])
            plt.yticks([0, 1])
            
            # Add labels and title with AUC value
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title(f'ROC curve (AUC = {roc_auc:.3f})', fontsize=14)
            
            # Remove grid
            plt.grid(False)
            
            plt.tight_layout()
            plt.show()
                
        return results
    
    def find_optimal_threshold(self, X_val, y_val, thresholds=None):
        """
        Find the optimal probability threshold to minimize cost.
        
        Parameters:
        -----------
        X_val : DataFrame
            Validation feature matrix
        y_val : Series
            Validation target vector
        thresholds : array-like, optional
            List of thresholds to try (default: np.arange(0.1, 0.95, 0.05))
            
        Returns:
        --------
        float
            Optimal threshold that minimizes cost
        """
        if self.best_model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        if thresholds is None:
            # More points for a smoother plot
            thresholds = np.arange(0.001, 0.1, 0.0005)
            
        y_pred_proba = self.best_model.predict_proba(X_val)[:, 1]
        costs = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            cost = self.calculate_cost(y_val, y_pred)
            costs.append(cost)
            
        # Find threshold with minimum cost
        best_idx = np.argmin(costs)
        best_threshold = thresholds[best_idx]
        min_cost = costs[best_idx]
        
        # Plot cost vs threshold
        plt.figure(figsize=(10, 6))
        
        # Create a smoother plot
        plt.plot(thresholds, costs, '-', linewidth=2)
        
        # Set x-axis with only 3 specified ticks
        plt.xlabel('Threshold')
        plt.xlim(0, 0.1)
        plt.xticks([0, 0.05, 0.1])
        
        # Set y-axis with specific ticks in thousands
        y_ticks = [20000, 30000, 40000]
        plt.yticks(y_ticks)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1000)}"))
        plt.ylabel('Cost (thousands USD $)')
        
        # Remove the plot frame but keep the axes
        for spine in ['top', 'right']:
            plt.gca().spines[spine].set_visible(False)
        
        # Add annotation in red with thousands notation and $ symbol
        plt.annotate(f'Optmized threshold: {best_threshold:.2f}\nCost: ${int(min_cost):,}',
                    xy=(best_threshold, min_cost),
                    xytext=(best_threshold + 0.01, min_cost),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    color='red',
                    fontweight='bold')
        
        plt.title('Cost vs Threshold')
        plt.show()
        
        return best_threshold
                
    def save_model(self, filepath):
        """Save the model to a file."""
        if self.best_model is None:
            raise ValueError("No model to save. Call fit() first.")
            
        joblib.dump(self.best_model, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load a saved model."""
        model_instance = cls()
        model_instance.best_model = joblib.load(filepath)
        return model_instance