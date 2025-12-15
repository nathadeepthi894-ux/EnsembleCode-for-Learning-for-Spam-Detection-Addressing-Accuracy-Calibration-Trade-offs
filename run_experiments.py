#!/usr/bin/env python3
"""
Comprehensive Experimental Validation for Improved Manuscript
Runs all experiments to replace placeholder data with real results
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, roc_curve, brier_score_loss)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from scipy import stats
import pickle
import json
import time
import warnings
warnings.filterwarnings('ignore')

class SpamDetectionExperiments:
    def __init__(self):
        self.results = {}
        self.models = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_and_preprocess_data(self):
        """Load Spambase dataset and preprocess"""
        print("="*80)
        print("STEP 1: Data Loading and Preprocessing")
        print("="*80)
        
        df = pd.read_csv('spambase.csv')
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Spam: {y.sum()} ({y.mean():.2%}), Legitimate: {len(y)-y.sum()} ({1-y.mean():.2%})")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Standardization
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_test = scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"Train: {len(y_train)}, Test: {len(y_test)}")
        print("Standardization complete\n")
        
        return scaler
    
    def train_base_classifiers(self):
        """Train all base classifiers"""
        print("="*80)
        print("STEP 2: Training Base Classifiers")
        print("="*80)
        
        classifiers = {
            'Naive Bayes': GaussianNB(),
            'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
            'SVM': SVC(kernel='rbf', C=10, gamma=0.01, probability=True, random_state=42),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=20, min_samples_split=5,
                min_samples_leaf=2, random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5,
                subsample=0.8, random_state=42
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=6,
                min_child_weight=1, random_state=42, eval_metric='logloss'
            )
        }
        
        for name, clf in classifiers.items():
            print(f"\nTraining {name}...")
            start_time = time.time()
            
            clf.fit(self.X_train, self.y_train)
            train_time = time.time() - start_time
            
            self.models[name] = clf
            
            # Predictions
            y_train_pred = clf.predict(self.X_train)
            y_test_pred = clf.predict(self.X_test)
            y_test_proba = clf.predict_proba(self.X_test)[:, 1]
            
            # Metrics
            self.results[name] = {
                'train_accuracy': float(accuracy_score(self.y_train, y_train_pred)),
                'test_accuracy': float(accuracy_score(self.y_test, y_test_pred)),
                'precision': float(precision_score(self.y_test, y_test_pred)),
                'recall': float(recall_score(self.y_test, y_test_pred)),
                'f1_score': float(f1_score(self.y_test, y_test_pred)),
                'roc_auc': float(roc_auc_score(self.y_test, y_test_proba)),
                'confusion_matrix': confusion_matrix(self.y_test, y_test_pred).tolist(),
                'train_time': train_time,
                'y_test_pred': y_test_pred.tolist(),
                'y_test_proba': y_test_proba.tolist()
            }
            
            print(f"  Train: {self.results[name]['train_accuracy']:.4f} | Test: {self.results[name]['test_accuracy']:.4f}")
            print(f"  Precision: {self.results[name]['precision']:.4f} | Recall: {self.results[name]['recall']:.4f}")
            print(f"  F1: {self.results[name]['f1_score']:.4f} | AUC: {self.results[name]['roc_auc']:.4f}")
            print(f"  Time: {train_time:.2f}s")
        
        print("\nBase classifier training complete!")
    
    def train_ensemble_methods(self):
        """Train ensemble methods"""
        print("\n" + "="*80)
        print("STEP 3: Training Ensemble Methods")
        print("="*80)
        
        # Get base classifiers (excluding XGBoost for ensemble)
        base_clfs = [
            ('nb', self.models['Naive Bayes']),
            ('lr', self.models['Logistic Regression']),
            ('svm', self.models['SVM']),
            ('rf', self.models['Random Forest']),
            ('gb', self.models['Gradient Boosting'])
        ]
        
        # Hard Voting
        print("\nTraining Hard Voting Ensemble...")
        hard_voting = VotingClassifier(estimators=base_clfs, voting='hard')
        start_time = time.time()
        hard_voting.fit(self.X_train, self.y_train)
        train_time = time.time() - start_time
        
        y_train_pred = hard_voting.predict(self.X_train)
        y_test_pred = hard_voting.predict(self.X_test)
        # Hard voting doesn't support predict_proba
        
        self.models['Hard Voting'] = hard_voting
        self.results['Hard Voting'] = {
            'train_accuracy': float(accuracy_score(self.y_train, y_train_pred)),
            'test_accuracy': float(accuracy_score(self.y_test, y_test_pred)),
            'precision': float(precision_score(self.y_test, y_test_pred)),
            'recall': float(recall_score(self.y_test, y_test_pred)),
            'f1_score': float(f1_score(self.y_test, y_test_pred)),
            'roc_auc': None,  # Not available for hard voting
            'confusion_matrix': confusion_matrix(self.y_test, y_test_pred).tolist(),
            'train_time': train_time,
            'y_test_pred': y_test_pred.tolist(),
            'y_test_proba': []  # Not available for hard voting
        }
        
        print(f"  Test Accuracy: {self.results['Hard Voting']['test_accuracy']:.4f}")
        print(f"  F1-Score: {self.results['Hard Voting']['f1_score']:.4f}")
        
        # Soft Voting
        print("\nTraining Soft Voting Ensemble...")
        soft_voting = VotingClassifier(estimators=base_clfs, voting='soft')
        start_time = time.time()
        soft_voting.fit(self.X_train, self.y_train)
        train_time = time.time() - start_time
        
        y_train_pred = soft_voting.predict(self.X_train)
        y_test_pred = soft_voting.predict(self.X_test)
        y_test_proba = soft_voting.predict_proba(self.X_test)[:, 1]
        
        self.models['Soft Voting'] = soft_voting
        self.results['Soft Voting'] = {
            'train_accuracy': float(accuracy_score(self.y_train, y_train_pred)),
            'test_accuracy': float(accuracy_score(self.y_test, y_test_pred)),
            'precision': float(precision_score(self.y_test, y_test_pred)),
            'recall': float(recall_score(self.y_test, y_test_pred)),
            'f1_score': float(f1_score(self.y_test, y_test_pred)),
            'roc_auc': float(roc_auc_score(self.y_test, y_test_proba)),
            'confusion_matrix': confusion_matrix(self.y_test, y_test_pred).tolist(),
            'train_time': train_time,
            'y_test_pred': y_test_pred.tolist(),
            'y_test_proba': y_test_proba.tolist()
        }
        
        print(f"  Test Accuracy: {self.results['Soft Voting']['test_accuracy']:.4f}")
        print(f"  F1-Score: {self.results['Soft Voting']['f1_score']:.4f}")
        
        # Stacking with Logistic Regression
        print("\nTraining Stacking (LR meta-learner)...")
        stacking_lr = StackingClassifier(
            estimators=base_clfs,
            final_estimator=LogisticRegression(random_state=42),
            cv=5
        )
        start_time = time.time()
        stacking_lr.fit(self.X_train, self.y_train)
        train_time = time.time() - start_time
        
        y_train_pred = stacking_lr.predict(self.X_train)
        y_test_pred = stacking_lr.predict(self.X_test)
        y_test_proba = stacking_lr.predict_proba(self.X_test)[:, 1]
        
        self.models['Stacking (LR)'] = stacking_lr
        self.results['Stacking (LR)'] = {
            'train_accuracy': float(accuracy_score(self.y_train, y_train_pred)),
            'test_accuracy': float(accuracy_score(self.y_test, y_test_pred)),
            'precision': float(precision_score(self.y_test, y_test_pred)),
            'recall': float(recall_score(self.y_test, y_test_pred)),
            'f1_score': float(f1_score(self.y_test, y_test_pred)),
            'roc_auc': float(roc_auc_score(self.y_test, y_test_proba)),
            'confusion_matrix': confusion_matrix(self.y_test, y_test_pred).tolist(),
            'train_time': train_time,
            'y_test_pred': y_test_pred.tolist(),
            'y_test_proba': y_test_proba.tolist()
        }
        
        print(f"  Test Accuracy: {self.results['Stacking (LR)']['test_accuracy']:.4f}")
        print(f"  F1-Score: {self.results['Stacking (LR)']['f1_score']:.4f}")
        
        # Stacking with Gradient Boosting
        print("\nTraining Stacking (GB meta-learner)...")
        stacking_gb = StackingClassifier(
            estimators=base_clfs,
            final_estimator=GradientBoostingClassifier(random_state=42),
            cv=5
        )
        start_time = time.time()
        stacking_gb.fit(self.X_train, self.y_train)
        train_time = time.time() - start_time
        
        y_train_pred = stacking_gb.predict(self.X_train)
        y_test_pred = stacking_gb.predict(self.X_test)
        y_test_proba = stacking_gb.predict_proba(self.X_test)[:, 1]
        
        self.models['Stacking (GB)'] = stacking_gb
        self.results['Stacking (GB)'] = {
            'train_accuracy': float(accuracy_score(self.y_train, y_train_pred)),
            'test_accuracy': float(accuracy_score(self.y_test, y_test_pred)),
            'precision': float(precision_score(self.y_test, y_test_pred)),
            'recall': float(recall_score(self.y_test, y_test_pred)),
            'f1_score': float(f1_score(self.y_test, y_test_pred)),
            'roc_auc': float(roc_auc_score(self.y_test, y_test_proba)),
            'confusion_matrix': confusion_matrix(self.y_test, y_test_pred).tolist(),
            'train_time': train_time,
            'y_test_pred': y_test_pred.tolist(),
            'y_test_proba': y_test_proba.tolist()
        }
        
        print(f"  Test Accuracy: {self.results['Stacking (GB)']['test_accuracy']:.4f}")
        print(f"  F1-Score: {self.results['Stacking (GB)']['f1_score']:.4f}")
        print(f"  Recall: {self.results['Stacking (GB)']['recall']:.4f}")
        
        print("\nEnsemble training complete!")
    
    def perform_cross_validation(self):
        """Perform 10-fold cross-validation"""
        print("\n" + "="*80)
        print("STEP 4: 10-Fold Cross-Validation")
        print("="*80)
        
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        # Combine train and test for CV
        X_full = np.vstack([self.X_train, self.X_test])
        y_full = np.concatenate([self.y_train, self.y_test])
        
        cv_results = {}
        
        for name, model in self.models.items():
            if name in ['Hard Voting', 'Soft Voting', 'Stacking (LR)', 'Stacking (GB)']:
                continue  # Skip ensembles for now (they take longer)
            
            print(f"\nCross-validating {name}...")
            scores = cross_val_score(model, X_full, y_full, cv=cv, scoring='accuracy', n_jobs=-1)
            
            mean_score = scores.mean()
            std_score = scores.std()
            ci_95 = 1.96 * std_score / np.sqrt(len(scores))
            
            cv_results[name] = {
                'mean_accuracy': float(mean_score),
                'std_accuracy': float(std_score),
                'ci_95': float(ci_95),
                'scores': scores.tolist()
            }
            
            print(f"  Mean Accuracy: {mean_score:.4f} ± {ci_95:.4f} (95% CI)")
        
        self.results['cross_validation'] = cv_results
        print("\nCross-validation complete!")
    
    def compute_calibration_metrics(self):
        """Compute calibration metrics for all classifiers"""
        print("\n" + "="*80)
        print("STEP 5: Calibration Analysis")
        print("="*80)
        
        calibration_results = {}
        
        for name in self.results.keys():
            if name == 'cross_validation':
                continue
            
            if name == 'Hard Voting':
                continue  # Hard voting doesn't output probabilities in the same way
            
            y_proba = np.array(self.results[name]['y_test_proba'])
            
            # Expected Calibration Error (ECE)
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0.0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_proba >= bin_lower) & (y_proba < bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = self.y_test[in_bin].mean()
                    avg_confidence_in_bin = y_proba[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            # Brier score
            brier = brier_score_loss(self.y_test, y_proba)
            
            calibration_results[name] = {
                'ece': float(ece),
                'brier_score': float(brier)
            }
            
            print(f"{name}: ECE = {ece:.4f}, Brier = {brier:.4f}")
        
        self.results['calibration'] = calibration_results
        print("\nCalibration analysis complete!")
    
    def save_results(self):
        """Save all results"""
        print("\n" + "="*80)
        print("Saving Results")
        print("="*80)
        
        with open('experimental_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        with open('trained_models.pkl', 'wb') as f:
            pickle.dump(self.models, f)
        
        print("Results saved to experimental_results.json")
        print("Models saved to trained_models.pkl")
    
    def run_all_experiments(self):
        """Run complete experimental pipeline"""
        self.load_and_preprocess_data()
        self.train_base_classifiers()
        self.train_ensemble_methods()
        self.perform_cross_validation()
        self.compute_calibration_metrics()
        self.save_results()
        
        print("\n" + "="*80)
        print("ALL EXPERIMENTS COMPLETE!")
        print("="*80)
        print("\nSummary of Results:")
        print(f"Base Classifiers: {len([k for k in self.results.keys() if k not in ['cross_validation', 'calibration', 'Hard Voting', 'Soft Voting', 'Stacking (LR)', 'Stacking (GB)']])}")
        print(f"Ensemble Methods: 4")
        print(f"Total Models: {len(self.models)}")

if __name__ == "__main__":
    experiments = SpamDetectionExperiments()
    experiments.run_all_experiments()
