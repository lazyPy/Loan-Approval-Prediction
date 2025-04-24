import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, permutation_test_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report, precision_recall_curve, auc
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

def check_data_leakage(df):
    """Check for potential sources of data leakage."""
    print("\nChecking for potential data leakage:")
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    print(f"  - Duplicate rows: {duplicates}")
    
    # Check for constant or near-constant features
    constant_cols = []
    near_constant_cols = []
    for col in df.columns:
        unique_count = df[col].nunique()
        if unique_count == 1:
            constant_cols.append(col)
        elif unique_count == 2 and df[col].value_counts().iloc[1] / len(df) < 0.01:
            near_constant_cols.append(col)
    
    print(f"  - Constant columns: {constant_cols}")
    print(f"  - Near-constant columns: {near_constant_cols}")
    
    return {
        'duplicates': duplicates,
        'constant_cols': constant_cols,
        'near_constant_cols': near_constant_cols
    }

def load_and_preprocess_data(file_path, test_file_path=None):
    """Load and preprocess the loan approval dataset."""
    # Load the dataset
    df = pd.read_csv(file_path)
    
    print("Original dataset shape:", df.shape)

    # Convert 'Loan Status' to binary (1 for Approved, 0 for Declined)
    df['Loan Status'] = df['Loan Status'].apply(lambda x: 1 if x == 'Approved' else 0)

    # Check for data leakage
    leakage_report = check_data_leakage(df)
    
    # Remove duplicates
    df_deduped = df.drop_duplicates()
    print(f"Removed {df.shape[0] - df_deduped.shape[0]} duplicate rows")
    df = df_deduped
    
    # Remove near-constant columns
    near_constant_threshold = 0.01
    cols_to_drop = []
    for col in df.columns:
        if col != 'Loan Status':  # Don't drop the target variable
            # Check if the column has very low variance
            value_counts = df[col].value_counts(normalize=True)
            if value_counts.iloc[0] > (1 - near_constant_threshold):
                cols_to_drop.append(col)
    
    if cols_to_drop:
        print(f"Dropping near-constant columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    
    # Print dataset info after cleaning
    print(f"Dataset shape after cleaning: {df.shape}")
    print(f"Class distribution: {df['Loan Status'].value_counts()}")
    print(f"Percentage of approved loans: {df['Loan Status'].mean() * 100:.2f}%")

    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_cols.remove('Loan Status')  # Remove target variable
    
    # Calculate class weights for imbalance
    approved_count = df['Loan Status'].sum()
    declined_count = len(df) - approved_count
    total_samples = len(df)
    n_classes = 2
    class_weights = {
        0: total_samples / (n_classes * declined_count),
        1: total_samples / (n_classes * approved_count)
    }

    # Create feature matrix X and target vector y
    X = df.drop('Loan Status', axis=1)
    y = df['Loan Status']

    return X, y, numerical_cols, categorical_cols, class_weights, df

def create_voting_classifier_with_poly(numerical_cols, categorical_cols, class_weights):
    """Create a voting classifier with polynomial features pipeline."""
    # Create preprocessing pipeline with polynomial features
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_cols),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_cols)
        ])
    
    # Create individual models
    dt = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=10,
        class_weight=class_weights,
        random_state=42
    )
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        min_samples_split=10,
        min_samples_leaf=10,
        max_features='sqrt',
        class_weight=class_weights,
        n_jobs=-1,
        random_state=42
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=10,
        subsample=0.8,
        max_features='sqrt',
        random_state=42
    )
    
    # Create voting classifier
    voting_clf = VotingClassifier(
        estimators=[
            ('dt', dt),
            ('rf', rf),
            ('gb', gb)
        ],
        voting='soft'
    )
    
    # Create final pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', voting_clf)
    ])
    
    return pipeline

def evaluate_model(model, X_test, y_test, model_name="Voting Classifier with Polynomial Features"):
    """Evaluate the model and print metrics."""
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Calculate precision-recall curve and AUC
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    print(f"\n{model_name} Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Precision-Recall AUC: {pr_auc:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Declined', 'Approved']))
    
    return {
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'accuracy': accuracy,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'precision': precision,
        'recall': recall
    }

def plot_confusion_matrix(y_test, y_pred, output_path='confusion_matrix.png'):
    """Plot and save the confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Declined', 'Approved'],
                yticklabels=['Declined', 'Approved'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_model(model, filepath='loan_approval_model.joblib'):
    """Save the trained model to a file."""
    joblib.dump(model, filepath)
    print(f"\nTrained model saved to '{filepath}'")

def predict_loan_approval(model, loan_data):
    """Make predictions on new loan applications."""
    # Make predictions
    predictions = model.predict(loan_data)
    probabilities = model.predict_proba(loan_data)[:, 1]
    
    # Add predictions to the dataframe
    results = loan_data.copy()
    results['Predicted_Status'] = predictions
    results['Predicted_Status_Label'] = results['Predicted_Status'].apply(
        lambda x: 'Approved' if x == 1 else 'Declined')
    results['Approval_Probability'] = probabilities
    
    return results

def main():
    """Main function to train and evaluate the voting classifier with polynomial features."""
    
    # Load and preprocess data
    X, y, numerical_cols, categorical_cols, class_weights, df = load_and_preprocess_data("LoanApprovalData.csv")
    
    # Create train/test splits with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Create and train the voting classifier with polynomial features
    model = create_voting_classifier_with_poly(numerical_cols, categorical_cols, class_weights)
    print("\nTraining voting classifier with polynomial features...")
    model.fit(X_train, y_train)
    
    # Evaluate using cross-validation
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
    print(f"Cross-validation F1 score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Evaluate on test set
    test_results = evaluate_model(model, X_test, y_test)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, test_results['y_pred'], 
                         output_path="confusion_matrix.png")
    
    # Save the model
    model_path = "loan_approval_model.joblib"
    save_model(model, filepath=model_path)
    
    # Example prediction
    print("\nExample predictions on first 5 test samples:")
    sample_predictions = predict_loan_approval(model, X_test.iloc[:5])
    prediction_cols = ['Predicted_Status_Label', 'Approval_Probability']
    available_cols = [col for col in X_test.columns if col in sample_predictions.columns]
    display_cols = available_cols[:2] + prediction_cols
    print(sample_predictions[display_cols])
    
    print(f"\nProcess complete. Model saved to {model_path}")
    return model

if __name__ == "__main__":
    main()