# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, make_scorer, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import os
import traceback # For printing detailed error information

# =============================================================================
# --- 1. Setup Working Directory and Output Directory ---
# =============================================================================
base_work_dir = '/project_directory'
output_dir = os.path.join(base_work_dir, 'rf_classification_results')
os.makedirs(output_dir, exist_ok=True)
print(f"All outputs will be saved to: {output_dir}")

# Attempt to set the working directory
try:
    os.chdir(base_work_dir)
    print(f"Current working directory: {os.getcwd()}")
except FileNotFoundError:
    print(f"ERROR: Base working directory '{base_work_dir}' not found. Please check the path.")
    exit()
except Exception as e:
    print(f"Error changing working directory: {e}")
    exit() # Exit if directory change fails

# --- Set plotting font (optional, for cross-system compatibility) ---
try:
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    print("Plotting font set to Arial.")
except Exception as e:
    print(f"Warning: Failed to set font. Plots may render with default font: {e}")


# =============================================================================
# --- 2. Load Training/CV Data (The "Training Set") ---
# =============================================================================
print("\n--- Loading Training/CV Data ('train_otu_table.txt') ---")
try:
    train_otu = pd.read_csv('train_otu_table.txt', sep='\t', index_col=0, encoding='utf-8')
    print("Training OTU file loaded successfully (UTF-8 encoding).")
except UnicodeDecodeError:
    try:
        train_otu = pd.read_csv('train_otu_table.txt', sep='\t', index_col=0, encoding='latin1')
        print("Training OTU file loaded successfully (latin1 encoding).")
    except Exception as e:
        print(f"ERROR: Failed to read training OTU file: {e}")
        exit()
except FileNotFoundError:
    print(f"ERROR: 'train_otu_table.txt' not found in '{base_work_dir}'.")
    exit()
except Exception as e:
    print(f"ERROR: An unknown error occurred while reading training OTU file: {e}")
    exit()

try:
    train_meta = pd.read_csv('train_metadata.txt', sep='\t')
    print("Training Metadata file loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: 'train_metadata.txt' not found in '{base_work_dir}'.")
    exit()
except Exception as e:
     print(f"ERROR: Error reading training Metadata file: {e}")
     exit()

# =============================================================================
# --- 3. Preprocess Training/CV Data ---
# =============================================================================
print("\n--- Preprocessing Training/CV Data ---")
train_meta_clean = train_meta.copy()
# Ensure column names are 'Sample' and 'Group'
if train_meta_clean.shape[1] == 2:
    train_meta_clean.columns = ['Sample', 'Group']
elif 'Sample' not in train_meta_clean.columns or 'Group' not in train_meta_clean.columns:
    print("ERROR: Training Metadata file is missing 'Sample' or 'Group' columns.")
    exit()
train_meta_clean.set_index('Sample', inplace=True)

# Transpose OTU table
train_otu_t = train_otu.T

# Align samples (critical step)
common_samples_train = list(set(train_otu_t.index) & set(train_meta_clean.index))
if not common_samples_train:
    print("ERROR: No common sample IDs found between Training/CV OTU data and Metadata.")
    exit()
print(f"Found {len(common_samples_train)} common samples for model training and cross-validation.")

# Filter data
X_train_full = train_otu_t.loc[common_samples_train] # Features (Train/CV set)
y_train_labels = train_meta_clean.loc[common_samples_train, 'Group'] # Original labels (Train/CV set)

# Label encoding (Train/CV set)
le_train = LabelEncoder() # Training set encoder
y_train_encoded = le_train.fit_transform(y_train_labels)
print(f"Training/CV labels encoded. Class mapping: {dict(zip(le_train.classes_, le_train.transform(le_train.classes_)))}")

# Determine positive class label and index (Train/CV set)
positive_class_label = 'DISEASE_LABEL' # <--- IMPORTANT: Confirm your positive class name!
try:
    positive_class_index = int(le_train.transform([positive_class_label])[0])
    print(f"Confirmed: Positive class '{positive_class_label}' in Train/CV set is encoded as index {positive_class_index}")
except ValueError:
    print(f"WARNING: Target class '{positive_class_label}' not found in Train/CV labels: {le_train.classes_}.")
    if len(le_train.classes_) == 2:
        # Automatically select index 1 as the positive class if binary
        positive_class_index = 1 
        positive_class_label = le_train.classes_[positive_class_index]
        print(f"WARNING: Auto-selecting index 1 ('{positive_class_label}') as the positive class for Train/CV set.")
    else:
        print(f"ERROR: Cannot auto-determine positive class. Please ensure '{positive_class_label}' exists in the 'Group' column.")
        exit()

# =============================================================================
# --- 4. Model Training and Cross-Validation (Hyperparameter Tuning) ---
# =============================================================================
print("\n--- Starting Model Training and Hyperparameter Grid Search (5-Fold CV) ---")

# --- Define a specific AUC scorer ---
# This ensures GridSearchCV uses the positive class index we identified in Section 3
def custom_auc_scorer(y_true, y_pred_proba):
    # y_pred_proba is (n_samples, n_classes)
    # We need the probability of the positive class
    pos_class_proba = y_pred_proba[:, positive_class_index]
    # y_true is already encoded (0, 1, etc.)
    # We need to binarize y_true against the positive class
    y_true_binary = (y_true == positive_class_index).astype(int)
    return roc_auc_score(y_true_binary, pos_class_proba)

# Create the scorer, telling it we need probabilities
explicit_auc_scorer = make_scorer(custom_auc_scorer, needs_proba=True)
print(f"Custom scorer created. Will use index {positive_class_index} ('{positive_class_label}') to calculate AUC.")
# --- End of custom scorer ---

param_grid = {
    'n_estimators': [150], # Number of trees
    'max_depth': [3, 5, 6],   # Max depth of trees
    'random_state': [42],       # Seed for reproducibility
    'class_weight': [ 'balanced', {0: 1, 1: 2}, {0: 1, 1: 3}], # Class weighting strategies
    'min_samples_split': [ 2, 3, 5], # Min samples to split a node
    'min_samples_leaf': [2, 5, 8]     # Min samples at a leaf node
}

rf = RandomForestClassifier()
# Inner cross-validation for GridSearchCV
cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(estimator=rf,
                           param_grid=param_grid,
                           cv=cv_inner,
                           # Use our custom scorer
                           scoring=explicit_auc_scorer,
                           n_jobs=-1,            # Use all CPU cores
                           verbose=1)            # Print search progress

grid_search.fit(X_train_full, y_train_encoded) # Search on the full Train/CV set

# Output best parameters
print(f"\nGrid search complete.")
print(f"Best hyperparameters found: {grid_search.best_params_}")
print(f"Corresponding 5-fold CV mean ROC AUC (GridSearch estimate): {grid_search.best_score_:.4f}")

# Get the final model, refit on the *entire* X_train_full with the best params
best_rf_model = grid_search.best_estimator_


# =============================================================================
# --- 5. Plot 5-Fold Cross-Validation ROC Curve (Visualize Stability) ---
# =============================================================================
print("\n--- Calculating and plotting 5-Fold CV ROC Curve (for visualization) ---")
# Outer cross-validation for plotting and stability assessment
cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 

tprs_cv = []
aucs_cv = []
mean_fpr_cv = np.linspace(0, 1, 100) # Define a common FPR axis

X_for_cv = X_train_full.values # Use NumPy arrays
y_for_cv = y_train_encoded

plt.figure(figsize=(10, 8))

print("Calculating ROC for each fold...")
for i, (train_idx, test_idx) in enumerate(cv_outer.split(X_for_cv, y_for_cv)):
    # Create a model instance with the best params found by GridSearchCV
    model_cv = RandomForestClassifier(**grid_search.best_params_)
    model_cv.fit(X_for_cv[train_idx], y_for_cv[train_idx])

    # Predict probabilities for the positive class
    probas_ = model_cv.predict_proba(X_for_cv[test_idx])[:, positive_class_index]

    # Binarize true labels for this fold
    y_true_binary_fold = (y_for_cv[test_idx] == positive_class_index).astype(int)
    fpr, tpr, thresholds = roc_curve(y_true_binary_fold, probas_) 
    
    tprs_cv.append(np.interp(mean_fpr_cv, fpr, tpr)) # Interpolate
    tprs_cv[-1][0] = 0.0 # Ensure start at 0
    roc_auc = auc(fpr, tpr)
    aucs_cv.append(roc_auc)
    print(f"   Fold {i+1}: AUC = {roc_auc:.4f}")

# --- Calculate mean and standard deviation ---
mean_tpr_cv = np.mean(tprs_cv, axis=0)
mean_tpr_cv[-1] = 1.0 # Ensure end at 1
mean_auc_cv = auc(mean_fpr_cv, mean_tpr_cv)
std_auc_cv = np.std(aucs_cv)
std_tpr_cv = np.std(tprs_cv, axis=0)
tprs_upper_cv = np.minimum(mean_tpr_cv + std_tpr_cv, 1)
tprs_lower_cv = np.maximum(mean_tpr_cv - std_tpr_cv, 0)
print(f"Cross-validation mean AUC: {mean_auc_cv:.4f} +/- {std_auc_cv:.4f}")

# --- Plot the figure ---
# Mean ROC curve
plt.plot(mean_fpr_cv, mean_tpr_cv, color='blue',
         label=rf'Mean ROC (AUC = {mean_auc_cv:.3f} $\pm$ {std_auc_cv:.3f})',
         lw=2.5, alpha=0.9)

# Standard deviation confidence interval
plt.fill_between(mean_fpr_cv, tprs_lower_cv, tprs_upper_cv, color='grey', alpha=0.2,
                 label=r'$\pm$ 1 std. dev.')

# Chance line
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='red',
         label='Chance', alpha=0.8)

# Set plot properties
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14)
plt.title('5-Fold Cross-Validation ROC Curve (Random Forest)', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, alpha=0.3)

# Save the figure
cv_roc_filename = os.path.join(output_dir, 'CrossValidation_Mean_ROC_Curve.png')
try:
    plt.savefig(cv_roc_filename, dpi=300, bbox_inches='tight')
    print(f"Cross-validation ROC curve plot saved to: {cv_roc_filename}")
except Exception as e:
    print(f"ERROR: Could not save CV ROC curve plot: {e}")
plt.show() 
plt.close() 


# =============================================================================
# --- 6. Evaluate Final Model on Full "Train/CV" Set (Check for Overfitting) ---
# =============================================================================
print("\n--- Evaluating final model on the full 'Train/CV' set (to check for overfitting) ---")
X_test_eval = X_train_full 
y_test_eval_encoded = y_train_encoded 

y_test_proba_eval = best_rf_model.predict_proba(X_test_eval)[:, positive_class_index]

# --- 6.1 ROC Curve (Train/CV set) ---
y_true_binary_train = (y_test_eval_encoded == positive_class_index).astype(int)
fpr_test_eval, tpr_test_eval, _ = roc_curve(y_true_binary_train, y_test_proba_eval) 
roc_auc_test_eval = auc(fpr_test_eval, tpr_test_eval)
print(f"'Train/CV' set (final model) ROC AUC: {roc_auc_test_eval:.4f}")

plt.figure(figsize=(10, 8))
plt.plot(fpr_test_eval, tpr_test_eval, lw=2.5, color='darkorange', label=f'\'Train/CV\' Set ROC (AUC = {roc_auc_test_eval:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curve on \'Train/CV\' Set (Final Model)', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, alpha=0.3)
test_roc_filename = os.path.join(output_dir, 'TrainSet_ROC_Curve.png') 
try:
    plt.savefig(test_roc_filename, dpi=300, bbox_inches='tight')
    print(f"'Train/CV' set ROC curve saved to: {test_roc_filename}")
except Exception as e:
    print(f"ERROR: Could not save 'Train/CV' set ROC curve plot: {e}")
plt.show()
plt.close()

# =============================================================================
# --- 7. Feature Importance Analysis ---
# =============================================================================
print("\n--- Performing Feature Importance Analysis ---")
try:
    feature_names = X_train_full.columns
    importances = best_rf_model.feature_importances_
    indices = np.argsort(importances)[::-1] 

    feature_importances_df = pd.DataFrame({'OTU': feature_names[indices],
                                          'Importance': importances[indices]})

    # Plot Top 20 features
    plt.figure(figsize=(10, 8)) 
    sns.barplot(x='Importance', y='OTU', data=feature_importances_df.head(20), palette='viridis')
    plt.title('Top 20 Important OTUs (Final Random Forest Model)', fontsize=16)
    plt.xlabel('Feature Importance Score', fontsize=14)
    plt.ylabel('OTU ID', fontsize=14)
    plt.tight_layout() 

    feature_imp_filename = os.path.join(output_dir, 'Feature_Importance_OTU_Level.png')
    try:
        plt.savefig(feature_imp_filename, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to: {feature_imp_filename}")
    except Exception as e:
        print(f"ERROR: Could not save feature importance plot: {e}")
    plt.show()
    plt.close()

    # Save all feature importances to CSV
    feature_imp_csv_filename = os.path.join(output_dir, 'RF_Feature_Importance_OTU_Level.csv')
    try:
        feature_importances_df.to_csv(feature_imp_csv_filename, index=False, encoding='utf-8')
        print(f"Feature importance data saved to: {feature_imp_csv_filename}")
    except Exception as e:
        print(f"ERROR: Could not save feature importance CSV file: {e}")

except Exception as e:
    print(f"Error during feature importance analysis: {e}")
    traceback.print_exc()


# =============================================================================
# --- 8. External Independent Validation Set Evaluation (The "Test Set") ---
# =============================================================================
print("\n" + "="*80)
print("--- Starting Evaluation on External Independent Validation Set (the 'Test Set') ---")
print("="*80 + "\n")

# --- 8.1 Load Validation Data ---
print("Loading validation set data...")
try:
    validation_otu = pd.read_csv('validation_otu_table.txt', sep='\t', index_col=0, encoding='utf-8')
    validation_meta = pd.read_csv('validation_metadata.txt', sep='\t')
    print("Validation set files loaded successfully (UTF-8 encoding)")
except UnicodeDecodeError:
    try:
        print("Attempting to load validation set files with latin1 encoding...")
        validation_otu = pd.read_csv('validation_otu_table.txt', sep='\t', index_col=0, encoding='latin1')
        validation_meta = pd.read_csv('validation_metadata.txt', sep='\t', encoding='latin1')
        print("Validation set files loaded successfully (latin1 encoding)")
    except Exception as e:
        print(f"ERROR: Failed to read validation set files (latin1): {e}")
        exit()
except FileNotFoundError:
    print(f"ERROR: Validation files ('validation_otu_table.txt' or 'validation_metadata.txt') not found in '{base_work_dir}'.")
    exit()
except Exception as e:
    print(f"ERROR: An unknown error occurred while reading validation set files: {e}")
    exit()

# --- 8.2 Preprocess Validation Data ---
print("Preprocessing validation set data...")
validation_meta_clean = validation_meta.copy()
if validation_meta_clean.shape[1] == 2:
    validation_meta_clean.columns = ['Sample', 'Group']
elif 'Sample' not in validation_meta_clean.columns or 'Group' not in validation_meta_clean.columns:
    print("ERROR: Validation Metadata file is missing 'Sample' or 'Group' columns.")
    exit()
validation_meta_clean.set_index('Sample', inplace=True)

validation_otu_t = validation_otu.T

common_samples_validation = list(set(validation_otu_t.index) & set(validation_meta_clean.index))
if not common_samples_validation:
    print("ERROR: No common sample IDs found between Validation OTU data and Metadata.")
    exit()
print(f"Found {len(common_samples_validation)} common samples for validation.")

X_validation_raw = validation_otu_t.loc[common_samples_validation] 
y_validation_labels = validation_meta_clean.loc[common_samples_validation, 'Group'] 

# --- Feature Alignment (Crucial!) ---
print("Aligning validation set features to training set features...")
train_features = X_train_full.columns # Get feature names from training
# Align validation set columns to training set columns
X_validation, _ = X_validation_raw.align(pd.DataFrame(columns=train_features), join='right', axis=1, fill_value=0)

missing_cols = set(train_features) - set(X_validation_raw.columns)
if missing_cols:
    print(f"   Note: {len(missing_cols)} features from training set were missing in validation set. Filled with 0.")
extra_cols = set(X_validation_raw.columns) - set(train_features)
if extra_cols:
    print(f"   Note: {len(extra_cols)} features from validation set were not in training set. Removed.")
print(f"   Validation set feature dimension aligned: {X_validation.shape}")

# --- Encode Validation Labels ---
# Use the *same* encoder (le_train) that was fit on the training data
try:
    y_validation_encoded = le_train.transform(y_validation_labels)
    print(f"Validation set labels encoded using training set's mapping.")
    le_val = le_train 
    try:
        gdm_class_index_val = int(le_train.transform([positive_class_label])[0]) 
        print(f"Confirmed: Validation target class '{positive_class_label}' is encoded as index {gdm_class_index_val}")
    except ValueError:
         print(f"ERROR: The positive class '{positive_class_label}' from training was not found in the validation set labels. This should not happen!")
         exit()
except ValueError as e:
     print(f"ERROR: Validation label encoding failed. Validation set may contain new labels not present in training data: {e}")
     print("Cannot proceed with validation. Please check 'Group' column in validation metadata.")
     exit()
except Exception as e:
    print(f"ERROR: An unknown error occurred during validation label encoding: {e}")
    exit()

# =============================================================================
# --- 8.5. Standard Validation Set Evaluation---
# =============================================================================
print("\n--- Starting standard validation set evaluation (ROC Curve only) ---")

try:
    # Use the best model to predict probabilities on the *aligned* validation set
    y_validation_proba_standard = best_rf_model.predict_proba(X_validation)[:, gdm_class_index_val] 
except Exception as e:
    print(f"ERROR: Failed to predict probabilities on validation set: {e}")
    if hasattr(best_rf_model, 'n_features_in_') and X_validation.shape[1] != best_rf_model.n_features_in_:
         print(f"Model expected {best_rf_model.n_features_in_} features, but validation set provided {X_validation.shape[1]}.")
    exit()

# --- 8.5.1 Standard ROC Curve (Validation Set) ---
y_true_binary_val = (y_validation_encoded == gdm_class_index_val).astype(int)
fpr_validation, tpr_validation, _ = roc_curve(y_true_binary_val, y_validation_proba_standard) 
roc_auc_validation = auc(fpr_validation, tpr_validation)
print(f"Standard Validation Set ROC AUC (from predict_proba): {roc_auc_validation:.4f}")

plt.figure(figsize=(10, 8))
plt.plot(fpr_validation, tpr_validation, lw=2.5, color='green', 
         label=f'Validation Set ROC (Standard) (AUC = {roc_auc_validation:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curve on Independent Validation Set (Standard predict_proba)', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, alpha=0.3)
val_roc_filename = os.path.join(output_dir, 'ValidationSet_ROC_Curve_Standard.png') 
try:
    plt.savefig(val_roc_filename, dpi=300, bbox_inches='tight')
    print(f"Validation set standard ROC curve saved to: {val_roc_filename}")
except Exception as e:
    print(f"ERROR: Could not save validation set standard ROC curve plot: {e}")
plt.show()
plt.close()

print("\n" + "="*80)
print("--- External Independent Validation Set evaluation complete ---")
print("="*80 + "\n")


# =============================================================================
# --- 9. Plot Combined ROC Curves (Standard Method) ---
# =============================================================================
print("\n--- Plotting Combined Standard ROC Curves (Train/CV, CV Mean, Validation) ---")

plt.figure(figsize=(12, 10)) 

# 1. Plot 'Train/CV' Set ROC
if 'fpr_test_eval' in locals() and 'tpr_test_eval' in locals() and 'roc_auc_test_eval' in locals():
    plt.plot(fpr_test_eval, tpr_test_eval, lw=2.5, alpha=0.8, color='darkorange',
             label=f'\'Train/CV\' Set (AUC = {roc_auc_test_eval:.3f})') 
else:
    print("Warning: 'Train/CV' set ROC data not available for combined plot.")


# 2. Plot 5-Fold CV Mean ROC
if 'mean_fpr_cv' in locals() and 'mean_tpr_cv' in locals() and 'mean_auc_cv' in locals() and 'std_auc_cv' in locals():
    plt.plot(mean_fpr_cv, mean_tpr_cv, lw=2.5, alpha=0.8, color='blue',
             label=rf'5-Fold CV Mean (AUC = {mean_auc_cv:.3f} $\pm$ {std_auc_cv:.3f})')
    if 'tprs_lower_cv' in locals() and 'tprs_upper_cv' in locals():
        plt.fill_between(mean_fpr_cv, tprs_lower_cv, tprs_upper_cv, color='grey', alpha=0.15)
else:
    print("Warning: CV mean ROC data not available for combined plot.")


# 3. Plot Independent Validation Set ROC
if 'fpr_validation' in locals() and 'tpr_validation' in locals() and 'roc_auc_validation' in locals():
    plt.plot(fpr_validation, tpr_validation, lw=2.5, alpha=0.8, color='green',
             label=f'Validation Set (Standard) (AUC = {roc_auc_validation:.3f})')
else:
    print("Warning: Standard validation set ROC data not available for combined plot.")

# 4. Plot Chance Line
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', label='Chance', alpha=0.5)

# Set plot properties
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14)
plt.title('Combined ROC Curves (Random Forest Evaluation)', fontsize=16)
plt.legend(loc="lower right", fontsize=11) 
plt.grid(True, alpha=0.3)

# Save combined plot
combined_roc_filename = os.path.join(output_dir, 'Combined_ROC_Curves_Evaluation.png')
try:
    plt.savefig(combined_roc_filename, dpi=300, bbox_inches='tight')
    print(f"Combined ROC curve plot saved to: {combined_roc_filename}")
except Exception as e:
    print(f"ERROR: Could not save combined ROC curve plot: {e}")

plt.show() # Display the plot
plt.close() # Close the plot

print("\n================= All processes completed ==================")