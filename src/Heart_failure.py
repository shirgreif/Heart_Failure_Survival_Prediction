# ------- Date structure
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# ------- Data preperation
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
# -------  Evaluation merics
from sklearn import metrics
from sklearn.metrics import fbeta_score # f2
from sklearn.metrics import matthews_corrcoef # mcc
from sklearn.metrics import roc_auc_score # auc roc
from sklearn.metrics import confusion_matrix
# -------  Models
from mlxtend.feature_selection import SequentialFeatureSelector # feature selection method

# Read Data From csv file
df = pd.read_csv('../data/heart_failure_clinical_records_dataset.csv')

## first view on the dataset
print("Features: "+ str(df.shape[1]-1)+ ", Observations: " + str(df.shape[0]))
df.info()

# Null values - There are none
df.isnull().sum()

# Duplicates - there are not
df.duplicated().sum()

# Seperating the dataset according to patients with and without death event
death_events = pd.DataFrame(df[df.DEATH_EVENT ==1])
non_death_events = pd.DataFrame(df[df.DEATH_EVENT ==0])

# Target proportion

y_count = sns.countplot(x="DEATH_EVENT", data=df, palette="Reds", order=df["DEATH_EVENT"].value_counts(ascending=True).index)
y_count.set_xlabel("Death event", fontweight='bold')
y_count.set_ylabel("Count", fontweight='bold')
y_count.set_xticklabels(["Yes (1)", "No (0)"])
rel_values = df['DEATH_EVENT'].value_counts(ascending=True, normalize=True).values * 100
orderLabel = df['DEATH_EVENT'].value_counts(ascending=True).values
lbl = [f'{p[0]} ({p[1]:.0f}%)' for p in zip(orderLabel, rel_values)]
# Add labels to the bars
for i, v in enumerate(orderLabel):
    y_count.text(i, v + 0.2, lbl[i], ha='center', va='bottom')
y_count.set_title('Death events of heart failure patients', fontweight='bold')
plt.show()

# Discreptive data
summary_stats = df.describe().T
df_summary = pd.DataFrame(summary_stats)
sns.heatmap(df_summary, annot=True, cmap='coolwarm', fmt=".2f", cbar=False)
plt.title('Summary Statistics')
plt.show()

# nice display for the summary
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)
print(df_summary)

# Dividing features into Numerical and Categorical
categorical_features = []
numerical_features = []
for i in df.columns:
    if len(df[i].unique()) <= 2:
        categorical_features.append(i)
    else:
        numerical_features.append(i)

# EDA categorical

# Anemia
anaemia_count = sns.countplot(data=df, x="anaemia", hue="DEATH_EVENT", palette="Reds", order=df["anaemia"].value_counts(ascending=True).index)
anaemia_count.set_title("Anemia patients distributed by death events", fontweight='bold')
anaemia_count.set_xlabel("Anemia", fontweight='bold')
anaemia_count.set_ylabel("Count", fontweight='bold')
anaemia_count.set_xticklabels(["Yes (1)", "No (0)"])
# Add labels showing count and percentage for each category and hue
for p in anaemia_count.containers:
    anaemia_count.bar_label(p, label_type='edge', fmt='%.0f')
plt.show()

# Diabetes
diabetes_count = sns.countplot(data=df, x="diabetes", hue="DEATH_EVENT", palette="Reds", order=df["diabetes"].value_counts(ascending=True).index)
diabetes_count.set_title("Diabetes distributed by death events", fontweight='bold')
diabetes_count.set_xlabel("Diabetes", fontweight='bold')
diabetes_count.set_ylabel("Count", fontweight='bold')
diabetes_count.set_xticklabels(["Yes (1)", "No (0)"])
# Add labels showing count and percentage for each category and hue
for p in diabetes_count.containers:
    diabetes_count.bar_label(p, label_type='edge', fmt='%.0f')
plt.show()

# high_blood_pressure
high_blood_pressure_count = sns.countplot(data=df, x="high_blood_pressure", hue="DEATH_EVENT", palette="Reds", order=df["high_blood_pressure"].value_counts(ascending=True).index)
high_blood_pressure_count.set_title("High blood pressure distributed by death events", fontweight='bold')
high_blood_pressure_count.set_xlabel("High blood pressure", fontweight='bold')
high_blood_pressure_count.set_ylabel("Count", fontweight='bold')
high_blood_pressure_count.set_xticklabels(["Yes (1)", "No (0)"])
# Add labels showing count and percentage for each category and hue
for p in high_blood_pressure_count.containers:
    high_blood_pressure_count.bar_label(p, label_type='edge', fmt='%.0f')
plt.show()

# sex
sex_count = sns.countplot(data=df, x="sex", hue="DEATH_EVENT", palette="Reds", order=df["sex"].value_counts(ascending=True).index)
sex_count.set_title("Sex distributed by death events", fontweight='bold')
sex_count.set_xlabel("Sex", fontweight='bold')
sex_count.set_ylabel("Count", fontweight='bold')
sex_count.set_xticklabels(["Woman (0)","Man (1)"])
# Add labels showing count and percentage for each category and hue
for p in sex_count.containers:
    sex_count.bar_label(p, label_type='edge', fmt='%.0f')
plt.show()

# smoking
smoking_count = sns.countplot(data=df, x="smoking", hue="DEATH_EVENT", palette="Reds", order=df["smoking"].value_counts(ascending=True).index)
smoking_count.set_title("Smoking distributed by death events", fontweight='bold')
smoking_count.set_xlabel("Smoking", fontweight='bold')
smoking_count.set_ylabel("Count", fontweight='bold')
smoking_count.set_xticklabels(["Yes (1)", "No (0)"])
# Add labels showing count and percentage for each category and hue
for p in smoking_count.containers:
    smoking_count.bar_label(p, label_type='edge', fmt='%.0f')
plt.show()

# Statistical quantitative description for category features

summary_data = []
for feature in categorical_features:
    if feature != "DEATH_EVENT":
        for value in sorted(df[feature].unique()):
            num_samples = df[df[feature] == value].shape[0]
            num_samples_death_event_1 = df[(df[feature] == value) & (df["DEATH_EVENT"] == 1)].shape[0]
            num_samples_death_event_0 = df[(df[feature] == value) & (df["DEATH_EVENT"] == 0)].shape[0]

            summary_data.append({
                "Feature": feature,
                "Value": value,
                "Number of Samples": num_samples,
                "Total (%)": num_samples / df.shape[0],
                "Death events": num_samples_death_event_1,
                "Death (%)": num_samples_death_event_1 / df.shape[0],
                "Without death event": num_samples_death_event_0,
                "Without death (%)": num_samples_death_event_0 / df.shape[0]
            })
df_summary_category = pd.DataFrame(summary_data)
print(df_summary_category)

# EDA numeric
sns.set_palette("Reds")
for column in numerical_features:
    if df[column].dtype != "object":
        plt.figure(figsize=(15, 6))
        plt.subplot(2, 2, 1)
        sns.histplot(data=df, x=column, kde=True, palette="Reds")
        plt.ylabel("Frequency")
        plt.xlabel(column)
        plt.title(f"Distribution of {column}")
        plt.subplot(2, 2, 2)
        sns.boxplot(data=df, x=column, palette="Reds")
        plt.title(f"Boxplot of {column}")
        plt.show()


###### Outliers

# Find outliers
def find_outliers(df,coef):
    outliers = {}
    outliers_by_class = {'DEATH_EVENT=0': {}, 'DEATH_EVENT=1': {}}

    for column in df.columns:
        if len(df[column].unique()) > 2:  # Check if the column is not binary
            outliers_count_0 = df[df['DEATH_EVENT'] == 0][column].apply(lambda x: (x < df[column].quantile(0.25) - coef * (df[column].quantile(0.75) - df[column].quantile(0.25))) or (x > df[column].quantile(0.75) + coef * (df[column].quantile(0.75) - df[column].quantile(0.25)))).sum()
            outliers_count_1 = df[df['DEATH_EVENT'] == 1][column].apply(lambda x: (x < df[column].quantile(0.25) - coef * (df[column].quantile(0.75) - df[column].quantile(0.25))) or (x > df[column].quantile(0.75) + coef * (df[column].quantile(0.75) - df[column].quantile(0.25)))).sum()

            outliers[column] = {'total': outliers_count_0 + outliers_count_1, 'DEATH_EVENT=0': outliers_count_0, 'DEATH_EVENT=1': outliers_count_1}
    return outliers
outliers_count = find_outliers(df,1.5)
for column, counts in outliers_count.items():
    print(f"{column}: {counts['total']} outliers")
    print(f"  DEATH_EVENT=0: {counts['DEATH_EVENT=0']}")
    print(f"  DEATH_EVENT=1: {counts['DEATH_EVENT=1']}")

outliers_count = find_outliers(df,3)
for column, counts in outliers_count.items():
    print(f"{column}: {counts['total']} outliers")
    print(f"  DEATH_EVENT=0: {counts['DEATH_EVENT=0']}")
    print(f"  DEATH_EVENT=1: {counts['DEATH_EVENT=1']}")

# Boxplot within features
plt.figure(figsize=(10, 6))
for feature in numerical_features:
    plt.figure(figsize=(5, 5))  # Create a new figure for each feature
    sns.boxplot(y=feature, data=df, color='lightcoral')
    # Calculate and plot upper and lower bounds
    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 3 * iqr
    lower_bound = q1 - 3 * iqr
    plt.axhline(upper_bound, color='red', linestyle='--', label='Upper Bound (3*IQR)')
    plt.axhline(lower_bound, color='salmon', linestyle='--', label='Lower Bound (3*IQR)')
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15))
    plt.title(f'Boxplot of {feature}')
    plt.show()

# Boxplot within classes of DEATH_EVENT
plt.figure(figsize=(10, 6))
for feature in numerical_features:
    plt.figure(figsize=(5, 5))  # Create a new figure for each feature
    sns.boxplot(x='DEATH_EVENT', y=feature, hue='DEATH_EVENT', data=df, palette={0: 'lightblue', 1: 'lightcoral'})
    plt.title(f'Boxplot of {feature} by DEATH_EVENT')
    plt.legend(title='DEATH_EVENT', loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()

# Box after removal of 3IQR
plt.figure(figsize=(10, 6))
for feature in numerical_features:
    plt.figure(figsize=(5, 5))  # Create a new figure for each feature
    # Filter data based on 3*IQR rule
    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr
    filtered_data = df[(df[feature] > lower_bound) & (df[feature] < upper_bound)]
    # Plot boxplot with updated outliers
    sns.boxplot(x='DEATH_EVENT', y=feature, hue='DEATH_EVENT', data=filtered_data,
                palette={0: 'lightblue', 1: 'lightcoral'})
    plt.title(f'Boxplot of {feature} by DEATH_EVENT')
    plt.legend(title='DEATH_EVENT', loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()

# outliers removal
def remove_outliers(df):
    for col in df.columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 3 * iqr
        lower_bound = q1 - 3 * iqr
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
    return df
df= remove_outliers(df)
df.shape
# Correlation matrixes

# Partial
import pingouin as pg
df.pcorr()

# Pearson

def pearson_corr(df):
    df_corr=df.corr()
    df_corr=round(df_corr,2)
    mask = np.triu(np.ones_like(df_corr, dtype=bool)) # Create a mask to hide the upper triangle
    # Plot the heatmap with the masked upper triangle
    plt.figure(figsize=(10, 5))
    plt.title("Pearson Correlation Matrix")
    sns.heatmap(df_corr, annot=True, cmap="Reds", mask=mask)
    plt.show()
pearson_corr(df)

## Define features and target
X = df.drop('DEATH_EVENT' , axis='columns')
Y = df['DEATH_EVENT']

## VIF score

# VIF dataframe
def vif_calc(vif_data):
    vif_data["feature"] = X.columns
    # calculating VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                       for i in range(len(X.columns))]
    return vif_data
vif_data = vif_calc(pd.DataFrame())
print(vif_data)

# Dropping 'serum_sodium'
df = df.drop(columns=['serum_sodium'], axis=1)
X = df.drop('DEATH_EVENT' , axis='columns')
Y = df['DEATH_EVENT']
vif_data = vif_calc(pd.DataFrame())
# Dropping 'age'
df = df.drop(columns=['age'], axis=1)
X = df.drop('DEATH_EVENT' , axis='columns')
Y = df['DEATH_EVENT']
vif_data = vif_calc(pd.DataFrame())

# Calculate Pearson correlation with the target variable

import matplotlib.colors as mcolors
corr = df.corrwith(df['DEATH_EVENT']).sort_values(ascending=False).to_frame()
corr.columns = ['DEATH_EVENT']
# Define a colormap with specified range and colors
cmap = mcolors.ListedColormap(['red', 'salmon','red'], 'custom_red')
# Create a normalization object for your condition
bounds = [-1, -0.2, 0.2, 1]
norm = mcolors.BoundaryNorm(bounds, cmap.N)
# Plot the correlation heatmap
plt.subplots(figsize=(5, 5))
sns.heatmap(corr, annot=True, cmap=cmap, norm=norm, linewidths=0.4, linecolor='black', cbar=False, fmt='.2f')
plt.title('Correlation with DEATH_EVENT')
plt.show()

# Seperating the data
X = df.drop('DEATH_EVENT' , axis='columns')
Y = df['DEATH_EVENT']

## Preprocessing

## Scaling - MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_rounded = np.round(X_scaled, decimals=3)
X_rounded_df = pd.DataFrame(X_rounded, columns=X.columns)

# Merge the DataFrames based on their new indexes
y_df = pd.DataFrame(Y)
y_df.reset_index(drop=True, inplace=True)
# Concatenate X_rounded_df and y_df
combined_df = pd.concat([X_rounded_df, y_df], axis=1)
# Save the combined DataFrame as a CSV file
combined_df.to_csv("~/Desktop/combined_data.csv", index=False)

# Splitting

x_train, x_test, y_train, y_test = train_test_split(X_rounded_df, Y, test_size=0.2, random_state=42, stratify=Y)

#  Models

## 1. Logistic Regression Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, fbeta_score, matthews_corrcoef
from sklearn.model_selection import GridSearchCV

# Defining and fitting the logistic regression model
lr_model = LogisticRegression(penalty='l2', C=15, max_iter=1000)
lr_model.fit(x_train, y_train)
lr_pred = lr_model.predict(x_test)
# Defining the parameter grid for GridSearchCV
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
# Performing grid search to find the best hyperparameters
best_lr = GridSearchCV(LogisticRegression(max_iter=1000), param_grid=param_grid)
best_lr.fit(x_train, y_train)
lr_pred = lr_model.predict(x_test)
params = lr_model.get_params()
print(params)
print(f"Best parameters: {best_lr.best_params_}")

# Define your logistic regression model

# Select the explanatory variables you want to include in the model
selected_features = ['time', 'serum_creatinine', 'ejection_fraction']
x_train_selected = x_train[selected_features]
glm_model = LogisticRegression()
# Fit the model on the training data with selected features
glm_model.fit(x_train_selected, y_train)

# Predict probabilities on the testing data with selected features
predicted_probs = glm_model.predict_proba(x_test[selected_features])[:, 1]
# Fit the model on the training data
glm_model.fit(x_train, y_train)

binary_predictions = (predicted_probs >= 0.6).astype(int) # Threshold is 0.6

# Calculate metrics on the testing data
auc_roc = roc_auc_score(y_test, predicted_probs)
f2_score = fbeta_score(y_test, binary_predictions, beta=2)
mcc = matthews_corrcoef(y_test, binary_predictions)
print("AUC-ROC:", auc_roc)
print("F2 Score:", f2_score)
print("MCC:", mcc)

# Calculate confusion matrix
cm_test = confusion_matrix(y_true=y_test, y_pred=binary_predictions)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Test Set')
plt.show()

# ----------------------------------------------------------------- Decision tree --------------------------------------------------------------------

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

# defining the default model
model_DT = DecisionTreeClassifier(random_state=42)
# training the default model with train data and matching labels.
model_DT.fit(x_train, y_train)

# full tree depth
max_depth_full_tree = model_DT.tree_.max_depth
print("Full depth of the decision tree: ", max_depth_full_tree)

# Plot the decision tree with feature names with depth
plt.figure(figsize=(45, 15))
plot_tree(model_DT, feature_names=x_train.columns, filled=True, rounded=True, class_names=['Death','Survival'], max_depth=3, fontsize=18)
plt.title('Decision Tree', fontweight='bold', fontsize=18)
plt.show()

# Probabilities of positive class (death event)
from sklearn.metrics import roc_auc_score
predicted_probabilities_train = model_DT.predict_proba(x_train)[:, 1]
auc_roc_score_train = roc_auc_score(y_train, predicted_probabilities_train)
print("AUC-ROC score for train set Decision Tree:", auc_roc_score_train)
predicted_probabilities = model_DT.predict_proba(x_test)[:, 1]
auc_roc_score = roc_auc_score(y_test, predicted_probabilities)
print("AUC-ROC Score for test set Decision Tree:", auc_roc_score)

#Check default hyperparameters
dt_default = DecisionTreeClassifier()
print("Decision Tree default hyperparameters:")
print(dt_default.get_params())

# Tuning tree hyperparameters - all parameters combine
## defining the grid that we want to tune the model according to.
param_grid = {
    'criterion': ['entropy', 'gini'],
    'max_depth': np.arange(1, max_depth_full_tree, 1),
    'max_features': np.arange(1, len(x_train.columns), 1),
    'min_samples_leaf': np.arange(1, 10, 1)
}
from sklearn.model_selection import GridSearchCV
# Grid search will perform a 10-fold CV (cv=10)
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42),
                           param_grid=param_grid,
                           refit=True,
                           cv=10, scoring='roc_auc', verbose=3)
grid_search.fit(x_train, y_train)

# Best parameters
print("Best hyperparameters:", grid_search.best_params_)
dt_model_after_tuning = grid_search.best_estimator_
dt_model_after_tuning.fit(x_train, y_train)

# Predict probabilities using the tuned model for train set
predicted_probabilities_train_tuned = dt_model_after_tuning.predict_proba(x_train)[:, 1]

# Calculate AUC-ROC score for the train set after tuning
auc_roc_score_train_tuned = roc_auc_score(y_train, predicted_probabilities_train_tuned)
print("AUC-ROC Score for train set Decision Tree after tuning:", auc_roc_score_train_tuned)

# Predict probabilities using the tuned model for test set
predicted_probabilities_test_tuned = dt_model_after_tuning.predict_proba(x_test)[:, 1]

# Calculate AUC-ROC score for the test set after tuning
auc_roc_score_test_tuned = roc_auc_score(y_test, predicted_probabilities_test_tuned)
print("AUC-ROC Score for test set Decision Tree after tuning:", auc_roc_score_test_tuned)

# ----------------------------------------------------------------- Random Forest --------------------------------------------------------------------

#---Default model----
RF_model = RandomForestClassifier(random_state=42)
RF_model.fit(x_train, y_train)

# Calculate AUC-ROC for the training set
train_pred_prob_rf_default = RF_model.predict_proba(x_train)[:, 1]
train_auc_roc = roc_auc_score(y_train, train_pred_prob_rf_default)
print("AUC-ROC Score for train set Random Forest:", train_auc_roc)

# Calculate AUC-ROC for the test set
test_pred_prob_rf_default = RF_model.predict_proba(x_test)[:, 1]
test_auc_roc = roc_auc_score(y_test, test_pred_prob_rf_default)
print("AUC-ROC Score for test set Random Forest:", test_auc_roc)

# Random Forest default hyperparameters
rf_default = RandomForestClassifier()
print("\nRandom Forest default hyperparameters:")
print(rf_default.get_params())

#---Tuned model----
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Create a random forest classifier
rf = RandomForestClassifier()

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 150, 200, 250, 300],
    'criterion': ['entropy', 'gini'],
    'max_depth': np.arange(1, 10, 1),
    'max_features': np.arange(1, len(x_train.columns), 1),
    'min_samples_leaf': np.arange(1, 10, 1)
}

# Create a 10-fold cross-validation object
kf = KFold(n_splits=2, shuffle=True, random_state=42)
# Instantiate the grid search with cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=kf, scoring='roc_auc', n_jobs=-1)
# Perform the grid search
grid_search.fit(x_train, y_train)
best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)
rf_model_after_tuning = grid_search.best_estimator_
rf_model_after_tuning.fit(x_train, y_train)

# Calculate AUC-ROC for the training set
train_pred_prob_rf = rf_model_after_tuning.predict_proba(x_train)[:, 1]
train_auc_roc = roc_auc_score(y_train, train_pred_prob_rf)
print("AUC-ROC Score for train set Random Forest after tuning:", train_auc_roc)

# Calculate AUC-ROC for the test set
test_pred_prob_rf = rf_model_after_tuning.predict_proba(x_test)[:, 1]
test_auc_roc = roc_auc_score(y_test, test_pred_prob_rf)
print("AUC-ROC Score for test set Random Forest after tuning:", test_auc_roc)

# Feature importance:

import matplotlib.pyplot as plt

# Get feature importances
feature_importances = rf.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(feature_importances)[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [x_train.columns[i] for i in indices]

# Create plot with y-axis in descending order
plt.figure(figsize=(12, 8))  # Adjust figure size as needed
plt.barh(range(x_train.shape[1]), feature_importances[indices], align='center')
plt.yticks(range(x_train.shape[1]), names)
plt.gca().invert_yaxis()  # Invert y-axis to show most important features at the top
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------Confusion matrix Decision Tree----------------------------------------------

# Get the predicted labels from the model
y_pred_train_dt = dt_model_after_tuning.predict(x_train)
y_pred_test_dt = dt_model_after_tuning.predict(x_test)

# Calculate the confusion matrices
cm_train_dt = confusion_matrix(y_true=y_train, y_pred=y_pred_train_dt)
cm_test_dt = confusion_matrix(y_true=y_test, y_pred=y_pred_test_dt)

# Create subplots for train set and test set confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot the train set confusion matrix
sns.heatmap(cm_train_dt, annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_xlabel('Predicted Labels')
axes[0].set_ylabel('True Labels')
axes[0].set_title('Confusion Matrix for Train Set')

# Plot the test set confusion matrix
sns.heatmap(cm_test_dt, annot=True, fmt="d", cmap="Blues", ax=axes[1])
axes[1].set_xlabel('Predicted Labels')
axes[1].set_ylabel('True Labels')
axes[1].set_title('Confusion Matrix for Test Set')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plots
plt.show()

#Confusion Matrix Random Forest

# Get the predicted labels from the model
y_pred_train = rf_model_after_tuning.predict(x_train)
y_pred_test = rf_model_after_tuning.predict(x_test)

# Calculate the confusion matrices
cm_train = confusion_matrix(y_true=y_train, y_pred=y_pred_train)
cm_test = confusion_matrix(y_true=y_test, y_pred=y_pred_test)

# Create subplots for train set and test set confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot the train set confusion matrix
sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_xlabel('Predicted Labels')
axes[0].set_ylabel('True Labels')
axes[0].set_title('Confusion Matrix for Train Set')

# Plot the test set confusion matrix
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", ax=axes[1])
axes[1].set_xlabel('Predicted Labels')
axes[1].set_ylabel('True Labels')
axes[1].set_title('Confusion Matrix for Test Set')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plots
plt.show()


#-----------------------------------------MCC and F2-score--------------------------------------
from sklearn.metrics import matthews_corrcoef, fbeta_score

# Calculate MCC and F2 score for the Decision Tree model after tuning
mcc_dt = matthews_corrcoef(y_test, dt_model_after_tuning.predict(x_test))
f2_dt = fbeta_score(y_test, dt_model_after_tuning.predict(x_test), beta=2)

print("MCC for Decision Tree after tuning:", mcc_dt)
print("F2 Score for Decision Tree after tuning:", f2_dt)

# Calculate MCC and F2 score for the Random Forest model after tuning
mcc_rf = matthews_corrcoef(y_test, rf_model_after_tuning.predict(x_test))
f2_rf = fbeta_score(y_test, rf_model_after_tuning.predict(x_test), beta=2)

print("MCC for Random Forest after tuning:", mcc_rf)
print("F2 Score for Random Forest after tuning:", f2_rf)

#-------------------------------------------------- Artificial Neural Networks (ANN) -----------------------------------------------------------------

## Basic ANN

# standardize the input with MinMax Scaler
from sklearn.preprocessing import MinMaxScaler
minmax_scaler = MinMaxScaler()
x_train_n = minmax_scaler.fit_transform(x_train)
x_test_n = minmax_scaler.fit_transform(x_test)
#x_test_n = minmax_scaler.transform(x_test)


# default parameters
model_MLP_n = MLPClassifier(random_state=42)
model_MLP_n.fit(x_train_n, y_train)

# Calculate AUC-ROC Score
predicted_probabilities_model_MLP_n_train = model_MLP_n.predict_proba(x_train_n)[:, 1]
auc_roc_score_model_MLP_n_train = roc_auc_score(y_train, predicted_probabilities_model_MLP_n_train)
print("AUC-ROC Score for MLP default on train set: ", auc_roc_score_model_MLP_n_train)

predicted_probabilities_model_MLP_n = model_MLP_n.predict_proba(x_test_n)[:, 1]
auc_roc_score_model_MLP_n = roc_auc_score(y_test, predicted_probabilities_model_MLP_n)
print("AUC-ROC Score for MLP default on test set: ", auc_roc_score_model_MLP_n)

# Calculate F2 Score
predicted_probabilities_model_MLP_n_train = model_MLP_n.predict_proba(x_train_n)[:, 1]
f2_score_model_MLP_n_train = fbeta_score(y_train, predicted_probabilities_model_MLP_n_train > 0.5, beta=2)
print("F2 Score for MLP default using MinMax Scaler on train set: ", f2_score_model_MLP_n_train)

predicted_probabilities_model_MLP_n = model_MLP_n.predict_proba(x_test_n)[:, 1]
f2_score_model_MLP_n = fbeta_score(y_test, predicted_probabilities_model_MLP_n > 0.5, beta=2)
print("F2 Score for MLP default using MinMax Scaler on test set: ", f2_score_model_MLP_n)

# Calculate Matthews Correlation Coefficient
predicted_probabilities_model_MLP_n_train = model_MLP_n.predict_proba(x_train_n)[:, 1]
mcc_score_model_MLP_n_train = matthews_corrcoef(y_train, predicted_probabilities_model_MLP_n_train > 0.5)
print("MCC Score for MLP default using MinMax Scaler on train set: ", mcc_score_model_MLP_n_train)

predicted_probabilities_model_MLP_n = model_MLP_n.predict_proba(x_test_n)[:, 1]
mcc_score_model_MLP_n = matthews_corrcoef(y_test, predicted_probabilities_model_MLP_n > 0.5)
print("MCC Score for MLP default using MinMax Scaler on test set: ", mcc_score_model_MLP_n)


# Print the model parameter values
params_default_MLP = model_MLP_n.get_params()
for param, value in params_default_MLP.items():
    print(param, ":", value)

# Get the characteristics of the model
input_neurons = model_MLP_n.n_features_in_
hidden_layers = model_MLP_n.n_layers_ - 2  # Subtract input and output layers
hidden_neurons = model_MLP_n.hidden_layer_sizes

# Print the characteristics
print("Number of neurons in the input layer:", input_neurons)
print("Number of hidden layers:", hidden_layers)
print("Number of hidden neurons in each layer:", hidden_neurons)

# confusion matrix for default model
print(confusion_matrix(y_true=y_train, y_pred=model_MLP_n.predict(x_train_n)))

# Get the predicted labels from the model
y_pred_train = model_MLP_n.predict(x_train_n)
y_pred_test = model_MLP_n.predict(x_test_n)

# Calculate the confusion matrix
cm_train = confusion_matrix(y_true=y_train, y_pred=y_pred_train)
cm_test = confusion_matrix(y_true=y_test, y_pred=y_pred_test)
# Create subplots for train set and test set confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# Plot the train set confusion matrix
sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_xlabel('Predicted Labels')
axes[0].set_ylabel('True Labels')
axes[0].set_title('Confusion Matrix for Train Set')
# Plot the test set confusion matrix
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", ax=axes[1])
axes[1].set_xlabel('Predicted Labels')
axes[1].set_ylabel('True Labels')
axes[1].set_title('Confusion Matrix for Test Set')
# Adjust the spacing between subplots
plt.tight_layout()
# Show the plots
plt.show()

## ANN - tuning hyper parameters of model

# hidden_layer_sizes #

## Test
X_train_s = pd.DataFrame(x_train_n)
Y_train_s = pd.DataFrame(y_train)

param_grid_hidden_layers = {
    'hidden_layer_sizes': [(i) for i in range(1, 50, 5)] + [(i, i) for i in range(1, 50, 5)] + [(i, i, i) for i in range(1, 50, 5)]
}
grid_search_hidden_layers = GridSearchCV(estimator=MLPClassifier(random_state=42),
                           param_grid= param_grid_hidden_layers,
                           refit=True,
                           cv=10, verbose=3, scoring='roc_auc', return_train_score=True)
# Fit the search according to grid values
grid_search_hidden_layers.fit(x_train_n, y_train)
# Get the results from the grid search
results_hidden_layers = grid_search_hidden_layers.cv_results_
# Extract relevant information from results
hidden_layers = results_hidden_layers['param_hidden_layer_sizes']
train_scores = results_hidden_layers['mean_train_score']
validation_scores = results_hidden_layers['mean_test_score']

# Sort the hidden layers and scores based on the validation score
sorted_indices = validation_scores.argsort()[::-1]
sorted_hidden_layers = [hidden_layers[i] for i in sorted_indices]
sorted_train_scores = train_scores[sorted_indices]
sorted_validation_scores = validation_scores[sorted_indices]

# Create and display the table
print("| Hidden layers  | Train Score  | Validation Score  |")
print("|----------------|--------------|-------------------|")
for hidden_layers_value, train_score, validation_score in zip(sorted_hidden_layers, sorted_train_scores, sorted_validation_scores):
    print(f"| {str(hidden_layers_value):<14} | {train_score:<12.6f} | {validation_score:<17.6f} |")

# activation function #

param_grid_activation = {
    'activation': ['logistic', 'relu', 'tanh', 'identity']
}
grid_search_activation = GridSearchCV(estimator=MLPClassifier(random_state=42),
                           param_grid=param_grid_activation,
                           refit=True,
                           cv=10, verbose=3, scoring='roc_auc', return_train_score=True)
# Fit the search according to grid values
grid_search_activation.fit(x_train_n, y_train)
# Get the results from the grid search
results_activation = grid_search_activation.cv_results_
# Extract relevant information from results
activations = results_activation['param_activation']
train_scores = results_activation['mean_train_score']
validation_scores = results_activation['mean_test_score']
# Sort the activations and scores based on the validation score
sorted_indices = validation_scores.argsort()[::-1]
sorted_activations = activations[sorted_indices]
sorted_train_scores = train_scores[sorted_indices]
sorted_validation_scores = validation_scores[sorted_indices]
# Create and display the table
print("| Activation | Train Score | Validation Score |")
print("|------------|-------------|------------------|")
for activation, train_score, validation_score in zip(sorted_activations, sorted_train_scores, sorted_validation_scores):
    print(f"| {activation:<10} | {train_score:<11.6f} | {validation_score:<15.6f} |")

# learning rate init
param_grid_learning_rate = {
    'learning_rate_init': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
}
grid_search_learning_rate = GridSearchCV(estimator=MLPClassifier(random_state=42),
                           param_grid=param_grid_learning_rate,
                           refit=True,
                           cv=10, verbose=3, scoring='roc_auc', return_train_score=True)
# Fit the search according to grid values
grid_search_learning_rate.fit(x_train_n, y_train)
# Get the results from the grid search
results_learning_rate = grid_search_learning_rate.cv_results_
# Extract relevant information from results
learning_rates = results_learning_rate['param_learning_rate_init']
train_scores = results_learning_rate['mean_train_score']
validation_scores = results_learning_rate['mean_test_score']
# Sort the learning rate and scores based on the validation score
sorted_indices = validation_scores.argsort()[::-1]
sorted_learning_rates = learning_rates[sorted_indices]
sorted_train_scores = train_scores[sorted_indices]
sorted_validation_scores = validation_scores[sorted_indices]
# Create and display the table
print("| Learning rate  | Train Score  | Validation Score |")
print("|----------------|--------------|------------------|")
for learning_rate, train_score, validation_score in zip(sorted_learning_rates, sorted_train_scores, sorted_validation_scores):
    print(f"| {learning_rate:<14} | {train_score:<12.6f} | {validation_score:<17.6f} |")

# Plot the learning rate vs. train score and validation score
learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
plt.figure(figsize=(10, 6))
plt.plot(range(len(learning_rates)), train_scores, marker='o', label='Train Score')
plt.plot(range(len(learning_rates)), validation_scores, marker='o', label='Validation Score')
plt.title('Learning Rate vs. Scores')
plt.xlabel('Learning Rate')
plt.ylabel('Score')
x_ticks = np.arange(len(learning_rates))
plt.xticks(x_ticks, learning_rates)  # Set x-axis tick locations and labels
plt.legend()
plt.grid(True)
plt.show()

# max iterations #

param_grid_max_iter = {
    'max_iter': [100, 200, 300, 400, 500]
}
grid_search_max_iter = GridSearchCV(estimator=MLPClassifier(random_state=42),
                           param_grid=param_grid_max_iter,
                           refit=True,
                           cv=10, verbose=3, scoring='roc_auc', return_train_score=True)
# Fit the search according to grid values
grid_search_max_iter.fit(x_train_n, y_train)
# Get the results from the grid search
results_max_iter = grid_search_max_iter.cv_results_
# Extract relevant information from results
max_iters = results_max_iter['param_max_iter']
train_scores = results_max_iter['mean_train_score']
validation_scores = results_max_iter['mean_test_score']
# Sort the activations and scores based on the validation score
sorted_indices = validation_scores.argsort()[::-1]
sorted_max_iters = max_iters[sorted_indices]
sorted_train_scores = train_scores[sorted_indices]
sorted_validation_scores = validation_scores[sorted_indices]
# Create and display the table
print("| Max iterations | Train Score | Validation Score |")
print("|----------------|-------------|------------------|")
for max_iters, train_score, validation_score in zip(sorted_max_iters, sorted_train_scores, sorted_validation_scores):
    print(f"| {max_iters:<14} | {train_score:<12.6f} | {validation_score:<17.6f} |")

# Plot the max iterations vs. train score and validation score
max_iter = [100, 200, 300, 400, 500]
plt.figure(figsize=(10, 6))
plt.plot(range(len(max_iter)), train_scores, marker='o', label='Train Score')
plt.plot(range(len(max_iter)), validation_scores, marker='o', label='Validation Score')
plt.title('Max iterations vs. Scores')
plt.xlabel('Max iterations')
plt.ylabel('Score')
x_ticks = np.arange(len(max_iter))
plt.xticks(x_ticks, max_iter)  # Set x-axis tick locations and labels
plt.legend()
plt.grid(True)
plt.show()

### Best ANN model

param_grid_bestMLP = {
    'hidden_layer_sizes':[(i) for i in range(1, 50, 5)] + [(i, i) for i in range(1, 50, 5)] + [(i, i, i) for i in range(1, 50, 5)],
    'activation': ['logistic', 'relu', 'tanh', 'identity'],
    'learning_rate_init': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
    'max_iter': [100, 200, 300, 400, 500]
}
grid_search_bestMLP = GridSearchCV(estimator=MLPClassifier(random_state=42),
                           param_grid=param_grid_bestMLP,
                           refit=True,
                           cv=10, verbose=3, scoring='roc_auc', return_train_score=True)
grid_search_bestMLP.fit(x_train_n, y_train)
# Get the results from the grid search
results_bestMLP = grid_search_bestMLP.cv_results_
# Best parameters
MLP_after_tuning = grid_search_bestMLP.best_estimator_
model_params = MLP_after_tuning.get_params()
for param, value in model_params.items():
    print(f"{param}: {value}")
MLP_after_tuning.fit(x_train_n, y_train)
# Extract relevant information from results
max_iters = results_bestMLP['param_max_iter']
train_scores = results_bestMLP['mean_train_score']
validation_scores = results_bestMLP['mean_test_score']
hidden_layer_sizes = results_bestMLP['param_hidden_layer_sizes']
activation = results_bestMLP['param_activation']
learning_rate_init = results_bestMLP['param_learning_rate_init']
grid_max_iters = results_bestMLP['param_max_iter']
grid_train_scores = results_bestMLP['mean_train_score']
grid_validation_scores = results_bestMLP['mean_test_score']

# Sort the scores based on the validation score in descending order
sorted_indices = validation_scores.argsort()[::-1]
sorted_max_iters = max_iters[sorted_indices]
sorted_train_scores = train_scores[sorted_indices]
sorted_validation_scores = validation_scores[sorted_indices]
sorted_hidden_layer_sizes = hidden_layer_sizes[sorted_indices]
sorted_activation = activation[sorted_indices]
sorted_learning_rate_init = learning_rate_init[sorted_indices]
sorted_grid_max_iters = grid_max_iters[sorted_indices]
sorted_grid_train_scores = grid_train_scores[sorted_indices]
sorted_grid_validation_scores = grid_validation_scores[sorted_indices]

# Saving the best Ann model after tuning
best_ann = grid_search_bestMLP.best_estimator_
best_params = grid_search_bestMLP.best_params_
best_params = {'activation': 'tanh', 'hidden_layer_sizes': (11, 11, 11), 'learning_rate_init': 0.001, 'max_iter': 200}
best_ann_tuned_model = MLPClassifier(activation='tanh', hidden_layer_sizes=(11, 11, 11),
              random_state=42)

# Create and display the table
print("| Max iterations | Train Score  | Validation Score  | Hidden Layer Sizes | Activation | Learning Rate  | Grid Max Iterations  | Grid Train Score | Grid Validation Score |")
print("|----------------|--------------|-------------------|--------------------|------------|----------------|----------------------|------------------|-----------------------|")
for max_iter, train_score, validation_score, hls, act, lr, grid_max_iter, grid_train_score, grid_validation_score in zip(
    sorted_max_iters, sorted_train_scores, sorted_validation_scores,
    sorted_hidden_layer_sizes, sorted_activation, sorted_learning_rate_init,
    sorted_grid_max_iters, sorted_grid_train_scores, sorted_grid_validation_scores
):
    hls_str = str(hls) if isinstance(hls, tuple) else hls
    act_str = str(act) if isinstance(act, tuple) else act
    lr_str = str(lr) if isinstance(lr, tuple) else lr
    grid_max_iter_str = str(grid_max_iter) if isinstance(grid_max_iter, tuple) else grid_max_iter

    print(
        "| {:<14} | {:<12.6f} | {:<17.6f} | {:<18} | {:<10} | {:<14} | {:<20} | {:<16.6f} | {:<21.6f} |".format(
            max_iter, train_score, validation_score,
            hls_str, act_str, lr_str,
            grid_max_iter_str, grid_train_score, grid_validation_score
        )
    )

# Create a dictionary with the data
data_best_MLP = {
    'Train Score': sorted_train_scores,
    'Validation Score': sorted_validation_scores,
    'Max iterations': sorted_max_iters,
    'Hidden Layer Sizes': sorted_hidden_layer_sizes,
    'Activation': sorted_activation,
    'Learning Rate': sorted_learning_rate_init,
}

# Create a DataFrame from the data
df_best_MLP = pd.DataFrame(data_best_MLP)
# Save the DataFrame to an Excel file
df_best_MLP.to_excel('results.xlsx', index=False)

# final best ANN model

# The best Ann model after tuning
best_params = {'activation': 'tanh', 'hidden_layer_sizes': (11, 11, 11), 'learning_rate_init': 0.001, 'max_iter': 200}
best_ann_tuned_model = MLPClassifier(activation='tanh', hidden_layer_sizes=(11, 11, 11), random_state=42)

best_ann_tuned_model.fit(x_train_n, y_train)

# AUC-ROC on the train set
auc_roc_score_best_MLP_s_train = roc_auc_score(y_train, best_ann_tuned_model.predict_proba(x_train_n)[:, 1])
print("AUC-ROC Score for MLP best model on train set: ", auc_roc_score_best_MLP_s_train)
# AUC-ROC on the test set
auc_roc_score_best_MLP_s_test = roc_auc_score(y_test, best_ann_tuned_model.predict_proba(x_test_n)[:, 1])
print("AUC-ROC Score for MLP best model on test set: ", auc_roc_score_best_MLP_s_test)
#roc_auc = roc_auc_score(y_test, best_ann_tuned_model.predict(x_test_n))

# F2 score on the train set
f2_score_best_MLP_s_train = fbeta_score(y_train, best_ann_tuned_model.predict(x_train_n), beta=2)
print("F2 Score for MLP best model on train set: ", f2_score_best_MLP_s_train)
# F2 Score on the test set
f2_score_best_MLP_s_test = fbeta_score(y_test, best_ann_tuned_model.predict(x_test_n), beta=2)
print("F2 Score for MLP best model on test set: ", f2_score_best_MLP_s_test)
#f2_score = fbeta_score(y_test, best_ann_tuned_model.predict(x_test_n))

# MCC on the train set
mcc_score_best_MLP_s_train = matthews_corrcoef(y_train, best_ann_tuned_model.predict(x_train_n))
print("MCC Score for MLP best model on train set: ", mcc_score_best_MLP_s_train)
# MCC on the test set
mcc_score_best_MLP_s_test = matthews_corrcoef(y_test, best_ann_tuned_model.predict(x_test_n))
print("MCC Score for MLP best model on test set: ", mcc_score_best_MLP_s_test)
#mcc = matthews_corrcoef(y_test, best_ann_tuned_model.predict(x_test_n))


# confusion matrix for tuned model
print(confusion_matrix(y_true=y_train, y_pred=best_ann_tuned_model.predict(x_train_n)))

# Get the predicted labels from the model
y_pred_train = best_ann_tuned_model.predict(x_train_n)
y_pred_test = best_ann_tuned_model.predict(x_test_n)
# Calculate the confusion matrixs
cm_train = confusion_matrix(y_true=y_train, y_pred=y_pred_train)
cm_test = confusion_matrix(y_true=y_test, y_pred=y_pred_test)
# Create subplots for train set and test set confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# Plot the train set confusion matrix with a slightly bigger font size
sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues", ax=axes[0], annot_kws={"fontsize": 16})
axes[0].set_xlabel('Predicted Labels')
axes[0].set_ylabel('True Labels')
axes[0].set_title('Confusion Matrix for Train Set')
# Plot the test set confusion matrix with a slightly bigger font size
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", ax=axes[1], annot_kws={"fontsize": 16})
axes[1].set_xlabel('Predicted Labels')
axes[1].set_ylabel('True Labels')
axes[1].set_title('Confusion Matrix for Test Set')
# Show the plot
plt.show()

