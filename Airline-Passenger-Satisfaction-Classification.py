from xml.etree.ElementTree import PI
import pandas as pd
import numpy as np
import missingno as msno # missing values plotcd
import seaborn as sns # missing plot
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split , cross_val_score # train-test split, cross validation
from sklearn.neighbors import KNeighborsClassifier # knn model
from sklearn.preprocessing import OneHotEncoder, StandardScaler # pre-processing
from sklearn.pipeline import Pipeline # pipeline is to assemble several steps that can be cross-validated together while setting different parameters.
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, jaccard_score


# airline training dataset
df = pd.DataFrame(pd.read_csv("./data/train.csv"))

# summary statistics
summary = df.describe()
print(summary)

print("dimensions:", df.shape)

# rename columns variables
df.columns = df.columns.str.replace(' ', '_')
df.columns = df.columns.str.replace('/', '_')

# drop variables 
df1 = df.drop(["id_1","id"], axis=1)

# plot missing NA values
sns.heatmap(df1.isna(), cbar=False)
#plt.show()
# check for NA values
na_count = df1.isna().sum()
print(na_count)
# percentage
na_count_ = df1.isna().sum() / df1.shape[0]
print(na_count_)



# one hot-encoder
encoder = OneHotEncoder(sparse_output=False)
df1_encod= pd.DataFrame(encoder.fit_transform(df1[['Gender', 'Customer_Type', 'Type_of_Travel','Class']]),
                         columns=encoder.get_feature_names_out(['Gender', 'Customer_Type', 'Type_of_Travel','Class'])).reset_index(drop=True)

df2 = df1.drop(['Gender', 'Customer_Type', 'Type_of_Travel','Class'], axis=1).reset_index(drop=True)

# combine dataset with one hot-encoder
df3 = pd.concat([df1_encod, df2], axis=1, ignore_index=True)

# combine the colnames of each dateset
cols = list(df1_encod.columns) + list(df2.columns)

# rename dataset columns
df3.columns = cols

# convert response to 1 or 0
def is_satisfied(satisfaction):
    if satisfaction == "satisfied":
        return 1
    else:
        return 0



df3['is_satisfied'] = df3['satisfaction'].apply(is_satisfied)
df4 = df3.drop(['satisfaction'], axis = 1)

# KNN model for NA value imputation
imputer = KNNImputer(n_neighbors = 5)
df4_imputed = pd.DataFrame(imputer.fit_transform(df4),columns=df4.columns)
#print(df.columns)

# check for NA values
na_count = df4_imputed.isna().sum()
print(na_count)

# convert data type
df4_imputed['is_satisfied'] = df4_imputed['is_satisfied'].astype(int)
print(df4_imputed['is_satisfied'].head)


# run KNN model 
# seperate predictors and response
x = df4_imputed.drop('is_satisfied', axis=1)
y = df4_imputed['is_satisfied']

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, balanced_accuracy_score

# decision tree, Boosting and Random Forest Model
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


# decision tree model 
tree_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('tree', DecisionTreeClassifier())
], verbose=True)

tree_param = {
    'tree__criterion': ['gini', 'entropy'],
    'tree__max_depth': [3, 5, 10, None],
    'tree__min_samples_split': [2, 5, 10],
    'tree__min_samples_leaf': [1, 2, 4]
}

# tune parameters, fit model on traning set for prediction 
tree_cv = GridSearchCV(tree_pipeline, tree_param, cv = 10, scoring = 'accuracy', n_jobs=-1)
tree_cv.fit(x, y)

# obtain the estimators
tree_best =  tree_cv.best_estimator_.named_steps['tree']

# obtain the predictors 
tree_importances = tree_best.feature_importances_
print(tree_importances)

# increase in gini index
variables_names = x.columns.str.replace("_", " ")

var_imp_tree = pd.DataFrame({'Predictors':variables_names, 'gini':tree_importances})
var_imp_tree_sorted = var_imp_tree.sort_values(by = 'gini', ascending=False)
print("Top 5 important predictors for bagging trees model")
print(var_imp_tree_sorted.head(5))

# chart of the important variables
importances = pd.Series(tree_importances, index=variables_names) # series - one dimension array
top_n = 10
top_n_idx = importances.argsort()[-top_n:] # know which features have the highest and lowest importance values
top_n_importances = importances.iloc[top_n_idx] # nteger-location
top_n_importances.plot(kind='barh', color='b')
plt.xlabel('Average Gini Decrease')
plt.title(f'Top {top_n} Important Features: Decision Tree Model')

var_imp_tree_sorted.head(10).sort_values(by = 'gini').plot(x = 'Predictors', y = 'gini', kind = 'barh', label = '')
plt.legend('')
plt.xlabel('Average Gini Decrease')
plt.title(f'Top {top_n} Important Features: Decision Tree Model')
plt.ylabel('')

# bagging trees model
bagging_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('bagging', BaggingClassifier(estimator=DecisionTreeClassifier()))
], verbose = True)


# parameters bagging
bagging_params = {
    'bagging__n_estimators': [10, 50, 100, 500],
}

# random forest model pipeline
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier())
], verbose=True)

# rf parameters
rf_param = {
    'rf__n_estimators': [10, 50, 100,500],
    'rf__max_depth': [None, 5, 10],
}

#bagging__max_samples - controls the maximum number of samples to draw from the dataset for each base estimator it is a fraction between 0 and 1
# rf_ma_depth - control maximum depth of each decision tree (the number of levels iof trees)

#Fit bagging tree model
bagging_cv = GridSearchCV(bagging_pipeline, param_grid = bagging_params, cv = 10, scoring = 'accuracy', n_jobs=-1)# ,'f1'])
bagging_cv.fit(x, y)

# obtain the estimators
bagging_best =  bagging_cv.best_estimator_.named_steps['bagging']

# obtain the predictors 
bagging_importances = bagging_best.estimators_[0].feature_importances_
print(bagging_importances)

# %IncMSE/gini percent increase of MSE 
# estimate the importance of each predictor
# by looking at the increase MSE
# when the predictor variable is removed from the model
variables_names = x.columns.str.replace("_", " ")

var_imp_bagg = pd.DataFrame({'Predictors':variables_names, 'gini':bagging_importances})
var_imp_bagg_sorted = var_imp_bagg.sort_values(by = 'gini', ascending=False)
print("Top 5 important predictors for bagging trees model")
print(var_imp_bagg_sorted.head(5))

# chart of the important variables
importances = pd.Series(bagging_importances, index=variables_names) # series - one dimension array
top_n = 10
top_n_idx = importances.argsort()[-top_n:] # know which features have the highest and lowest importance values
top_n_importances = importances.iloc[top_n_idx] # nteger-location based indexing of a dataframe or a series
top_n_importances.plot(kind='barh', color='b') # bar chart
plt.xlabel('Average Gini Decrease')
plt.title(f'Top {top_n} Important Features: Bagging Decision Tree')


#Random forest model
rf_cv = GridSearchCV(rf_pipeline, param_grid= rf_param, cv = 10, scoring = 'accuracy', n_jobs=-1)
rf_cv.fit(x, y)

#calculate important predictors

# obtain the estimators
rf_best =  rf_cv.best_estimator_.named_steps['rf']

# obtain the predictors %incmse for RF
rf_importances = rf_best.estimators_[0].feature_importances_
print(rf_importances)

# top five important variables using %IncMSE
variables_names = x.columns.str.replace("_", " ")


var_imp_rf = pd.DataFrame({'Predictors':variables_names, 'gini':rf_importances})
var_imp_rf_sorted = var_imp_rf.sort_values(by = 'gini', ascending=False)
print("Top 5 important predictors for Random Forest trees model")
print(var_imp_rf_sorted.head(5))

importances = pd.Series(rf_importances, index=variables_names) # series - one dimension array
top_n = 10
top_n_idx = importances.argsort()[-top_n:] # know which features have the highest and lowest importance values
top_n_importances = importances.iloc[top_n_idx] # nteger-location based indexing of a dataframe or a series
top_n_importances.plot(kind='barh', color='b') # bar chart
plt.xlabel('Average Gini Decrease')
plt.title(f'Top {top_n} Important Features: Random Forest')

#boosting
from sklearn.ensemble import GradientBoostingClassifier

# boosting - GradientBoostingClassifier default trees
boost_tree_pipeline =  Pipeline([
    ('scaler', StandardScaler()),
   ('gbt', GradientBoostingClassifier())
], verbose = True)

# parameters
boost_tree_param =  {
   'gbt__learning_rate': [0.1, 0.05, 0.01],
    'gbt__n_estimators': [50, 100, 200],
    'gbt__max_depth': [3, 4, 5]
}

#n_estimators: number of decision trees to include in the ensemble.
#learning_rate: the learning rate shrinks the contribution of each tree by learning_rate amount.
#max_depth: the maximum depth of the decision trees.
#max_features: the number of features to consider when looking for the best split. 
#subsample: the fraction of samples to be used for fitting the individual base learners. Values lower than 1.0 would make the algorithm stochastic.

# fit model & cross validation k-fold
boosting_tree_cv = GridSearchCV(boost_tree_pipeline, param_grid=boost_tree_param, cv = 10, scoring = 'accuracy', n_jobs=-1)
boosting_tree_cv.fit(x, y)

#calculate important predictors

# obtain the estimators
boosting_best =  boosting_tree_cv.best_estimator_.named_steps['gbt']

# obtain the predictors %incmse for RF
boosting_importances = boosting_best.feature_importances_
print(boosting_importances)

# top five important variables using %IncMSE
variables_names = x.columns.str.replace("_", " ")


var_imp_boost = pd.DataFrame({'Predictors':variables_names, 'gini':boosting_importances})
var_imp_boost_sorted = var_imp_boost.sort_values(by = 'gini', ascending=False)
print("Top 5 important predictors for bagging trees model")
print(var_imp_boost_sorted.head(5))

importances = pd.Series(boosting_importances, index=variables_names) # series - one dimension array
top_n = 10
top_n_idx = importances.argsort()[-top_n:] # know which features have the highest and lowest importance values
top_n_importances = importances.iloc[top_n_idx] # nteger-location based indexing of a dataframe or a series
top_n_importances.plot(kind='barh', color='b') # bar chart
plt.xlabel('Average Gini Decrease')
plt.title(f'Top {top_n} Important Features: Boosting Decision Tree')

# logistic Lasso regression
from sklearn.linear_model import LogisticRegression

#logistic
logistic_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('log', LogisticRegression(penalty= 'l1',  solver='saga', max_iter = 2000))
], verbose = True) 

# hyperparameters 
param_grid = {
    'log__class_weight': [None, 'balanced']
}

# fit logistic lasso regression
logistic_cv = GridSearchCV(logistic_pipe, param_grid, cv=10, scoring='accuracy', n_jobs = -1)
logistic_cv.fit(x, y)

# Fit grid search object to the data
log_best_params = logistic_cv.best_params_
print(log_best_params)

# coefficient interms of log-odds and odds
from tabulate import tabulate

coeff = logistic_cv.best_estimator_.named_steps['log'].coef_

coef_df = pd.DataFrame({'Variable': x.columns.str.replace("_", " "),
                        'Log-Odds Coefficient': coeff[0],
                        'Odds Ratio coefficient': np.exp(coeff[0]) })


print(tabulate(coef_df.sort_values('Odds Ratio coefficient', ascending= False), headers='keys', tablefmt='psql'))

# random sample for prediction analysis
df5 = df4_imputed.sample(n= 20000, random_state=1)

# seperate predictors and response
x = df5.drop('is_satisfied', axis=1)
y = df5['is_satisfied']

# 70/30 split training/test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=1)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

k_val = list(range(1,40))
k_scores = []

for k in k_val:
    pipeline.set_params(knn__n_neighbors=k)
    scores = cross_val_score(pipeline, x_train, y_train, cv = 10, scoring = 'accuracy')
    k_scores.append(scores.mean())


# optimal k values for neighbors knn model 
opt_k = k_val[k_scores.index(max(k_scores))]
print(f"optimal k value:  {opt_k}")

# fit knn model to full dataset
pipeline.set_params(knn__n_neighbors=opt_k)
pipeline.fit(x_train, y_train)

# predictions
pred_knn = knn_cv.predict(x_test)

# confusion matrix
knn_cm = confusion_matrix(y_test, pred_knn)
print(knn_cm)
print(pd.crosstab(y_test, pred_knn, rownames=['True'], colnames=['Predicted'], margins=True))

sns.heatmap(knn_cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
#plt.show()

#  confusion matrix
report = classification_report(y_test, pred_knn)

# Print the report
print(report)
knn_accuracy = accuracy_score(y_test, pred_knn).round(2)
knn_precision = precision_score(y_test, pred_knn).round(2)
knn_recall = recall_score(y_test, pred_knn).round(2)
knn_f1 = f1_score(y_test, pred_knn).round(2)
knn_balanced_acc = balanced_accuracy_score(y_test, pred_knn). round(2)

print(f"knn model (k = {opt_k}): accuracy: {knn_accuracy}, precision: {knn_precision}, recal: {knn_recall}, f1 score: {knn_f1}", 'Balanced Accuracy: ' {knn_balanced_acc})

# decision tree model
# tune parameters, fit model on traning set for prediction 
tree_cv = GridSearchCV(tree_pipeline, tree_param, cv = 10, scoring = 'accuracy', n_jobs=-1)
tree_cv.fit(x_train, y_train)

#predictions on test set
tree_cv.best_estimator_
tree_pred = tree_cv.predict(x_test)

# confusion matrix
print(pd.crosstab(y_test, tree_pred, rownames=['True'], colnames=['Predicted'], margins=True))

# heatmap of results
tree_cm = confusion_matrix(y_test, tree_pred)
sns.heatmap(tree_cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

#  confusion matrix
report = classification_report(y_test, tree_pred)

# Print the report
print(report)
t_accuracy = accuracy_score(y_test, tree_pred).round(2)
t_precision = precision_score(y_test, tree_pred).round(2)
t_recall = recall_score(y_test, tree_pred).round(2)
t_f1 = f1_score(y_test, tree_pred).round(2)
t_balanced_acc = balanced_accuracy_score(y_test, tree_pred). round(2)

print(f"Decision tree model: accuracy: {t_accuracy}, precision: {t_precision}, recall: {t_recall}, f1 score: {t_f1}, balanced accuracy {t_balanced_acc}")

# cross validation bagging model
bagging_cv = GridSearchCV(bagging_pipeline, param_grid = bagging_params, cv = 10, scoring = 'accuracy', n_jobs=-1)# ,'f1'])
bagging_cv.fit(x_train, y_train)

# cross validation rf model 
rf_cv = GridSearchCV(rf_pipeline, param_grid= rf_param, cv = 10, scoring = 'accuracy', n_jobs=-1)
rf_cv.fit(x_train, y_train)

# parameters of n trees bagging
bagging_best_params = bagging_cv.best_params_
print(bagging_best_params)

# parameters of rf 
rf_best_params = rf_cv.best_params_
print(rf_best_params)

# prediction on test dataset using bagging model
pred_bagging = bagging_cv.predict(x_test)

bagging_cm = confusion_matrix(y_test, pred_bagging)
print(pd.crosstab(y_test, pred_bagging, rownames=['True'], colnames=['Predicted'], margins=True))

# heatmap of results
sns.heatmap(bagging_cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

#  confusion matrix
report = classification_report(y_test, pred_bagging)

# Print the report
print(report)
bag_accuracy = accuracy_score(y_test, pred_bagging).round(2)
bag_precision = precision_score(y_test, pred_bagging).round(2)
bag_recall = recall_score(y_test, pred_bagging).round(2)
bag_f1 = f1_score(y_test, pred_bagging).round(2)
bag_balanced_acc = balanced_accuracy_score(y_test, pred_bagging). round(2)


print(f"Decision tree model: accuracy: {bag_accuracy}, precision: {bag_precision}, recall: {bag_recall}, f1 score: {bag_f1}, Balanced accuracy {bag_balanced_acc}")

########### random forest ############
# random forest predicitons
pred_rf = rf_cv.predict(x_test)

rf_cm = confusion_matrix(y_test, pred_rf)
print(pd.crosstab(y_test, pred_rf, rownames=['True'], colnames=['Predicted'], margins=True))

# heatmap of results
sns.heatmap(rf_cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

#  confusion matrix
report_rf = classification_report(y_test, pred_rf)

# Print the report
print(report_rf)
rf_accuracy = accuracy_score(y_test, pred_rf).round(2)
rf_precision = precision_score(y_test, pred_rf).round(2)
rf_recall = recall_score(y_test, pred_rf).round(2)
rf_f1 = f1_score(y_test, pred_rf).round(2)
rf_balanced_acc = balanced_accuracy_score(y_test, pred_rf). round(2)

print(f"Decision tree model: accuracy: {rf_accuracy}, precision: {rf_precision}, recall: {rf_recall}, f1 score: {rf_f1}, Balanced accuracy {rf_balanced_acc}")

# boosting model
# fit model & cross validation k-fold
boosting_tree_cv = GridSearchCV(boost_tree_pipeline, param_grid=boost_tree_param, cv = 10, scoring = 'accuracy', n_jobs=-1)
boosting_tree_cv.fit(x_train, y_train)

# parameters of boosting 
boost_best_params = boosting_tree_cv.best_params_
print(boost_best_params)

#{'gbt__learning_rate': 0.1, 'gbt__max_depth': 6, 'gbt__n_estimators': 200}

# prediction on test dataset using bagging model
pred_boosting = boosting_tree_cv.predict(x_test)

print(pd.crosstab(y_test, pred_boosting, rownames=['True'], colnames=['Predicted'], margins=True))

# heatmap of results
boosting_cm = confusion_matrix(y_test, pred_boosting) # matrix

sns.heatmap(boosting_cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

#  confusion matrix
report_boosting = classification_report(y_test, pred_boosting)

# Print the report
print(report_boosting)
boost_accuracy = accuracy_score(y_test, pred_boosting).round(2)
boost_precision = precision_score(y_test, pred_boosting).round(2)
boost_recall = recall_score(y_test, pred_boosting).round(2)
boost_f1 = f1_score(y_test, pred_boosting).round(2)
boost_balanced_acc = balanced_accuracy_score(y_test, pred_boosting). round(2)


print(f"Decision tree model: accuracy: {boost_accuracy}, precision: {boost_precision}, recall: {boost_recall}, f1 score: {boost_f1}, Balanced accuracy {boost_balanced_acc}")

#################### SVM
from sklearn.svm import SVC

# SVM MODEL
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True))
], verbose=True)

# svm parameter
svm_grid = {
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'poly','rbf'],
    'svm__gamma': [0.1, 1, 10 ]
}

# fit model & cross validation k-fold
svm_cv = GridSearchCV(svm_pipeline, param_grid=svm_grid, cv = 10, scoring = 'accuracy', n_jobs=-1)
svm_cv.fit(x_train, y_train)

# parameters of svm
svm_best_params = svm_cv.best_params_
print(svm_best_params)

# prediction on test dataset using svm model
pred_svm = svm_cv.predict(x_test)

print(pd.crosstab(y_test, pred_svm, rownames=['True'], colnames=['Predicted'], margins=True))


# heatmap of results
svm_cm = confusion_matrix(y_test, pred_svm) # matrix

sns.heatmap(svm_cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

#  confusion matrix
report_svm = classification_report(y_test, pred_svm)

# Print the report
print(report_svm)
svm_accuracy = accuracy_score(y_test, pred_svm).round(2)
svm_precision = precision_score(y_test, pred_svm).round(2)
svm_recall = recall_score(y_test, pred_svm).round(2)
svm_f1 = f1_score(y_test, pred_svm).round(2)
svm_balanced_acc = balanced_accuracy_score(y_test, pred_svm). round(2)


print(f"SVM model: accuracy: {svm_accuracy}, precision: {svm_precision}, recall: {svm_recall}, f1 score: {svm_f1}, balanced accuracy {svm_balanced_acc}")

# Fit grid search object to the data
log_best_params = logistic_cv.best_params_
print(log_best_params)

# predict on test set
pred_log = logistic_cv.predict(x_test)
print(pd.crosstab(y_test, pred_log, rownames=['True'], colnames=['Predicted'], margins=True))

# heatmap of results
log_cm = confusion_matrix(y_test, pred_log) # matrix

sns.heatmap(log_cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

#  confusion matrix
report_log = classification_report(y_test, pred_log)

# Print the report
print(report_log)
log_accuracy = accuracy_score(y_test, pred_log).round(2)
log_precision = precision_score(y_test, pred_log).round(2)
log_recall = recall_score(y_test, pred_log).round(2)
log_f1 = f1_score(y_test, pred_log).round(2)
log_balanced_acc = balanced_accuracy_score(y_test, pred_log). round(2)


print(f"Logistic Lasso Regression model: accuracy: {log_accuracy}, precision: {log_precision}, recall: {log_recall}, f1 score: {log_f1}, balanced accuracy {log_balanced_acc}")

#AUC curve
from sklearn.metrics import roc_curve, auc

# probabilites on test set logisitc
log_y_prob = logistic_cv.predict_proba(x_test)[:, 1]
log_fpr, log_tpr, log_thresholds = roc_curve(y_test, log_y_prob)
log_roc_auc = auc(log_fpr, log_tpr)

# decision tree
tree_y_prob = tree_cv.predict_proba(x_test)[:, 1]
tree_fpr, tree_tpr, tree_thresholds = roc_curve(y_test, tree_y_prob)
tree_roc_auc = auc(tree_fpr, tree_tpr)

# bagging trees
bagging_y_prob = bagging_cv.predict_proba(x_test)[:, 1]
bagging_fpr, bagging_tpr, bagging_thresholds = roc_curve(y_test, bagging_y_prob)
bagging_roc_auc = auc(bagging_fpr, bagging_tpr)

# Boosting trees
boosting_y_prob = boosting_tree_cv.predict_proba(x_test)[:, 1]
boosting_fpr, boosting_tpr, boosting_thresholds = roc_curve(y_test, boosting_y_prob)
boosting_roc_auc = auc(boosting_fpr, boosting_tpr)

# Random Forest
rf_y_prob = rf_cv.predict_proba(x_test)[:, 1]
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf_y_prob)
rf_roc_auc = auc(rf_fpr, rf_tpr)

# SVM
svm_y_prob = svm_cv.predict_proba(x_test)[:, 1]
svm_fpr, svm_tpr, svm_thresholds = roc_curve(y_test, svm_y_prob)
svm_roc_auc = auc(svm_fpr, svm_tpr)

print(f"Logistic AUC: {log_roc_auc.round(2)}")
print(f"Decision tree AUC: {tree_roc_auc.round(2)}")
print(f"Bagging Decision tree AUC: {bagging_roc_auc.round(2)}")
print(f"Boosting Decision tree AUC: {boosting_roc_auc.round(2)}")
print(f"Random Forest AUC: {rf_roc_auc.round(2)}")
print(f"SVM AUC: {svm_roc_auc.round(2)}")

# Plot ROC curve
plt.figure()
plt.plot(log_fpr, log_tpr, color='skyblue', lw=1, label='Logistic ROC curve (area = %0.3f)' % log_roc_auc)
plt.plot(tree_fpr, tree_tpr, color='green', lw=1, label='Decision trees ROC curve (area = %0.3f)' % tree_roc_auc)
plt.plot(bagging_fpr, bagging_tpr, color='red', lw=1, label='Bagging Decision trees ROC curve (area = %0.3f)' % bagging_roc_auc)
plt.plot(boosting_fpr, boosting_tpr, color='orange', lw=1, label='Boosting Decision trees ROC curve (area = %0.3f)' % boosting_roc_auc)
plt.plot(rf_fpr, rf_tpr, color='purple', lw=1, label='Random Forest ROC curve (area = %0.3f)' % rf_roc_auc)
plt.plot(svm_fpr, svm_tpr, color='gold', lw=1, label='SVM ROC curve (area = %0.3f)' % svm_roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


tb = {
    'Models'   : ['Boosting Decision tree', 'Random Forest', 'Bagging Decision tree', 'SVM', 'Decision tree', 'KNN', 'Logistic Lasso Regression'],
    "Accuracy" : [boost_accuracy, rf_accuracy, bag_accuracy, svm_accuracy, t_accuracy, knn_accuracy ,log_accuracy],
    'Precision': [boost_precision, rf_precision, bag_precision, svm_precision, t_precision, knn_precision, log_precision],
    'Recall'   : [boost_recall,  rf_recall, bag_recall, svm_recall, t_recall, t_precision,log_recall],
    'F1 Score' : [boost_f1,  rf_f1, bag_f1, svm_f1, t_f1, t_f1,log_f1],
    'Balanced accuracy' : [boost_balanced_acc, rf_balanced_acc, bag_balanced_acc, svm_balanced_acc, t_balanced_acc, log_balanced_acc ,log_balanced_acc]
}

tb_df = pd.DataFrame(tb)
print(tb_df.to_markdown(index = False))




