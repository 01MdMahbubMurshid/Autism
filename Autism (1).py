#!/usr/bin/env python
# coding: utf-8

# In[1]:


#necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


# In[2]:


adu = pd.read_csv("combined1.csv",na_values=['?'])


# In[3]:


adu.sample(5)


# In[4]:


adu.info()


# In[5]:


adu.describe()


# In[6]:


print("Total no. of missing values in Adults's dataset     : ",adu.isnull().sum().sum())


# In[7]:


#Correlation
plt.figure(figsize=(12,12))
sns.heatmap(adu.corr(), cmap='jet', linewidths=1, linecolor='black', annot=True)
plt.show()


# In[8]:


#Imputing missing values of categorical features with mode
imputer_mode = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')



adu.age = imputer_mode.fit_transform(adu.age.values.reshape(-1,1))[:,0]
adu.ethnicity = imputer_mode.fit_transform(adu.ethnicity.values.reshape(-1,1))[:,0]
adu.relation = imputer_mode.fit_transform(adu.relation.values.reshape(-1,1))[:,0]


#Imputing missing values of numerical features with mean
imputer_mode = SimpleImputer(missing_values=np.NaN, strategy='mean')


adu.age = imputer_mode.fit_transform(adu.age.values.reshape(-1,1))[:,0]


# In[9]:


print("After imputing,\nNo of missing values in dataset      = ",adu.isnull().sum().sum())


# In[10]:


#Since age of toddlers are represented in months, age(in years) of children, adolescents and adults is converted to age in months.
adu.rename(columns = {'age':'Age_Mons'}, inplace = True)

adu['Age_Mons'] = adu['Age_Mons']*12


# In[11]:


#Making classes of categorical variables same for all datasets
adu['ethnicity'] = adu['ethnicity'].replace('Others','others')

adu["relation"] = adu["relation"].replace('self','Self')


# In[12]:


#Adding a new field that represents the age group

adu['Age_group'] = 'Adults'


# In[13]:


#imputing missing values
imputer_mode = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')

adu.contry_of_res = imputer_mode.fit_transform(adu.contry_of_res.values.reshape(-1,1))[:,0]
adu.used_app_before = imputer_mode.fit_transform(adu.used_app_before.values.reshape(-1,1))[:,0]


# In[14]:


adu.head()


# In[15]:


adu.shape


# In[16]:


print("\nNo of individuals diagonised with ASD = ",len(adu[adu['Class/ASD'] == 'YES']))
print("No of individuals not diagonised with ASD = ",len(adu[adu['Class/ASD'] == 'NO']))


# In[17]:


#Correlation
fig, ax = plt.subplots(figsize=(10,7))
sns.heatmap(adu.corr(),annot=True,cmap='coolwarm',vmin=0, vmax=1,ax=ax)
ax.set_title("Combined dataset",fontsize = 15)
fig.tight_layout()
plt.savefig('correlation_final.pdf', transparent=True, dpi=300)


# In[18]:


shuffled_data = adu.sample(frac=1,random_state=4)
ASD_data = shuffled_data.loc[shuffled_data['Class/ASD'] == 'YES']
non_ASD_data = shuffled_data.loc[shuffled_data['Class/ASD'] == 'NO'].sample(n=666)
final= pd.concat([ASD_data, non_ASD_data])


# In[19]:


# Split the data into features and target label
raw_target= final['Class/ASD']
raw_features = final[['A1_Score','A2_Score','A3_Score','A4_Score','A5_Score','A6_Score','A7_Score','A8_Score','A9_Score','A10_Score','Age_Mons', 'gender', 'ethnicity', 'jundice', 'austim', 'contry_of_res', 'result','relation']]


# In[20]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
num_features = ['Age_Mons', 'result']

features_minmax_transform = pd.DataFrame(data = raw_features)
features_minmax_transform[num_features] = scaler.fit_transform(raw_features[num_features])


# In[21]:


features_minmax_transform.head()


# In[22]:


#Encoding Categorical variAbles
features = pd.get_dummies(features_minmax_transform)
print('features.shape:', features.shape)


# In[23]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
target = le.fit_transform(raw_target)


# In[24]:


#Function for Evaluation
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, roc_curve, auc, log_loss

def model_report(y_act, y_pred):
    print("Accuracy = ", accuracy_score(y_act, y_pred))
    print("Precision = " ,precision_score(y_act, y_pred))
    print("Recall\Sensitivity = " ,recall_score(y_act, y_pred))
    confusion = metrics.confusion_matrix(y_act, y_pred)
    #[row, column]
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    specificity = TN / (TN + FP)
    print("Specificity = " ,specificity)
    print("F1 Score = " ,f1_score(y_act, y_pred))
    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_act, y_pred)
    print("AUC Score =", auc(false_positive_rate, true_positive_rate))
    print("Kappa score = ",cohen_kappa_score(y_act,y_pred))
    print("Log Loss = " ,log_loss(y_act, y_pred),"\n")


# In[25]:


import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("AUC")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring = 'roc_auc')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="b")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="b",
             label="Cross-validation score")

    plt.legend(loc="lower right")
    return plt


# In[26]:


X = features
y = target


# In[27]:


from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2  
chi2_features = SelectKBest(chi2,k=75)
fit= chi2_features.fit(X, y)
scores = pd.DataFrame(fit.scores_)
columns = pd.DataFrame(features.columns)
featureScores = pd.concat([columns,scores],axis=1)
featureScores.columns = ['Features','Score']
print(featureScores.nlargest(50,'Score')) 


# In[28]:


from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2  
# 700 features with highest chi-squared statistics are selected 
chi2_features = SelectKBest(chi2,k=75)
X = chi2_features.fit_transform(X, y)
y = target


# In[29]:


#Splitting the data into train test spit
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)


# In[43]:


from sklearn.model_selection import RandomizedSearchCV

n_neighbors = [int(x) for x in np.linspace(1,1000,10)]
weights = ['uniform','distance']
algorithm =['auto', 'ball_tree', 'kd_tree', 'brute']
leaf_size = [int(x) for x in np.linspace(1,1000,10)]

random_grid = {'n_neighbors':n_neighbors,
               'weights':weights,
               'algorithm':algorithm,
               'leaf_size':leaf_size}
print(random_grid)


# In[44]:


from sklearn import neighbors

knn = neighbors.KNeighborsClassifier()
knn_randomcv=RandomizedSearchCV(estimator= knn, param_distributions=random_grid, n_iter=100, cv=5, verbose=2,
                               random_state=100,n_jobs=-1)
#fit the randomized model
knn_randomcv.fit(X_train,y_train)


# In[45]:


print('\n All results:')
print(knn_randomcv.cv_results_)


# In[46]:


print('\n Best estimator:')
print(knn_randomcv.best_estimator_)


# In[47]:


print('\n Best hyperparameters:')
print(knn_randomcv.best_params_)


# In[48]:


from sklearn import neighbors

knn = neighbors.KNeighborsClassifier(weights='uniform', n_neighbors= 112, leaf_size=112, algorithm='brute')
knn.fit(X_train,y_train)


# In[49]:


y_pred_knn = knn.predict(X_test)
model_report(y_test, y_pred_knn)


# In[50]:


title = "Learning Curves (KNeighborsClassifier)"                                        # fill this in
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
knn = neighbors.KNeighborsClassifier(weights='uniform', n_neighbors= 112, leaf_size=112, algorithm='brute')                                 # fill this in
plot_learning_curve(knn, title, X, y, ylim=(0.2, 1.25), cv=cv, n_jobs=4)
plt.savefig('KNN_curve.pdf', transparent=True, dpi=300)
plt.savefig('KNN_curve.eps', transparent=True, dpi=300)
plt.show()


# In[51]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(10, 8))
ax = sns.heatmap(cm, cmap=plt.cm.Greens, annot=True, square=True, annot_kws={"size": 25})
plt.title('Confusion Matrix', fontsize=20)
ax.set_ylabel('Actual Label', fontsize=20)
ax.set_xlabel('Predicted Label', fontsize=20)
plt.savefig('KNN_confusion.pdf', transparent=True, dpi=300)
plt.savefig('KNN_confusion.eps', transparent=True, dpi=300)


# In[52]:


#SVM CLASSIFIER
from sklearn.model_selection import RandomizedSearchCV

C = [int(x) for x in np.linspace(start = 1, stop = 20, num = 10)]
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
degree = [int(x) for x in np.linspace(start = 1, stop = 10, num = 10)]

random_grid = {'C':C,
               'kernel':kernel,
               'degree':degree}
print(random_grid)


# In[53]:


from sklearn.svm import SVC
svc = SVC()
svc_randomcv=RandomizedSearchCV(estimator= svc, param_distributions=random_grid, n_iter=100, cv=5, verbose=2,
                               random_state=100,n_jobs=-1)
#fit the randomized model
svc_randomcv.fit(X_train,y_train)


# In[54]:


print('\n All results:')
print(svc_randomcv.cv_results_)


# In[55]:


print('\n Best estimator:')
print(svc_randomcv.best_estimator_)


# In[56]:


print('\n Best hyperparameters:')
print(svc_randomcv.best_params_)


# In[57]:


svc = SVC(kernel='linear',degree=3, C=13)
svc.fit(X_train,y_train)


# In[58]:


y_pred_svc = svc.predict(X_test)
model_report(y_test, y_pred_svc)


# In[59]:


title = "Learning Curves (SVM)"                                        # fill this in
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
svc = SVC(kernel='linear',degree=3, C=13)                               # fill this in
plot_learning_curve(svc, title, X, y, ylim=(0.2, 1.25), cv=cv, n_jobs=4)
plt.savefig('SVM_curve.pdf', transparent=True, dpi=300)
plt.savefig('SVM_curve.eps', transparent=True, dpi=300)
plt.show()


# In[60]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_svc)
plt.figure(figsize=(10, 8))
ax = sns.heatmap(cm, cmap=plt.cm.Greens, annot=True, square=True, annot_kws={"size": 25})
plt.title('Confusion Matrix', fontsize=20)
ax.set_ylabel('Actual Label', fontsize=20)
ax.set_xlabel('Predicted Label', fontsize=20)
plt.savefig('SVC_confusion.pdf', transparent=True, dpi=300)
plt.savefig('SVC_confusion.eps', transparent=True, dpi=300)


# In[61]:


#Random Forest Classifier
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt','log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 1000,10)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10,14]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4,6,8]
#criterion used in trees
criterion = ['entropy','gini']

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
              'criterion':criterion}
              
print(random_grid)


# In[62]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf_randomcv=RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=5, verbose=2,
                               random_state=100, n_jobs=-1)
#fit the randomized model
rf_randomcv.fit(X_train,y_train)


# In[63]:


print('\n All results:')
print(rf_randomcv.cv_results_)


# In[64]:


print('\n Best estimator:')
print(rf_randomcv.best_estimator_)


# In[65]:


print('\n Best hyperparameters:')
print(rf_randomcv.best_params_)


# In[66]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 200, min_samples_split = 2, min_samples_leaf = 2, max_features = 'sqrt', max_depth = 1000, criterion = 'entropy')
rf.fit(X_train, y_train)


# In[67]:


y_pred_rf = rf.predict(X_test)
model_report(y_test, y_pred_rf)


# In[72]:


title = "Learning Curves (RandomForestClassifier)"                                        # fill this in
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators = 200, min_samples_split = 2, min_samples_leaf = 2, max_features = 'sqrt', max_depth = 1000, criterion = 'entropy')                             # fill this in
plot_learning_curve(rf, title, X, y, ylim=(0.2, 1.25), cv=cv, n_jobs=4)
plt.savefig('RF_curve.pdf', transparent=True, dpi=300)
plt.savefig('RF_curve.eps', transparent=True, dpi=300)
plt.show()


# In[69]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(10, 8))
ax = sns.heatmap(cm, cmap=plt.cm.Greens, annot=True, square=True, annot_kws={"size": 25})
plt.title('Confusion Matrix', fontsize=20)
ax.set_ylabel('Actual Label', fontsize=20)
ax.set_xlabel('Predicted Label', fontsize=20)
plt.savefig('RF_confusion.pdf', transparent=True, dpi=300)
plt.savefig('RF_confusion.eps', transparent=True, dpi=300)


# In[71]:


from sklearn.tree import DecisionTreeClassifier

param_dict ={
             "criterion":['gini','entropy'],
             "max_depth":(150, 155, 160),
             "min_samples_split":range(1,10),
             "min_samples_leaf":range(1,5)
 }

decision_tree = DecisionTreeClassifier(random_state=42)

dt_decision_tree=RandomizedSearchCV(estimator=decision_tree, param_distributions=param_dict, n_iter=100, cv=5, verbose=2,
                               random_state=100,n_jobs=-1)
#fit the randomized model
dt_decision_tree.fit(X_train,y_train)


# In[73]:


print('\n All results:')
print(dt_decision_tree.cv_results_)


# In[74]:


print('\n Best estimator:')
print(dt_decision_tree.best_estimator_)


# In[75]:


print('\n Best hyperparameters:')
print(dt_decision_tree.best_params_)


# In[76]:


from sklearn.ensemble import RandomForestClassifier
dt = DecisionTreeClassifier(min_samples_split = 3, min_samples_leaf = 3, max_depth = 160, criterion = 'gini')
dt.fit(X_train, y_train)


# In[77]:


y_pred_dt = dt.predict(X_test)
model_report(y_test, y_pred_dt)


# In[78]:


title = "Learning Curves (DecisionTreeClassifier)"                                        # fill this in
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
dt = DecisionTreeClassifier(min_samples_split = 3, min_samples_leaf = 3, max_depth = 160, criterion = 'gini')                             # fill this in
plot_learning_curve(dt, title, X, y, ylim=(0.2, 1.25), cv=cv, n_jobs=4)
plt.savefig('DT_curve.pdf', transparent=True, dpi=300)
plt.savefig('DT_curve.eps', transparent=True, dpi=300)
plt.show()


# In[80]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_dt)
plt.figure(figsize=(10, 8))
ax = sns.heatmap(cm, cmap=plt.cm.Greens, annot=True, square=True, annot_kws={"size": 25})
plt.title('Confusion Matrix', fontsize=20)
ax.set_ylabel('Actual Label', fontsize=20)
ax.set_xlabel('Predicted Label', fontsize=20)
plt.savefig('DT_confusion.pdf', transparent=True, dpi=300)
plt.savefig('DT_confusion.eps', transparent=True, dpi=300)


# In[81]:


from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

estimator = XGBClassifier(
    objective= 'binary:logistic',
    nthread=4,
    seed=42
)

parameters = {
    'max_depth': range (2, 10, 1),
    'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05]
}

xg_randomcv = RandomizedSearchCV(
    estimator=estimator,
    param_distributions=parameters,
    scoring = 'roc_auc',
    n_jobs = 10,
    cv = 5,
    verbose=True
)

#fit the randomized model
xg_randomcv.fit(X_train,y_train)


# In[82]:


print('\n All results:')
print(xg_randomcv.cv_results_)


# In[83]:


print('\n Best estimator:')
print(xg_randomcv.best_estimator_)


# In[84]:


print('\n Best hyperparameters:')
print(xg_randomcv.best_params_)


# In[85]:


from xgboost import XGBClassifier
xg = XGBClassifier(n_estimators = 140, max_depth = 4, learning_rate = 0.01)
xg.fit(X_train, y_train)


# In[86]:


y_pred_xg = xg.predict(X_test)
model_report(y_test, y_pred_xg)


# In[87]:


title = "Learning Curves (XGBClassifier)"                                        # fill this in
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
xg = XGBClassifier(n_estimators = 140, max_depth = 2, learning_rate = 0.01)                             # fill this in
plot_learning_curve(xg, title, X, y, ylim=(0.2, 1.25), cv=cv, n_jobs=4)
plt.savefig('XG_curve.pdf', transparent=True, dpi=300)
plt.savefig('XG_curve.eps', transparent=True, dpi=300)
plt.show()


# In[88]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_xg)
plt.figure(figsize=(10, 8))
ax = sns.heatmap(cm, cmap=plt.cm.Greens, annot=True, square=True, annot_kws={"size": 25})
plt.title('Confusion Matrix', fontsize=20)
ax.set_ylabel('Actual Label', fontsize=20)
ax.set_xlabel('Predicted Label', fontsize=20)
plt.savefig('XG_confusion.pdf', transparent=True, dpi=300)
plt.savefig('XG_confusion.eps', transparent=True, dpi=300)


# In[89]:


from sklearn.linear_model import LogisticRegression

grid_values = {'penalty': ['l1','l2'], 
               'C': [0.001,0.01,0.1,1,10,100,1000]}
lr=LogisticRegression(random_state=42)
lr_randomcv = RandomizedSearchCV(lr, param_distributions=grid_values, cv=5)
#fit the randomized model
lr_randomcv.fit(X_train,y_train)


# In[90]:


print('\n All results:')
print(lr_randomcv.cv_results_)


# In[91]:


print('\n Best estimator:')
print(lr_randomcv.best_estimator_)


# In[92]:


print('\n Best hyperparameters:')
print(lr_randomcv.best_params_)


# In[93]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty = 'l2', C = 100)
lr.fit(X_train, y_train)


# In[94]:


y_pred_lr = lr.predict(X_test)
model_report(y_test, y_pred_lr)


# In[95]:


title = "Learning Curves (LogisticRegression)"                                        # fill this in
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
lr = LogisticRegression(penalty = 'l2', C = 100)                            # fill this in
plot_learning_curve(lr, title, X, y, ylim=(0.2, 1.25), cv=cv, n_jobs=4)
plt.savefig('LR_curve.pdf', transparent=True, dpi=300)
plt.savefig('LR_curve.eps', transparent=True, dpi=300)
plt.show()


# In[96]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(10, 8))
ax = sns.heatmap(cm, cmap=plt.cm.Greens, annot=True, square=True, annot_kws={"size": 25})
plt.title('Confusion Matrix', fontsize=20)
ax.set_ylabel('Actual Label', fontsize=20)
ax.set_xlabel('Predicted Label', fontsize=20)
plt.savefig('LR_confusion.pdf', transparent=True, dpi=300)
plt.savefig('LR_confusion.eps', transparent=True, dpi=300)


# In[97]:


from sklearn.metrics import roc_curve

# Compute fpr, tpr, thresholds and roc auc
fpr_DT, tpr_DT, thresholds = roc_curve(y_test, y_pred_dt)
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test, y_pred_knn)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(y_test, y_pred_lr)
fpr_svc, tpr_svc, thresholds_svc = roc_curve(y_test, y_pred_svc)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_pred_rf)
fpr_xg, tpr_xg, thresholds_xg = roc_curve(y_test, y_pred_xg)


# In[98]:


from sklearn.metrics import auc

roc_auc_DT = auc(fpr_DT, tpr_DT)
roc_auc_knn = auc(fpr_knn, tpr_knn)
roc_auc_LR = auc(fpr_LR, tpr_LR)
roc_auc_svc = auc(fpr_svc, tpr_svc)
roc_auc_rf = auc(fpr_rf, tpr_rf)
roc_auc_xg = auc(fpr_xg, tpr_xg)


# In[99]:


from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# Compute fpr, tpr, thresholds and roc auc
plt.figure(figsize = (7, 5))
plt.plot(fpr_knn, tpr_knn, label='KNN ROC curve (area = %0.3f)' % roc_auc_knn)
plt.plot(fpr_LR, tpr_LR, label='LR ROC curve (area = %0.3f)' % roc_auc_LR)
plt.plot(fpr_svc, tpr_svc, label='SVM ROC curve (area = %0.3f)' % roc_auc_svc)
plt.plot(fpr_DT, tpr_DT, label='DT ROC curve (area = %0.3f)' % roc_auc_DT)
plt.plot(fpr_rf, tpr_rf, label='RF ROC curve (area = %0.3f)' % roc_auc_rf)
plt.plot(fpr_xg, tpr_xg, label='XGB ROC curve (area = %0.3f)' % roc_auc_xg)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for 80%-20% Splitting')
plt.legend(loc="lower right")
plt.savefig('all_roc.pdf', transparent=True, dpi=300)
plt.savefig('all_roc.eps', transparent=True, dpi=300)
plt.show()


# In[117]:


from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# Compute fpr, tpr, thresholds and roc auc
plt.figure(figsize = (7, 5))

plt.plot(fpr_LR, tpr_LR, label='Voting ROC curve (area = %0.3f)' % roc_auc_LR)

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for 80%-20% Splitting')
plt.legend(loc="lower right")
plt.savefig('all_roc.pdf', transparent=True, dpi=300)
plt.savefig('all_roc.eps', transparent=True, dpi=300)
plt.show()


# In[56]:


from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2  
# 700 features with highest chi-squared statistics are selected 
chi2_features = SelectKBest(chi2,k=20)
X = chi2_features.fit_transform(features, target)
y = target

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)


# In[66]:


importance = pd.DataFrame(model.feature_importances_)
columns = pd.DataFrame(features.columns)
featureScores = pd.concat([columns,importance],axis=1)
featureScores.columns = ['Features','importance']
print(featureScores.nlargest(20,'importance')) 


# In[62]:


plt.figure(figsize=(20,7))
featureScores.nlargest(20,'importance').plot(kind='barh')
plt.title('Feature Importance')
plt.savefig('feature_im_combined.pdf', 
           transparent=True, dpi=300)
plt.savefig('feature_im_combined.eps', 
           transparent=True, dpi=300)


# In[104]:


#Step_1: Import VotingClassifier
from sklearn.ensemble import VotingClassifier

#Step_2: Import all base classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

#Step_3: Create all base classifier
b_rf  = RandomForestClassifier(n_estimators=100, max_features='sqrt')
#b_knn = KNeighborsClassifier(n_neighbors=3)
b_dt = DecisionTreeClassifier()
b_xg = xgb.XGBClassifier(learning_rate=0.6,max_depth=5)



#Step_4: Create the VotingClassifier 
VC = VotingClassifier (estimators=[('cl1', b_rf), ('cl3', b_xg), ('cl9', b_dt)], voting='soft') 
                                                                        # you can use hard voting or soft voting
VC.fit(X_train, y_train)
y_pred_vc = VC.predict(X_test)
model_report(y_test, y_pred_vc)


# In[105]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print(accuracy_score(y_test, y_pred_vc))


# In[107]:


from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
print(classification_report(y_test,y_pred_vc,target_names=['0', '1']))
score = metrics.accuracy_score(y_test, y_pred_vc)*100
print("accuracy:   %0.2f" % score)


# In[113]:


sns.heatmap(confusion_matrix(y_test, y_pred_vc), cmap='coolwarm', linecolor='white', linewidths=1, annot=True)
plt.show()


# In[108]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_vc)
plt.figure(figsize=(10, 8))
ax = sns.heatmap(cm, cmap=plt.cm.Greens, annot=True, square=True, annot_kws={"size": 25})
plt.title('Confusion Matrix', fontsize=20)
ax.set_ylabel('Actual Label', fontsize=20)
ax.set_xlabel('Predicted Label', fontsize=20)
plt.savefig('LR_confusion.pdf', transparent=True, dpi=300)
plt.savefig('LR_confusion.eps', transparent=True, dpi=300)


# In[ ]:




