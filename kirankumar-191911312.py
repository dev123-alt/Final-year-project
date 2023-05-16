#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install catboost 


# In[70]:


# To read data and numeric operations
import pandas as pd
import numpy as np
from statistics import mean
# Graph plotting library
import matplotlib.pyplot as plt
import seaborn as sns
# Preprocessing
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
get_ipython().system('pip install imbalanced-learn')
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split, cross_val_score
# Sklearn Models
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
# Evaluation Metrics
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report,ConfusionMatrixDisplay,  precision_score, recall_score, f1_score, roc_auc_score,roc_curve,confusion_matrix
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[71]:


data = pd.read_csv("C:\\Users\\Windows\\Downloads\\archive (1)\\hypothyroid.csv")


# In[72]:


df = data.copy()


# In[73]:


df.head()


# In[74]:


df.shape


# In[75]:


df.columns


# In[76]:


df.duplicated().sum()


# In[77]:


df.drop_duplicates(keep = False, inplace=True)


# In[78]:


df.duplicated().sum()


# In[79]:


df.isnull().sum()


# In[80]:


df.dtypes


# In[81]:


df.info()


# In[82]:


df.head()


# In[83]:


for i in df.columns:
 print(f"'?' count in '{i}' column is {len(df[df[i]=='?'])} and percentage count is {round(len(df[df[i]=='?'])/df.shape[0]*100, 4)}")


# In[84]:


df = df.replace('?', np.nan)


# In[85]:


df.head(3)


# In[86]:


df.isna().sum()


# In[87]:


fig, ax = plt.subplots(figsize=(15,5))
missing = df.isna().sum().div(df.shape[0]).mul(100).to_frame().sort_values(by=0, ascending=False)
ax.bar(missing.index, missing.values.T[0])
plt.xticks([])
plt.ylabel("Percentage missing")
plt.show()


# In[88]:


missing[missing[0]>20]


# In[89]:


#df.drop('TBG', axis=1, inplace=True)
del df['TBG']


# In[90]:


df.drop(['TSH measured', 'T3 measured', 'TT4 measured', 'T4U measured', 'FTI measured', 'TBG measured'], axis=1, inplace=True)


# In[91]:


df.shape


# In[92]:


df = df.replace({'f':0, 't':1})


# In[93]:


df.head(2)


# In[94]:


df['binaryClass'].value_counts()


# In[95]:


df['binaryClass'] = df['binaryClass'].replace({'N':0, 'P':1})


# In[96]:


df['binaryClass'].head()


# In[97]:


df = pd.get_dummies(df, columns=['referral source'])


# In[98]:


df.head(1)


# In[99]:


df['sex'].value_counts()


# In[100]:


df['sex'] = df['sex'].replace({'F':0, 'M':1})


# In[101]:


obj_cols = df.columns[df.dtypes.eq('O')]
df[obj_cols] = df[obj_cols].apply(pd.to_numeric, errors='coerce') # ignore errors
df.dtypes


# In[102]:


sns.distplot(df['age'])


# In[103]:


sns.boxplot(df['age'])


# In[104]:


plt.figure(figsize=(20,10))
df.value_counts('sex').plot(kind="pie",autopct = '%1.1f%%')
plt.title ("Feature name : sex", fontsize = 15)


# In[105]:


plt.figure(figsize=(20,10))
df.value_counts('sick').plot(kind="pie",autopct = '%1.1f%%')
plt.title ("Feature name : sick", fontsize = 15)


# In[106]:


sns.countplot('pregnant', data=df, palette="tab10")
plt.title('not-pregnant: 0, pregnant: 1', fontsize=15)


# In[107]:


plt.figure(figsize=(20,10))
df.value_counts('pregnant').plot(kind="pie",autopct = '%1.1f%%')
plt.title ("Feature name : pregnant", fontsize = 15)


# In[108]:


sns.countplot('thyroid surgery', data=df, palette="tab10")
plt.title('thyroid surgery', fontsize=15)


# In[109]:


len(df[df['thyroid surgery'] == 1])/len(df['thyroid surgery'])*100


# In[110]:


plt.figure(figsize=(20,10))
df.value_counts('I131 treatment').plot(kind="pie",autopct = '%1.1f%%')
plt.title ("Feature name : I131 treatment", fontsize = 15)


# In[111]:


sns.countplot('lithium', data=df, palette="tab10")
plt.title('lithium', fontsize=15)


# In[112]:


plt.figure(figsize=(20,10))
df.value_counts('tumor').plot(kind="pie",autopct = '%1.1f%%')
plt.title ("Feature name : tumor", fontsize = 15)


# In[113]:


df['age'].median()


# In[114]:


df['age'] = df['age'].replace(np.nan, df['age'].median())
df['age'].isna().sum()


# In[115]:


df['sex'].value_counts()


# In[116]:


df['sex'] = df['sex'].replace(np.nan, df['sex'].mode()[0])
df['sex'].isna().sum()


# In[117]:


df['sex'].value_counts()


# In[118]:


df['sex'] = df['sex'].replace(np.nan, df['sex'].mode()[0])
df['sex'].isna().sum()


# In[119]:


df.isna().sum().sum()


# In[120]:


df[df['age'].max() > df['age']]['age'].max()


# In[121]:


median = df.loc[df['age']<94, 'age'].median()
df.loc[df.age > 94, 'age'] = np.nan
df.fillna(median,inplace=True)


# In[122]:


sns.boxplot(df['age'])


# In[123]:


sns.countplot('binaryClass', data=df, palette="tab10")
plt.title('binaryClasses', fontsize=15)


# In[124]:


plt.figure(figsize=(20,10))
df.value_counts('binaryClass').plot(kind="pie",autopct = '%1.1f%%')
plt.title ("Feature name : binaryClass", fontsize = 15)


# In[125]:


df.head()


# In[163]:


X = df.drop('binaryClass', axis=1)


# In[164]:


y = df.binaryClass


# In[165]:


results=[]
# define imputer
imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
strategies = [str(i) for i in [1,3,5,7,9]]
for s in strategies:
 pipeline = Pipeline(steps=[('i', KNNImputer(n_neighbors=int(s))), ('m', LogisticRegression())])
 scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=2, n_jobs=-1)
 results.append(scores)
 print('n_neighbors= %s || accuracy (%.4f)' % (s , mean(scores)))


# In[166]:


knn_pipeline = Pipeline(steps=[
 ('imputer', KNNImputer(n_neighbors=3))])


# In[167]:


X_knn = knn_pipeline.fit_transform(X)


# In[168]:


# Resampling the minority class. The strategy can be changed as required.
smt = SMOTETomek(random_state=42,sampling_strategy='minority',n_jobs=-1)
# Fit the model to generate the data.
X_res, y_res = smt.fit_resample(X_knn, y)


# In[169]:


len(X), len(y), len(X_res), len(y_res)


# In[170]:


df_smote= pd.DataFrame(y_res)


# In[171]:


df_smote['binaryClass'].value_counts()


# In[172]:


def evaluate_models(X, y, models):
    '''
    This function takes in X, y and models dictionary as input
    It splits the data into Train Test split
    Iterates through the given model dictionary and evaluates the metrics
    Returns: Dataframe which contains report of all models metrics with cost
    '''
    # separate dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    models_list = []
    accuracy_list = []

    for i in range(len(list(models))):
        model = list(models.values())[i] # Iterating through each model
        model.fit(X_train, y_train) # Train model
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        # Training set performance
        model_train_accuracy, model_train_f1,model_train_precision,        model_train_recall,model_train_rocauc_score = evaluate_clf(y_train ,y_train_pred)
        # Test set performance
        model_test_accuracy,model_test_f1,model_test_precision,        model_test_recall,model_test_rocauc_score=evaluate_clf(y_test, y_test_pred)
        print(list(models.keys())[i])
        models_list.append(list(models.keys())[i])

        print('Model performance for Training set')
        print("- Accuracy: {:.4f}".format(model_train_accuracy))
        print('- F1 score: {:.4f}'.format(model_train_f1))
        print('- Precision: {:.4f}'.format(model_train_precision))
        print('- Recall: {:.4f}'.format(model_train_recall))
        print('- Roc Auc Score: {:.4f}'.format(model_train_rocauc_score))
        print('----------------------------------')
        print('Model performance for Test set')
        print('- Accuracy: {:.4f}'.format(model_test_accuracy))
        print('- F1 score: {:.4f}'.format(model_test_f1))
        print('- Precision: {:.4f}'.format(model_test_precision))
        print('- Recall: {:.4f}'.format(model_test_recall))
        print('- Roc Auc Score: {:.4f}'.format(model_test_rocauc_score))
        print('='*35)
        print('\n')

        report = pd.DataFrame(list(zip(models_list, accuracy_list)), columns=['Model Name', 'Accuracy_Score']).sort_values(by=["Accuracy_Score"])

    return report


# In[173]:


# Function to evaluate model using metrics
def evaluate_clf(true, predicted):
 '''
 This function takes in true values and predicted values
 Returns: Accuracy, F1-Score, Precision, Recall, Roc-auc Score
 '''
 acc = accuracy_score(true, predicted) # Calculate Accuracy
 f1 = f1_score(true, predicted) # Calculate F1-score
 precision = precision_score(true, predicted) # Calculate Precision
 recall = recall_score(true, predicted) # Calculate Recall
 roc_auc = roc_auc_score(true, predicted) #Calculate Roc
 return acc, f1 , precision, recall, roc_auc


# In[174]:


# Dictionary which contains models for experiment
models = {
 "Random Forest": RandomForestClassifier(),
 "Gradient Boosting": GradientBoostingClassifier(),
 "XGBClassifier": XGBClassifier(),
 "CatBoosting Classifier": CatBoostClassifier(verbose=False),
 "AdaBoost Classifier": AdaBoostClassifier()
}


# In[175]:


# Training the models and getting report
report_knn = evaluate_models(X_res, y_res, models)


# In[176]:


# Fit the Simple imputer with strategy median
median_pipeline = Pipeline(steps=[
 ('imputer', SimpleImputer(strategy='median'))
])


# In[177]:


# Fit X with median_pipeline
X_median = median_pipeline.fit_transform(X)


# In[178]:


# Resampling the minority class. The strategy can be changed as required.
smt = SMOTETomek(random_state=42,sampling_strategy='minority')
# Fit the model to generate the data.
X_res, y_res = smt.fit_resample(X_median, y)


# In[179]:


# Training the models and getting report
report_median = evaluate_models(X_res, y_res, models)


# In[ ]:





# In[180]:


# Create a pipeline with simple imputer with strategy constant and fill value 0
constant_pipeline = Pipeline(steps=[
 ('Imputer', SimpleImputer(strategy='constant', fill_value=0))
])


# In[181]:


X_const = constant_pipeline.fit_transform(X)


# In[182]:


# Resampling the minority class. The strategy can be changed as required.
smt = SMOTETomek(random_state=42,sampling_strategy='minority', n_jobs=-1 )
# Fit the model to generate the data.
X_res, y_res = smt.fit_resample(X_const, y)


# In[183]:


# training the models
report_const = evaluate_models(X_res, y_res, models)


# In[184]:


# Create a pipeline with Simple imputer with strategy mean
mean_pipeline = Pipeline(steps=[
 ('Imputer', SimpleImputer(strategy='mean'))
])


# In[185]:


X_mean = mean_pipeline.fit_transform(X)


# In[186]:


# Resampling the minority class. The strategy can be changed as required.
smt = SMOTETomek(random_state=42,sampling_strategy='minority' , n_jobs=-1)
# Fit the model to generate the data.
X_res, y_res = smt.fit_resample(X_mean, y)


# In[187]:


# Training the models and getting report
report_mean = evaluate_models(X_res, y_res, models)


# In[188]:


pca_pipeline = Pipeline(steps=[
 ('Imputer', SimpleImputer(strategy='constant', fill_value=0))
])


# In[189]:


X_pca = pca_pipeline.fit_transform(X)


# In[190]:


#Applying PCA
from sklearn.decomposition import PCA
var_ratio={}
for n in range(2,27):
 pc=PCA(n_components=n)
 df_pca=pc.fit(X_pca)
 var_ratio[n]=sum(df_pca.explained_variance_ratio_)


# In[191]:


# plotting variance ratio
pd.Series(var_ratio).plot()


# In[156]:


pip install kneed


# In[157]:


from kneed import KneeLocator
i = np.arange(len(var_ratio))
variance_ratio= list(var_ratio.values())
components= list(var_ratio.keys())
knee = KneeLocator(i, variance_ratio, S=1, curve='concave', interp_method='polynomial')
fig = plt.figure(figsize=(5, 5))
knee.plot_knee()
plt.xlabel("Points")
plt.ylabel("Distance")
plt.show()
k= components[knee.knee]
print('Knee Locator k =', k)


# In[158]:


# Reducing the dimensions of the data
pca_final=PCA(n_components=7, random_state=42).fit(X_res)
reduced_X =pca_final.fit_transform(X_pca)


# In[159]:


# Resampling the minority class. The strategy can be changed as required.
smt = SMOTETomek(random_state=42,sampling_strategy='minority', n_jobs=-1)
# Fit the model to generate the data.
X_res, y_res = smt.fit_resample(reduced_X, y)


# In[160]:


# Training all models
report_pca = evaluate_models(X_res,y_res, models)


# In[193]:


pip install prettytable


# In[194]:




from prettytable import PrettyTable
pt=PrettyTable()
pt.field_names=["Model","Imputation_method","Model_Score"]
pt.add_row(["Random Forest","KNN imputer","99.85%"])
pt.add_row(["XGBClassifier","KNN imputer","99.85%"])
pt.add_row(["Random Forest","Simple Imputer(median)","99.85%"])
pt.add_row(["XGBClassifier","Simple Imputer(median)","99.85%"])
pt.add_row(["Random Forest","Simple Imputer(Constant)","99.85%"])
pt.add_row(["XGBClassifier","Simple Imputer(Constant)","99.85%"])
pt.add_row(["Random Forest","Simple Imputer(Mean)","99.85%"])
pt.add_row(["XGBClassifier","Simple Imputer(Mean)","99.85%"])
pt.add_row(["Random Forest","Principle component(SimpleImputer-median)","97.93%"])
pt.add_row(["XGBClassifier","Principle component(SimpleImputer-median)","98.88%"])
print(pt)


# In[195]:


final_model = XGBClassifier()
# Resampling the minority class. The strategy can be changed as required.
smt = SMOTETomek(random_state=42,sampling_strategy='minority', n_jobs=-1)
# Fit the model to generate the data.
X_res, y_res = smt.fit_resample(X_knn, y)


# In[196]:


X_train, X_test, y_train, y_test = train_test_split(X_res,y_res,test_size=0.2,random_state=42)
final_model = final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)


# In[197]:


print("Final XGBoost Classifier Accuracy Score (Train) :", round(final_model.score(X_train,y_train)*100,2))
print("Final XGBoost Classifier Accuracy Score (Test) :", round(accuracy_score(y_pred,y_test)*100,2))


# In[198]:


from sklearn.metrics import plot_confusion_matrix
#plots Confusion matrix
plot_confusion_matrix(final_model, X_test, y_test, cmap='Blues', values_format='d')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




