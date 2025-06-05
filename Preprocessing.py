
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
df=pd.read_csv("flightdata.csv")
print(df)

df1=df.head(5)
print("------------------------------")
print("Top 5 entries of the dataset")
print(df1)
print("------------------------------")


df2=df.info()
print("------------------------------")
print("Dataset Iformation")
print(df2)
print("------------------------------")


df3=df.shape
print("------------------------------")
print("shape of the dataset")
print(df3)
print("------------------------------")


df4=df.columns
print("------------------------------")
print("Columns in the dataset")
print(df4)
print("------------------------------")


df5=df.describe()
print("------------------------------")
print("describing the dataset")
print(df5)
print("------------------------------")


df6=(df.isnull().sum()/len(df))*100
print("------------------------------")
print("Check missing values in %")
print(df6)
print("------------------------------")

df7=df.isnull()
print("------------------------------")
print("Check missing values")
print(df7)
print("------------------------------")



df['ORIGIN'].value_counts().plot(kind='bar',title='origin')
plt.xlabel('Origin Points')
plt.ylabel('No of counts')
plt.show()

df['DEST'].value_counts().plot(kind='bar',title='dest')
plt.xlabel('Destination Points')
plt.ylabel('No of counts')
plt.show()

df['ORIGIN_AIRPORT_ID'].value_counts().plot(kind='bar',title='ORIGIN_AIRPORT_ID')
plt.xlabel('ORIGIN_AIRPORT_ID Points')
plt.ylabel('No of counts')
plt.show()

df['DEST_AIRPORT_ID'].value_counts().plot(kind='bar',title='DEST_AIRPORT_ID')
plt.xlabel('DEST_AIRPORT_ID')
plt.ylabel('No of counts')
plt.show()

df['DAY_OF_MONTH'].value_counts().plot(kind='bar',title='DAY_OF_MONTH')
plt.xlabel('DAY_OF_MONTH Points')
plt.ylabel('No of counts')
plt.show()

df['DAY_OF_WEEK'].value_counts().plot(kind='bar',title='DAY_OF_WEEK')
plt.xlabel('DAY_OF_WEEK Points')
plt.ylabel('No of counts')
plt.show()

df['DEP_TIME'].value_counts().plot(kind='bar',title='DEP_TIME')
plt.xlabel('DEP_TIME Points')
plt.ylabel('No of counts')
#plt.show()

df['ARR_TIME'].value_counts().plot(kind='bar',title='ARR_TIME')
plt.xlabel('ARR_TIME Points')
plt.ylabel('No of counts')
#plt.show()

df['CANCELLED'].value_counts().plot(kind='bar',title='CANCELLED')
plt.xlabel('CANCELLED Points')
plt.ylabel('No of counts')
plt.show()



pivot_df = df.pivot_table(values='ARR_DEL15', index='DAY_OF_MONTH', columns='MONTH')
plt.figure(figsize=(14, 8))
days = np.arange(1, 32)  
bar_width = 0.07  
for i, month in enumerate(pivot_df.columns):
    plt.bar(days + i * bar_width, pivot_df[month], width=bar_width, label=f'Month {month}')

plt.xlabel('Day of the Month')
plt.ylabel('Arrival Delay (minutes)')
plt.title('Arrival Delay vs Day of the Month for Different Months')
plt.xticks(days + bar_width * (len(pivot_df.columns) / 2), days)
plt.legend(title='Month')
plt.grid(True)
plt.tight_layout()
plt.show()



arr_delay_counts=df['ARR_DEL15'].value_counts()
plt.figure(figsize=(12,6))
arr_delay_counts.plot(kind='bar',color='skyblue')
plt.xlabel('Arrival Delay(minutes)')
plt.ylabel('Number of Occurrences')
plt.title('Distribution of Arrival Delays')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

cnc_delay_counts=df['CANCELLED'].value_counts()
plt.figure(figsize=(12,6))
cnc_delay_counts.plot(kind='bar',color='skyblue')
plt.xlabel('Arrival Delay(minutes)')
plt.ylabel('Number of Cancellation')
plt.title('Distribution of Cancellation')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



df['DEST'],_=pd.factorize(df['DEST'])
df['ORIGIN'],_=pd.factorize(df['ORIGIN'])
corr_matrix=df.corr()
plt.figure(figsize=(15,8))
sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

df=df.drop(columns=['YEAR'])
print(df)

df['ORIGIN_AIRPORT_ID'],_=pd.factorize(df['ORIGIN_AIRPORT_ID'])
df['DEST_AIRPORT_ID'],_=pd.factorize(df['DEST_AIRPORT_ID'])
corr_matrix=df.corr()
plt.figure(figsize=(15,8))
sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()





from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
import pandas as pd





label_encoder_origin = LabelEncoder()
label_encoder_dest = LabelEncoder()

perfect_df=pd.read_csv('filtered_file.csv')
perfect_df = perfect_df[['MONTH','DAY_OF_MONTH','DAY_OF_WEEK','CRS_DEP_TIME','DEP_TIME','ORIGIN','DEST','ARR_DEL15']] 
perfect_df['ORIGIN'] = label_encoder_origin.fit_transform(perfect_df['ORIGIN'])
perfect_df['DEST'] = label_encoder_dest.fit_transform(perfect_df['DEST'])
print("Class distribution before undersampling:")
print(perfect_df['ARR_DEL15'].value_counts())


majority_class = perfect_df[perfect_df['ARR_DEL15'] == 0]
minority_class = perfect_df[perfect_df['ARR_DEL15'] == 1]




majority_class_undersampled = resample(
    majority_class,
    replace=False,
    n_samples=len(minority_class),
    random_state=42
)


undersampled_df = pd.concat([majority_class_undersampled, minority_class])
undersampled_df = undersampled_df.sample(frac=1, random_state=42).reset_index(drop=True)


print("Class distribution after undersampling:")
print(undersampled_df['ARR_DEL15'].value_counts())




X = undersampled_df.drop('ARR_DEL15', axis=1)

y = undersampled_df['ARR_DEL15']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train the model using RF

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
y_pred = rf_model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# Train the model
from lightgbm import LGBMClassifier
lgb_model = LGBMClassifier(random_state=42, class_weight='balanced')
lgb_model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
y_pred = lgb_model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=lgb_model.classes_, yticklabels=lgb_model.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()






from xgboost import XGBClassifier

# Train XGBoost model
xgb_model = XGBClassifier(random_state=42, scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]))
xgb_model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
y_pred = xgb_model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=xgb_model.classes_, yticklabels=xgb_model.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()






# from catboost import CatBoostClassifier

# # Train CatBoost model
# catboost_model = CatBoostClassifier(random_state=42, class_weights=[len(y_train[y_train == 0]) / len(y_train[y_train == 1])], verbose=0)
# catboost_model.fit(X_train, y_train)


# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# y_pred = catboost_model.predict(X_test)
# print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
# print("Classification Report:")
# print(classification_report(y_test, y_pred))

# # Confusion Matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=catboost_model.classes_, yticklabels=catboost_model.classes_)
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Confusion Matrix')
# plt.show()






from sklearn.linear_model import LogisticRegression

# Train Logistic Regression model
log_reg_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
log_reg_model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
y_pred = log_reg_model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=log_reg_model.classes_, yticklabels=log_reg_model.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

