#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc


# In[5]:


df = pd.read_csv('titanic.csv')
df.head()


# In[6]:


df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)


# In[8]:


le = LabelEncoder()

df['Sex'] = le.fit_transform(df['Sex'])        # male=1, female=0
df['Embarked'] = le.fit_transform(df['Embarked'])


# In[9]:


X = df.drop('Survived', axis=1)
y = df['Survived']


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[11]:


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[12]:


ann_model = Sequential()

ann_model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
ann_model.add(Dropout(0.3))

ann_model.add(Dense(32, activation='relu'))
ann_model.add(Dense(1, activation='sigmoid'))

ann_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

ann_model.summary()


# In[16]:


history_ann = ann_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2
)


# In[20]:


loss_ann, acc_ann = ann_model.evaluate(X_test, y_test)
print("ANN Accuracy:", acc_ann)
print("ANN Accuracy in percent:", acc_ann*100,"%")


# In[21]:


X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


# In[22]:


cnn_model = Sequential()
cnn_model.add(Conv1D(32, kernel_size=2, activation='relu',
                     input_shape=(X_train.shape[1], 1)))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Flatten())
cnn_model.add(Dense(32, activation='relu'))
cnn_model.add(Dense(1, activation='sigmoid'))
cnn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
cnn_model.summary()


# In[24]:


history_cnn = cnn_model.fit(
    X_train_cnn, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)


# In[26]:


loss_cnn, acc_cnn = cnn_model.evaluate(X_test_cnn, y_test)
print("CNN Accuracy:", acc_cnn)
print("CNN Accuracy in percentage:", acc_cnn*100, "%")


# In[27]:


plt.plot(history_ann.history['val_accuracy'], label='ANN')
plt.plot(history_cnn.history['val_accuracy'], label='CNN')
plt.legend()
plt.title("Model Comparison")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.show()


# In[35]:


plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="hot")
plt.title("Feature Correlation Heatmap")
plt.show()


# In[34]:


y_pred_ann = (ann_model.predict(X_test) > 0.5).astype(int)
cm_ann = confusion_matrix(y_test, y_pred_ann)
plt.figure(figsize=(6,5))
sns.heatmap(cm_ann, annot=True, fmt='d', cmap="viridis")
plt.title("ANN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[33]:


y_pred_cnn = (cnn_model.predict(X_test_cnn) > 0.5).astype(int)
cm_cnn = confusion_matrix(y_test, y_pred_cnn)
plt.figure(figsize=(6,5))
sns.heatmap(cm_cnn, annot=True, fmt='d')
plt.title("CNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[36]:


plt.plot(history_ann.history['accuracy'], label='ANN Train')
plt.plot(history_ann.history['val_accuracy'], label='ANN Val')
plt.plot(history_cnn.history['accuracy'], label='CNN Train')
plt.plot(history_cnn.history['val_accuracy'], label='CNN Val')
plt.title("Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[37]:


plt.plot(history_ann.history['loss'], label='ANN Train')
plt.plot(history_ann.history['val_loss'], label='ANN Val')
plt.plot(history_cnn.history['loss'], label='CNN Train')
plt.plot(history_cnn.history['val_loss'], label='CNN Val')
plt.title("Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[40]:


y_prob_ann = ann_model.predict(X_test)
fpr_ann, tpr_ann, _ = roc_curve(y_test, y_prob_ann)
roc_auc_ann = auc(fpr_ann, tpr_ann)
plt.plot(fpr_ann, tpr_ann, label=f'ANN AUC = {roc_auc_ann:.2f}')
y_prob_cnn = cnn_model.predict(X_test_cnn)
fpr_cnn, tpr_cnn, _ = roc_curve(y_test, y_prob_cnn)
roc_auc_cnn = auc(fpr_cnn, tpr_cnn)
plt.plot(fpr_cnn, tpr_cnn, label=f'CNN AUC = {roc_auc_cnn:.2f}')
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# In[41]:


sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")
plt.show()
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival by Class")
plt.show()

