#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[17]:


titanic_data=pd.read_csv(r'D:\aidata.csv')
titanic_data.head(5)


# In[18]:


print("# of country in orginal data :" +str(len(titanic_data.index)))


# ## Analyzing data

# In[19]:


sns.countplot(x="sex_19",data=titanic_data)


# In[20]:


titanic_data["ap(2019)"].plot.hist()


# In[21]:


titanic_data.info()


# ## Data Wrangling

# In[22]:


titanic_data.isnull()


# In[23]:


titanic_data.isnull().sum()


# In[24]:


titanic_data.drop("capital",axis=1, inplace=True)


# In[25]:


titanic_data.head(5)


# In[ ]:





# In[26]:


sex_19=pd.get_dummies(titanic_data["sex_19"],drop_first=True)


# In[27]:


sex_19.head(5)


# In[28]:


sex_09=pd.get_dummies(titanic_data["sex_09"],drop_first=True)


# In[29]:


sex_09.head(5)


# In[30]:


name=pd.get_dummies(titanic_data["name"],drop_first=True)


# In[31]:


name.head(5)


# In[32]:


titanic_data=pd.concat([titanic_data,sex_19,sex_09,name],axis=1)


# In[33]:


titanic_data.head(5)


# In[34]:


titanic_data.drop(['sex_19','sex_09','name'],axis=1,inplace=True)


# In[35]:


titanic_data.head()


# ### Train data

# In[36]:


x=titanic_data.drop("condition",axis=1)
y=titanic_data["condition"]


# In[37]:


from sklearn.model_selection import train_test_split


# In[38]:


x_train, x_test, y_train, y_test = train_test_split(
x, y, test_size=0.2, random_state=10)


# In[39]:


from sklearn.linear_model import LogisticRegression


# In[40]:


logmodel=LogisticRegression()


# In[41]:


logmodel.fit(x_train,y_train)


# In[42]:


predictions = logmodel.predict(x_test)


# In[43]:


from sklearn.metrics import classification_report


# In[44]:


classification_report(y_test,predictions)


# In[45]:


from sklearn.metrics import confusion_matrix


# In[46]:


confusion_matrix(y_test,predictions)


# In[47]:


from sklearn.metrics import accuracy_score


# In[48]:


accuracy_score(y_test,predictions)


# In[49]:


logmodel.score(x_test,y_test)


# In[50]:


from sklearn import svm


# In[51]:


logmodel = svm.SVC()


# In[52]:


logmodel.fit(x_train, y_train)


# In[53]:


predictions = logmodel.predict(x_test)


# In[54]:


from sklearn.metrics import classification_report


# In[55]:


classification_report(y_test,predictions)


# In[56]:


from sklearn.metrics import confusion_matrix


# In[57]:


confusion_matrix(y_test,predictions)


# In[58]:


from sklearn.metrics import accuracy_score


# In[59]:


accuracy_score(y_test,predictions)


# In[60]:


logmodel.score(x_test,y_test)


# In[61]:


from sklearn.naive_bayes import GaussianNB


# In[62]:


logmodel = GaussianNB()


# In[63]:


logmodel.fit(x_train, y_train)


# In[64]:


predictions = logmodel.predict(x_test)


# In[65]:


from sklearn.metrics import classification_report


# In[66]:


classification_report(y_test,predictions)


# In[67]:


from sklearn.metrics import confusion_matrix


# In[68]:


confusion_matrix(y_test,predictions)


# In[69]:


from sklearn.metrics import accuracy_score


# In[70]:


accuracy_score(y_test,predictions)


# In[71]:


logmodel.score(x_test,y_test)


# In[72]:


from sklearn.ensemble import AdaBoostClassifier


# In[73]:


from sklearn.datasets import make_classification


# In[74]:


x_train, y_train = make_classification(n_samples=1000, n_features=4,
n_informative=2, n_redundant=0,
random_state=0, shuffle=False)


# In[75]:


clf = AdaBoostClassifier(n_estimators=100, random_state=0)


# In[76]:


clf.fit(x_train, y_train)


# In[77]:


clf.score(x_train, y_train)


# In[78]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[79]:


clf = LinearDiscriminantAnalysis()


# In[80]:


clf.fit(x_train, y_train)


# In[81]:


clf.score(x_train, y_train)


# In[82]:


from sklearn import tree


# In[83]:


dest = tree.DecisionTreeClassifier()


# In[84]:


dest = dest.fit(x_train,y_train)


# In[85]:


dest.score(x_train, y_train)


# In[86]:


from sklearn.ensemble import GradientBoostingClassifier


# In[87]:


gb = GradientBoostingClassifier(n_estimators=20, learning_rate = 0.75, max_features=2, max_depth = 2, random_state = 0)


# In[88]:


gb.fit(x_train, y_train)


# In[89]:


gb.score(x_train, y_train)


# In[90]:


from sklearn.neural_network import MLPClassifier


# In[91]:


anna = MLPClassifier(solver='lbfgs', alpha=1e-5,
hidden_layer_sizes=(5, 2), random_state=1)


# In[92]:


anna.fit(x_train, y_train)


# In[93]:


anna.score(x_train, y_train)


# In[115]:


from matplotlib import pyplot as plt 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[116]:


plt.plot(x_train,y_train)


# In[ ]:





# In[117]:


plt.plot(x_test,y_test)


# In[ ]:





# In[118]:


a = ['SVM','NB','ADA','LDA','LR','DT','GB','ANN']


# In[119]:


b = [87,37,57,98,93,100,98,96]


# In[120]:


Accurecy=[]


# In[ ]:





# In[ ]:





# In[121]:


plt.plot(b,a)
plt.ylabel("Algorithm")
plt.xlabel("Accuracy %")
plt.title("Classifier Accuracy")
plt.show()


# In[122]:


plt.bar(a,b)
plt.ylabel("Accurecy")
plt.xlabel("Algorithms")

plt.title("Classifier Accuracy")
plt.show()


# In[123]:


plt.scatter(b,a, label='skitscat',color='k',s=25,marker='o')
plt.ylabel("Algorithm")
plt.xlabel("Accuracy %")
plt.title("Classifier Accuracy")
plt.show()


# In[126]:


c=['Bangladesh','India','Pakistan','Bhutan','Nepal','Srilanka','China','Japan','Malyasia','England','Spain','France','Germany','Denmark','Portugal','Poland','Finland','Italy','Sweden','Belgium','Greece','Hungary','Austria','Norway','Croatia','Lithuania','Solvenia','Estonia','Iceland','Russia','USA','Argentina','Canada','Australia','Brazil','Colombia','Uruguay','Chile','New Zealand']


# In[127]:


d=y


# In[ ]:





# In[128]:


plt.bar(d,c)
plt.ylabel("Country")
plt.xlabel("Condition")
plt.title("Condotion of Countries")
plt.show()


# In[129]:


e=y_test


# In[130]:


y_test


# In[131]:


f=['Argentina','Canada','Bangladesh','Spain','Sri Lanka','Bhutan','Iceland','India']


# In[132]:


plt.bar(e,f)
plt.ylabel("Test data Country")
plt.xlabel("Condition")
plt.title("Condotion of some Countries")
plt.show()


# In[133]:


plt.plot(e,f)
plt.ylabel("Country")
plt.xlabel("Condition")
plt.title("Condotion of Countries")
plt.show()


# In[134]:


plt.scatter(e,f, label='skitscat',color='k',s=25,marker='o')
plt.ylabel("Algorithm")
plt.xlabel("Accuracy %")
plt.title("Classifier Accuracy")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[135]:


print="the end"


# In[233]:


print


# In[ ]:




