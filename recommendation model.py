#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import json
import csv
from urllib.request import urlopen
from itertools import zip_longest
import re #Regex
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


# In[2]:


#page_num = 1

url= "https://unlockassist.com/services/all"

r=requests.get(url)
src= r.content 
#soup = BeautifulSoup(src,'lxml')
#soup
soup =BeautifulSoup(src,'html.parser')
#print(soup.prettify())
#soup
#bs.prettify()


# In[3]:


company_names = soup.findAll('a',{"class":"title heading h-4"})
company_name = soup.findAll('a',{"class":"title heading h-4"})[0].text.strip().replace('.'," ")
len(company_names)
#len(company_names)
#company_names=company_names.replace('\n'," ")
#company_names[18].text.strip()


# In[4]:


industries =soup.findAll('div',{"class":"categories"})
industries = soup.findAll('div',{"class":"categories"})[0].text.strip().replace('\n',' ')
industries 
#len(industries)
#industry=print(industry.strip())
#industry
#industries[0].text.strip()
#len(industries)


# In[5]:


describtions = soup.findAll('p',{"class":"paragraph"})[1].text.strip()
describtions
#len(describtions)
#len(describtions)


# In[6]:


industry = soup.findAll('div',{"class":"categories"})
industry = soup.findAll('div',{"class":"categories"})[0].find('a').attrs['href'].replace('-in-egypt',' ').replace('/services',' ').replace('-companies',' ').strip()
industry


# In[7]:


#c_categories = []
c_names = []
c_describtions = []
c_industries =[]


for i in range(84):
    #c_categories.append(soup.findAll('a',{"class":"toggle-title"})[i].text.strip().replace('\n', " "))
    c_names.append(soup.findAll('a',{"class":"title heading h-4"})[i].text.strip().replace('.'," "))
    
    c_describtions.append(soup.findAll('p',{"class":"paragraph"})[i].text.strip().replace('\n'," "))
    
    c_industries.append(soup.findAll('div',{"class":"categories"})[i].find('a').attrs['href'].replace('-in-egypt',' ').replace('/services',' ').replace('-companies',' ').strip())


    #print(c_name,c_industry,c_describtion)
    
    
    print(c_names)
    #print(c_industries)
    print(c_describtions)
    print(c_industries)
    print()
    #print(str(c_names) + ", ", str(c_industries) + ", "+str(c_describtions))
    


# In[21]:


file_list = [c_names,c_describtions,c_industries]

exported = zip_longest(*file_list)

with open (r"C:\Users\Wello\Downloads\Final_unlock.csv",'w') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(['c_names','c_describtions','c_industries'])

    wr.writerows(exported)
    


# In[22]:


df = pd.read_csv(r"C:\Users\Wello\Downloads\Final_unlock.csv")
df


# In[23]:


df = df.replace(r'/',' ', regex=True)
df


# In[24]:


df.c_describtions.shift(-1)
df


# In[25]:


df['c_describtions']= df.c_describtions.shift(-1)


# In[26]:


df


# In[27]:


df['c_names']


# In[28]:


df['c_describtions']


# In[29]:


#df['c_industries'].head(20)


# In[30]:


from sklearn.feature_extraction.text import TfidfVectorizer
df['c_industries']=df['c_industries'].fillna(' ')

# Instantiate a TfidfVectorizer object
vectorizer = TfidfVectorizer()
# It fits the data and transform it as a vector
X = vectorizer.fit_transform(c_industries)
# Convert the X as transposed matrix
X = X.T.toarray()
# Create a DataFrame and set the vocabulary as the index
df = pd.DataFrame(X, index=vectorizer.get_feature_names())
df


# In[31]:


#cosine_sim=linear_kernel(X,X)
#cosine_sim


# In[32]:


# Reverse mapping of indices and Providers
indices = pd.Series(df.index, index=vectorizer.get_feature_names())
#indices['Zammit']
#print(indices.drop_duplicates())
#print(indices['Extreme Solution'])


# In[34]:


def get_similar_articles(q, df):
  print("query:", q)
  print("Berikut artikel dengan nilai cosine similarity tertinggi: ")
  # Convert the query become a vector
  q = [q]
  q_vec = vectorizer.transform(q).toarray().reshape(df.shape[0],)
  sim = {}
    
  # Calculate the similarity
  for i in range(1,84):
    sim[i] = np.dot(df.loc[:, i].values, q_vec) / np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vec)
    #print( sim[i])
  
  # Sort the values 
  sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)
  #print(sim_sorted)
  # Print the company_names and their similarity values
  for k, v in sim_sorted:
    if v !=0:
      print("Nilai Similaritas:", v)
      print(c_names[k])
      print()
# Add The Query
q1 = 'branding'
# Call the function
get_similar_articles(q1, df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




