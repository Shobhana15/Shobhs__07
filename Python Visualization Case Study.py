#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[16]:


python_data = pd.read_csv("C:\\Users\ASUS\Desktop\Analytix labs\ALABS_Python_CaseStudy\Python Foundation Case Study 4_Python Visualization Case Study\SalesData.csv")
python_data.head()


# In[17]:


python_data.count()


# In[18]:


python_data.duplicated().sum()


# In[19]:


python_data.info()


# In[ ]:





# ### 1. Compare Sales by region for 2016 with 2015 using bar chart

# In[30]:


data1 = python_data.groupby(['Region'])['Sales2015','Sales2016'].sum()
data1


# In[38]:


plt.figure(figsize=(6,5))
data1.plot(kind = 'bar');
plt.xlabel('Region')
plt.ylabel('Sales year-wise')
plt.show()


# In[ ]:





# ### 2. What are the contributing factors to the sales for each region in 2016. Visualize it using a Pie Chart.

# In[ ]:





# ### 3. Compare the total sales of 2015 and 2016 with respect to Region and Tiers

# In[ ]:





# ### 4. In East region, which state registered a decline in 2016 as compared to 2015?
# 

# In[ ]:





# ### 5. In all the High tier, which Division saw a decline in number of units sold in 2016 compared to 2015?
# 

# In[ ]:





# ### 6. Create a new column Qtr using numpy.where() or any suitable utility in the imported dataset. The Quarters are based on months and defined as -
# <br>
# • Jan - Mar : Q1
# • Apr - Jun : Q2
# • Jul - Sep : Q3
# • Oct - Dec : Q4 
# 

# In[ ]:





# ### 7. Compare Qtr wise sales in 2015 and 2016 in a bar plot
# 

# In[ ]:





# ### 8. Determine the composition of Qtr wise sales in and 2016 with regards to all the Tiers in a pie chart.

# In[ ]:




