#!/usr/bin/env python
# coding: utf-8

# # Analyzing Consumer Trends in Cereal Consumption: A Demographic and Marketing Perspective📝
# 
# ![Banner](./assets/banner.jpeg)

# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
# 📝 <!-- Answer Below -->
# The projects goal is to examine the factors that affect cereal consumption patterns across groups in order to assist cereal producers and sellers in customizing their products and marketing approaches to suit consumer tastes better. Understanding these trends is essential, for identifying market openings and guiding public health efforts by taking into account the effects of cereal consumption. 

# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# 📝 <!-- Answer Below -->
# 1. What demographic factors such, as age groupings income levels and educational backgrounds impact the consumption of cereals? 
# 
# 2. What are the top cereal brands favored by segments? 
# 
# 3. What effect do marketing tactics, for cereals have on the decisions made by consumers? 
# 

# ## What would an answer look like?
# *What is your hypothesized answer to your question?*
# 📝 <!-- Answer Below -->
# 1. A detailed analysis of how various demographic groups consume cereal products and how these patterns are represented graphically using bar charts or pie charts to illustrate the relationship, between factors and consumption levels.
# 
# 2. A list showing the popularity of cereal brands, among age groups or income levels in a bar graph, with stacked bars has been compiled and ranked. 
# 
# 3. Insights, on the impact of marketing strategies like social media campaigns and TV advertisements on consumer purchasing behaviors can be demonstrated through a sequence of marketing events along with sales data, over time. 

# ## Data Sources
# *What 3 data sources have you identified for this project?*
# *How are you going to relate these datasets?*
# 📝 <!-- Answer Below -->
# 
# 1. Cereal Consumption Data (CSV):
# https://www.kaggle.com/datasets/mhansraj/cereal-consumption
# 
# 2. Cereal Popularity Data (CSV):
# https://www.kaggle.com/datasets/yangshun/cereal-popularity
# 
# 3. U.S. Census Demographic Data (API):
# https://api.census.gov/data.html
# 
# To relate the datasets effectively, I will first merge the cereal consumption data with the cereal popularity data based on common demographic attributes such as age, income, and education level. This will allow for a deeper understanding of how different demographic groups consume various cereal brands. Next, I will integrate the U.S. Census demographic data by matching identifiers like age group and income range to enrich the analysis of consumption patterns. Additionally, I will assess the influence of marketing efforts by correlating historical marketing expenditures with changes in consumption over time, providing insights into the factors driving cereal consumption trends across demographics.
# 
# 
# 
# 
# 
# 
# 
# 
# 

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->

# To address the project questions on cereal consumption trends, I will first prepare the data by loading and cleaning the datasets to ensure accuracy. Then, I will conduct exploratory data analysis (EDA) to identify patterns in cereal consumption and brand popularity across various demographics using visualizations like histograms and bar charts. Next, I will merge the cereal consumption data with the popularity data based on demographic attributes and integrate U.S. Census data for deeper analysis. I will perform statistical tests, including regression analysis, to determine significant relationships between demographic factors and cereal consumption, as well as assess the impact of marketing expenditures over time. Finally, I will create visual representations of the findings and summarize key insights to provide actionable recommendations for stakeholders in the food industry. This structured approach will effectively utilize the identified datasets to answer the project questions and yield valuable insights into cereal consumption trends.

# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->

# In[11]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

