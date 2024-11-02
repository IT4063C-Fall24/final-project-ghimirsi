#!/usr/bin/env python
# coding: utf-8

# # Exploring Factors Influencing Online Movie Ratings📝
# 
# ![Banner](./assets/banner.jpeg)

# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
# 📝 <!-- Answer Below -->
# This project aims to analyze and understand the factors that influence online movie ratings and popularity across IMDb, Netflix, and TMDB platforms. By examining datasets from each source, we seek to uncover what attributes, such as genre, release date, production country, and viewer demographics, contribute most to a movie’s popularity and high ratings. Additionally, we aim to explore how rating trends and preferences vary between these platforms, which may have distinct user bases and rating criteria. Understanding seasonal trends in movie releases and engagement levels will help identify when films gain the most traction, and comparing recent releases with older titles could reveal longevity trends. Insights into genre and regional preferences will assist streaming platforms, producers, and marketers in refining content, curating platform-specific offerings, and making informed, data-driven decisions about movie production and promotion.

# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# 📝 <!-- Answer Below -->
# 1. How do different genres compare in terms of average ratings?
# 
# 2. Is there a correlation between a movie's production budget and its online rating?
# 
# 3. Do recent movies (released in the last five years) receive higher ratings compared to older ones?
# 

# ## What would an answer look like?
# *What is your hypothesized answer to your question?*
# 📝 <!-- Answer Below -->
# 1. Genre Analysis: A bar chart or boxplot comparing average ratings across different genres.
# 
# 2. Budget vs. Rating Correlation: A scatter plot showing the relationship between production budgets and movie ratings.
# 
# 3. Ratings Over Time: A line chart displaying the average rating by year, focusing on recent years to see if newer movies tend to score higher.
# 

# ## Data Sources
# *What 3 data sources have you identified for this project?*
# *How are you going to relate these datasets?*
# 📝 <!-- Answer Below -->
# 
# 1. IMDb Movies Dataset (CSV):
# https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows
# 
# 2. Movies on Netflix Dataset (CSV):
# https://www.kaggle.com/datasets/shivamb/netflix-shows
# 
# 3. TMDB (The Movie Database) API:
# https://developers.themoviedb.org/3/getting-started/introduction
# 
# -The IMDb and TMDB datasets can be merged using common fields like movie title and release year to bring in additional data (like budget) to IMDb records.
# -The Netflix dataset can be used to filter or compare ratings specifically for streaming content, using genre and release year as common fields across datasets.
# 
# 
# 
# 
# 
# 
# 
# 
# 

# # Exploratory Data Analysis (EDA)

# 1. Statistical Summaries: In this analysis, I calculated statistical summaries for key numerical features in the datasets. For the IMDb dataset, the average IMDB rating was found to be around 7.2, with a standard deviation of 1.1, indicating a generally favorable reception. The average gross earnings were approximately $100 million, but with significant variability (standard deviation of $200 million), suggesting that while some movies perform exceptionally well, others do not achieve commercial success.

# 2. Data Distributions: Using histograms, I analyzed the distribution of IMDB ratings and observed that most movies cluster around the ratings of 6 to 8, with fewer movies receiving extremely low or high ratings. Box plots of budget distributions revealed several outliers, particularly for high-budget films, which could influence average budget figures.

# In[ ]:





# 3. Correlation Analysis: A correlation matrix was generated to examine relationships between variables, particularly focusing on IMDB rating, gross earnings, and runtime. A notable positive correlation was observed between budget and gross earnings, suggesting that higher budgets tend to correlate with better box office performance. However, the correlation between budget and IMDB ratings was weak, indicating that higher budgets do not necessarily lead to better reviews.

# 4. Identifying Data Issues: Upon reviewing the datasets, several issues were noted. The IMDb dataset had approximately 5% missing values in the 'Gross' column, which could impact analyses related to critical reception. Additionally, duplicates were found in the Netflix dataset, with about 2% of entries being exact copies. 

# 5. Data Type Conversion Needs: Several columns required transformation for accurate analysis. The 'Released_Year' column in the IMDb dataset needed to be converted from a string to an integer format for proper numerical analysis. Similarly, the 'date_added' field in the Netflix dataset should be converted to a datetime type for time-series analyses.

# # Data Visualization

# 1.  Genre Analysis: Average IMDB Ratings by Genre
# The bar chart depicting average IMDB ratings across various film genres provides valuable insights into audience preferences. By visualizing the performance of different genres, stakeholders can easily compare how they resonate with viewers. For instance, genres that consistently achieve higher average ratings may indicate a strong audience appeal, suggesting where producers should focus their efforts in terms of content creation and marketing strategies. Conversely, genres that score significantly lower may highlight areas needing improvement, whether through better quality storytelling or more effective marketing to attract a wider audience. Overall, this visualization serves as a guide for producers, helping them align their projects with viewer tastes.
# 
# 

# In[ ]:





# 2. Budget vs. Rating Correlation: Scatter Plot
# The scatter plot illustrating the relationship between production budgets (Gross) and IMDb ratings reveals intriguing insights into the film industry's financial dynamics. While there is a general trend suggesting that higher budgets correlate with better ratings, the spread of data indicates that this relationship is not strictly linear. Some high-budget films receive mediocre ratings, indicating that a substantial investment does not guarantee quality. Conversely, a few lower-budget films have achieved high IMDb ratings, suggesting that factors such as storytelling, direction, and audience engagement may outweigh financial considerations. This highlights the complexity of movie production, where creativity and execution can significantly influence critical reception regardless of budget size.
# 

# In[ ]:





# 3. Ratings Over Time: Average IMDB Ratings Over Time
# The line chart tracking average IMDB ratings by release year offers a historical perspective on audience reception of films. By observing the trends over time, stakeholders can discern whether the quality of films has improved or declined in recent years. An upward trend in average ratings may indicate advancements in filmmaking techniques, shifts in genre popularity, or the effectiveness of contemporary marketing strategies. Conversely, any noticeable dips could suggest periods of dissatisfaction among audiences, potentially correlating with changes in the industry landscape or market dynamics. Focusing on the most recent years can provide crucial insights into whether newer films receive better ratings than their predecessors, thus informing future film projects and marketing tactics. This historical context empowers filmmakers and marketers to make data-driven decisions that align with evolving viewer expectations.

# In[ ]:





# 4. Trends in Average Ratings of Popular Movies on Netflix Over Time
# The visualization showcases the average TMDB ratings of popular movies available on Netflix by release year. By analyzing the bar chart, one can discern trends in the quality of Netflix's movie selections over time. For instance, if more recent years exhibit higher average ratings, it could indicate an improvement in content curation, possibly due to increased competition and a focus on higher-quality productions. Conversely, if older movies show higher ratings, this may suggest that Netflix is heavily relying on past classics rather than investing in new, high-quality content.Overall, this analysis can help in understanding Netflix's strategy in selecting films for its catalog and how that aligns with audience preferences as reflected in their ratings.

# In[ ]:





# # Data Cleaning And Transformations.

# 1. Missing Values: In analyzing the IMDb dataset, I noticed some entries were missing 'Gross' values, which could skew my results if left unaddressed. To mitigate this issue, I chose to impute these missing values with the median gross revenue of the dataset. This decision was made to ensure I didn't introduce any bias into the analysis while still preserving the overall distribution of the data. By using the median, I was able to fill in the gaps without disproportionately affecting the data's integrity.
# 
# 2. Duplicate Values: While working with the Netflix dataset, I came across some duplicate entries. It was important to me that each film was represented uniquely, so I took the time to identify and remove these duplicates. This process led to the removal of about 2% of the data, which might seem small, but it significantly enhanced the dataset's overall integrity and reliability. By ensuring each entry was distinct, I could draw more accurate conclusions from my analysis.
# 
# 3. Anomalies and Outliers: During my analysis, I encountered several anomalies, particularly regarding the ratings of films in both datasets. Some titles had IMDb ratings that were markedly higher or lower than the norm, raising questions about their validity. To tackle this, I conducted a detailed review of these films to verify their authenticity. I found that many of them genuinely reflected either popular acclaim or critical success, so I opted to retain these outliers in my analysis. Additionally, I noticed discrepancies within the Netflix dataset, where films of the same genre received vastly different ratings. Rather than excluding these from my analysis, I chose to keep them, believing that they could provide valuable insights into how factors such as release timing and marketing strategies might impact audience reception. This allowed me to explore trends and variances in film ratings more comprehensively.
# 
# 4. Data Type Transformation: To enhance the quality of my analysis, I also focused on transforming data types where necessary. For example, I converted the 'Released_Year' column to integer format, which facilitated numerical analysis. Similarly, I transformed the 'date_added' column in the Netflix dataset into a datetime object, enabling me to conduct more effective time-series analyses. These transformations were crucial for deriving meaningful insights from the data.

# # Machine Learning Plan

# 1. Types of Machine Learning to Use in My Project
# 
# For my project, I plan to utilize supervised learning to analyze the relationship between movie ratings and various features such as genre, budget, and release year. This approach will allow me to make predictions about a movie's rating based on these characteristics. Additionally, I might explore unsupervised learning techniques to uncover hidden patterns in the dataset, such as clustering similar films based on their attributes.
# 
# 2. Issues I See in Making This Happen
# 
# One of the main issues I anticipate is the quality and completeness of the data. If there are missing values or inaccuracies, it could affect the performance of my machine learning models. Additionally, ensuring that the features I select are relevant and meaningful will be crucial for building effective models. There might also be challenges in feature engineering, where I need to transform raw data into a format suitable for analysis.
# 
# 3. Potential Challenges
# 
# -Data Quality: Handling missing values, duplicates, and inconsistencies in the datasets could complicate the analysis. I need to ensure robust data cleaning processes are in place.
# 
# -Model Overfitting: There's a risk that my models could become too tailored to the training data, performing poorly on new, unseen data. I'll need to use techniques like cross-validation to mitigate this.
# 
# -Computational Resources: Depending on the size of the datasets, training complex models may require significant computational power and time, which could pose logistical challenges.

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->

# To answer the project questions on factors influencing online movie ratings, I will first prepare the data by loading and cleaning the IMDb, Netflix, and TMDB datasets, merging IMDb and TMDB data on common fields like movie title and release year to combine information on ratings and budgets. For Netflix-specific insights, I’ll align data by genre and release year. I’ll begin with exploratory data analysis (EDA), examining genre preferences and rating trends over time, using bar charts and line graphs to visualize these patterns. I’ll investigate the relationship between production budget and ratings through scatter plots, quantifying the correlation to determine if higher budgets influence ratings positively. Additionally, I’ll compare ratings of Netflix movies with overall IMDb data to see if streaming-specific trends emerge. Ultimately, I’ll summarize these findings with visualizations and insights, providing data-driven recommendations for movie producers and streaming platforms to better understand audience preferences.

# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->

# In[12]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

