# USD MADS Capstone Project - Team 14
## Predicting Cell Culture Viability using a Sequential Hybrid Model

# Table of Contents

--------

- [Authors and Collaborators](#authors-and-collaborators)
- [Installation](#installation)
- [Technologies](#technologies)
- [About the Project](#about-the-project)
  * [Problem Statement](#problem-statement)
  * [Justification](#justification)
- [Data Exploration](#data-exploration)
  * [Original Data](#original-data)
   - [Data Set Files](#data-set-files)
 - [Data Preparation](#data-preparation)
- [Model Training](#Model-training)
- [Model Strategies](#model-strategies)
- [Presentations and Reports](#presentations-and-reports)
- [Code](#code)
- [Conclusion](#conclusion)
- [References](#references)

-------

# Authors and Collaborators
- Hanmaro Song
- Emanuel Lucban


# Installation
To clone this repository onto your device, use the commands below:

	1. git init
	2. git clone git@github.com:https://github.com/zeegeeko/ADS509-Topic-Modeling

## Technologies

- Python
  - Pre-Trained BERT Model (large-uncased)
  - Scikit Learn
  - Gensim
  - Selenium (web scraping)
    - Chromium 
  - BeautifulSoup
- Jupyter Lab
- Amazon S3


# About the Project

This project's primary goal is to predict the ratings given to a film by the Rotten 
Tomatoes audience. We will be gathering data from EIDR (The Entertainment Identifier Registry Association) 
to pull a large number of industry standard, movie identifiers. The identifiers will then be used
to cross-reference against titles on Rotten Tomatoes, where we will employ web-scraping techniques
to gather each movie title's synopsis, audience scores, genres and other metadata. Our hope
is to be able to leverage text-mining techniques on the movie synopsis data in order to gain 
insights and identify the potential relationships between keywords, topics, semantics that may
exist in a given movie's premise gleaned from its synopsis. From the results of text-mining, 
we can then use the salient features to train a classification model to approximate the Rotten Tomatoes' audience
response in the form a 1-5 star rating. Accurate predictions can aid movie studios, producers, directors, writers,
etc. in aligning business goals with audience expectations.

## Problem Statement

The global film industry is a market worth well over $100 billion according to the 
Motion Picture Association report in 2019. Global revenue from box-office sales totaling $42.2 billion
and streaming, home/mobile entertainment at $58.8 billion. Despite COViD-19 dropping in-theater
sales by 82%, the growth of streaming service amounted to a 37% increase in 2020 and still 
expected a compounded annual growth rate of 21% from 2021 to 2028. 

With steady growth of consumers of video entertainment, in both theaters and now on-demand 
streaming services, steady delivery of quality content is vital to the industry's continued
success. With the cost of producing a movie or show reaching from tens to hundreds of millions
of dollars, it is important for studios to maximize the returns on such large investments. Therefore
it is imperative that studios, producers, directors, writers, etc. can properly predict the
response of their audience. Sites like Rotten Tomatoes and Metacritic have become the de-facto
standard in movie and show ratings, providing scores from both professional critics and the
general audience. 

With Rotten Tomatoes having over 30 million unique visitors a month, they
are well positioned to influence viewership. The goal of this project is to provide an easy-to-use
tool that can predict the ratings of a Rotten Tomatoes audience based on the synopsis of a 
movie or show. This tool can be leveraged by studios, producers, writers, etc. to make adjustments
to their content to better suit their audience or modulate expectations to improve content quality, increase
viewership and maximize return on investments.

**Sources**  
Rosal, M.-L. (2022). U.S. film industry statistics [2022]: Facts about the U.S. film industry. Zippia. Retrieved from https://www.zippia.com/advice/us-film-industry-statistics/   

Escandon, R. (2020, March 13). The film industry made a record-breaking $100 billion last year. Forbes. Retrieved from https://www.forbes.com/sites/rosaescandon/2020/03/12/the-film-industry-made-a-record-breaking-100-billion-last-year/?sh=311a6cf234cd 

## Justification

Ratings from sites such as Rotten Tomatoes and Metacritic are important indicators for the
success or failure of a movie or show. With over 30 million unique visitors a month, these sites
have tremendous influence and having the ability to predict the response of these audiences to 
a movie or show before release, can provide an opportunity for studios, producers, directors, writters, etc.
to improve their content.

# Data Exploration


We scraped 84,219 movie titles from EIDR but only managed to scrape 17,313 movie data from 
Rotten Tomatoes. Rotten Tomatoes data is missing 202 synopsis entries and the number of rows 
missing for audience scores is 2,303. There are 15 column totals, 2 index columns, 2 date columns, 
3 numerical columns. Genre column has 23 unique values but will be converted to 7 distinct genres 
and is nominal valued. Rating is missing 12,380 values and will most likely be dropped. Synopsis 
is text data that will be the primary focus of our text-mining efforts. There are total of 17,050 fully 
populated rows and 16,315 of those are in English language.


## Original Data 

EIDR source: https://ui.eidr.org/search  
Rotten Tomatoes: http://www.rottentomatoes.com


## Data Set Files

The dataset is in two files one for the EIDR listings and one for the Rotten Tomatoes
movie data scrape. Both files are in CSV format and stored on Amazone S3.

**EIDR Scrape**  
https://usd-mads-projects.s3.us-west-1.amazonaws.com/ADS509/EIDR+Data.csv

**Rotten Tomatoes Scrape**  
https://usd-mads-projects.s3.us-west-1.amazonaws.com/ADS509/RT+Data.csv  

# Data Preparation

The main focus of this project is to develop a model to predict ratings based on the title's synopsis, so 
the synopsis data will be the most important part of the dataset. Other features such as 
genre, runtime will also be used against the score_audience ratings. 

Since we wanted to generate more "ball-park" ratings predictions, we decided to bin the audience
scores of 0-100 into 5 bins which allows us to turn this into a multi-class classification problem.
We binned the 0-100 Rotten Tomatoes audience scores as followed.  

- 1 Star: ~ 20%
- 2 Star: 21% - 40%
- 3 Star: 41% - 60%
- 4 Star: 61% - 80%
- 5 Star: 81% - 100%

Release dates for the data are not to be used as features since the tool will be designed to 
predict audience scores (by Stars) of unreleased or titles still in production. However, the 
theatrical release dates and the streaming release dates were converted from dates to number of 
days since released for data analysis purposes only. 

Since we require both audience scores (target) and synopsis data, we removed all rows that had either
null scores or null synopses. We decided againt score imputation in order to prevent any form
of bias to be introduced. After cleaning the dataset was reduced to 17,050 rows.

We found that the 27 genres from the Rotten Tomatoes data was not distinct enough and can be 
condensed. Genres that appeared less than 100 times such as fitness, anime and stand-up was combined
into a genre called other. After condensing we ended up with 11 distinct genres and then one-hot-encoded.

For the synopsis data, we created 2 forms of preprocessing, lemmatized_tokens and bert_tokens. This 
was done due to the differences in processing requirements for Topic Modeling and encoding using the
Bi-directional Encoders Representations from Transformers (BERT) model. For topic modeling, we stripped
punctuation, lower-cased, removed stop-words and lemmetized or stemmed words. For BERT, since 
contextual information is encoded, we performed very minimal processing, lower-casing and tokenizing.


# Model Training

The data outlined in the Data Preparation step is only a preliminary step before classification models can be trained.
Other models are used in an unsupervised manner for text-mining are utilized in order to generate
salient features to be used for the classification model.

**Topic Modeling**  
Latent Dirichlet Allocation (LDA) is a unsupervised technique for generating topics. The discovered
topics will be used to generate topic probability distributions for each movie synopsis in our dataset as a 
feature. The LDA model was able to identify 7 optimal topics based on coherence scores. 

**BERT Model**  
We wanted to also encode contextual information from the synopsis data to use as a feature. The 
pretrained BERT (large, uncased) model was used to encode each synopsis into a 768 length vector
embedding that will be used as features for the classification modeling.

**Classification Models**  
The classification model will perform the actual rating prediction. The training set for the 
classification model includes the 768 column BERT embedding, 7 topic probabilities, 11 genres and runtime. The target
is the binned audience score of 1-5 stars. The dataset was then split into training and validation sets using
a 75/25 split. Since there is a large class imbalance with 5 stars only representing less than 3% of the dataset, we
utilized SMOTE to evenly balance all the classes.

The classification models that were trained and tuned are:
- Linear Discriminant Analysis
- Quadratic Discriminant Analysis
- Logistic Regression (Multinomial)
- Random Forest
- K-Nearest Neighbors
- Support Vector Machines

# Model Strategies 

All models were trained using 5-fold cross validation and models with hyper-parameters were
trained and evaluated across various parameters for tuning. The tunings with the highest testing
accuracy were then selected as candidate models for further evaluation

Ideally we want to ensure that all ratings classes (1-5 Stars) are correctly classified. We decided that
overall classification accuracy and individual F1 scores will be taken into account for final model
selection criteria.

# Conclusion

Based on overall accuracy and individual class F1 scores, the selection was very narrow between
the tuned Random Forest model and tuned Support Vector Machines. Although, the Support Vector Machines was
able to slightly better predict 5-Star ratings, we opted for the Random Forest Model which had
a overall higher accuracy of 39% since 5 Star movies are very rare and Random Forest was able
to better predict overall.  

# Working Proof of Concept (Flask App)

We have developed a working inference pipeline using the selected classification model (Random Forest) and
developed a working Flask App that will take in a synopsis, genres and runtime as input and return
probabilities for each Star rating (highest probability is the predicted rating).

The working proof of concept, Flask App is available for use here:  
http://b52a-104-175-34-167.ngrok.io 

Please use http instead of https since we do not yet have an SSL certificate

# Next Steps  
  
We would like to further refine this project as time allows by:
- Identifying new features
- Introduce non-linear dimensionality reduction for better generalization
  - T-SNE
  - UMAP
- Test more classification models
  - Gradient Boosting Machines
  - XGBoost
  - Neural Networks

# Code
https://github.com/zeegeeko/ADS509-Topic-Modeling





