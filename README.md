# Twitter_Sentiment_Analysis

# METHODOLOGY
Data collection, data preprocessing, method selection, classification model selection, and
preparation for selected classification methods are the five parts of the methodology section.
Each subsection describes the methodologies and techniques used to evaluate tweets about
artificial intelligence and emerging technologies.
# Data Collection
The Twitter API is used to query tweets containing predetermined keywords and hashtags related
to AI and emerging technologies to obtain relevant tweets. Tweets covering various emerging
technologies between 2014 and 2022 were used as the dataset. Artificial Intelligence, Machine
Learning, the Internet of Things, 3D Printing, Additive Manufacturing, Industry 4.0, Augmented
Reality, Bioengineering, and Renewable Energy Technologies were these predetermined
keywords. The Twitter query string that was used to obtain the dataset is as follows:
querystring='(("Artificial Intelligence" OR "AI" OR "Machine Learning") OR ("Internet of
Things" OR "IoT") OR ("3D Printing" OR "Additive Manufacturing") OR ("Advanced
Manufacturing" OR "Industry 4.0") OR ("Augmented Reality" OR "AR") OR ("Virtual Reality"
OR "VR") OR ("Blockchain Technology" OR "Cryptocurrency") OR ("Big Data" OR "Data
Analytics") OR ("Biotechnology" OR "Bioengineering") OR ("Renewable Energy Technologies"
OR "Clean Energy")) (place_country:AU) (-is:retweet) (-is:reply) (lang:en)'
This data set consists of 56245 rows in total. 48907 of these lines contain information about
tweets from the country of Australia. The remaining 7338 include information about tweets from
New Zealand. All tweets in the dataset are in English. Therefore, there was no need to separate
tweets by language.
On the other hand, a supervised data set is used for sentiment analysis in this study. The dataset is
gathered from the Kaggle platform, specifically from a publication of Kaggle user Yasser H., and
contains sentiment analysis scores connected with tweets (Yasser, 2022). Kaggle provides a huge
number of datasets as well as a cloud-based data science environment, thus a reliable source for
model training on tweets sentiment analysis (Quaranta et al., 2021).
The dataset from Kaggle is used for data processing and model training. The dataset is split into
training and testing datasets, the testing dataset was for testing the resulting model trained by
using the training dataset. The training data are preprocessed for use in machine learning.
# Data Preprocessing
Unstructured text data makes up a large portion of social media data. Text data regarded as
unstructured includes non-significant expressions. For this reason, data preprocessing is
necessary to clean up this "dirty" data for NLP to function effectively. Both the datasets provided
by Kaggle and received using the Twitter API underwent these data preparation processes.
Making tweets cleaner with data preprocessing includes the following steps (Ağralı et al., 2022).
- Noise removal in Tweets
- Tokenization
- Stemming
- Lemmatization
- Joining text again
4.2.1. Noise Removal in Tweets
Unnecessary noise that will complicate the sentiment analysis should be removed to gather more
reliable results using the trained model (Hemalatha et al., 2012).
- Removing URLs
- Removing special characters
- Removing hashtags
- Removing emojis
- Removing stop words
Unnecessary characters, words, and concepts make it difficult to get to the root of the word or the
sentence. Special characters such as hashtags, URLs, emojis, and others should be removed as the
sentiment analysis process is word-based. Stop words should also be removed because they have
no benefits for sentiment analysis (Hemalatha et al., 2012).
# Tokenization
Tokenization is the next step after removing unnecessary characters. And this method is used to
sort sentences into words in a way that can be analyzed by the model. Tokenization allows for the
subsequent phases to identify each word's root. A tokenized word can also be called "token"
(Wang et al., 2021).
# Stemming
After breaking the sentence into words, it is necessary to understand the root of each token. At
the stemming stage, unnecessary suffixes, such as plural suffixes and verb conjugation suffixes,
are removed from the tokens (Wang et al., 2021).
# Lemmatization
Lemmatization is a necessary stage of data preprocessing because stemming sometimes has
trouble getting root words. An alternative is a lemmatization, which takes into consideration word
morphology analysis and correctly separates the meaningful word into its roots (Wang et al.,
2021).
# Data Vectorization
This study used the TfidfVectorizer method to convert the obtained text into numerical values.
This method is one of the most common methods used for text vectorization, and therefore, was
chosen by this study to be used as well because of its trustworthiness (Kumar et al., 2020).
# Method Selection
Considering what kind of method should be used in this study, the main concern was between
using either unsupervised or supervised sentiment analysis methods. Unsupervised sentiment
analysis is an effective method for extracting emotional trends in text data. However, traditional
approaches for this type of analysis often require labeled data. Transfer learning is an approach
that allows a model that uses a previously trained model in another dataset for the target dataset.
Thanks to transfer learning, labeled data is obtained from a different dataset. For the purposes of
this study, as a different complete dataset from Kaggle also exists already for purposes of model
training, it is deemed more reliable to conduct a supervised method by using the transfer learning
approach (Liu et al. 2019).
# Classification Model Selection
Different classification models were created by utilizing the Naive Bayes, Logistic Regression,
Support Vector Classification, Linear Support Vector Classification, Gradient Boosting Classifier,
Voting Classifier, and Random Forest Classifier algorithms, and they were tested on the test
dataset obtained from Kaggle. These classification methods were selected due to their
comprehensive use cases (Amrani et al., 2018). As a result of using the models, it aims to obtain
TF, TN, FP, FN, F1 score, and accuracy rate. These parameters are calculated for each model to
decide on the best model for using tweets from the Twitter API. The purpose was to select the
best-performing model to be used for the calculation of the sentiment analysis of tweets fetched
by using the custom query (Devika et al., 2016).
Within the scope of this study, the labeled dataset obtained from Kaggle split into train and test.
The ratio of the test dataset was 0.33, and the training dataset was 0.67, as it is the commonly
acknowledged practice for cross-checking mechanisms (Van der Goot, 2021).
The training dataset is necessary for the training process of the model. This data set, used to learn
emotional tendencies, enables the model to detect patterns from the data and perform sentiment
analysis (Bauer et al., 1999).
The testing dataset is necessary for estimating the model for various parameters such as accuracy,
precision, recall, and F1 score. Then, selected classification algorithms were applied to the data
set obtained from Kaggle (Bauer et al., 1999).
Using the cross-validation method, the performance of different classification algorithms for
sentiment analysis was evaluated. The "cross_val_score" function was used from Python's
sklearn's "sklearn.model_selection" library.
The following classification algorithms were tested:
- Logistic Regression
- Gradient Boosting Classifier
- Random Forest Classifier
- Naive Bayes
- Voting Classifier
- Support Vector Classifier
- Linear Support Vector Machine Classifier
Performance measures were obtained by performing 5-fold cross-validation (cv=5) for each
algorithm. These criteria return the accuracy score of each layer, and cross-validation should
increase trust in the resulting accuracy scores (Natekin et al., 2013).
The parameter F1 was chosen for reflecting the accuracy of the method in the method selection
process. The "Support Vector Classifier" algorithm was chosen as a result of the cross-validation
process, having the highest F1 score among other candidates as can be seen in Table 4.1, and this
model was applied to the dataset obtained using Twitter-API (Fersini et al., 2014).

Table 4.1 F1 scores of different models.
Model F1 score
Support Vector Classifier 0.8081899636504215
Voting Classifier 0.7938266208938398
Random Forest Classifier 0.7933661141902797
Logistic Regression 0.7899951150068742
Linear Support Vector Machine Classifier 0.7876349603004758
Naive Bayes 0.78040211302194
Gradient Boosting Classifier 0.6862665729493147
# ANALYSIS
For the purposes of the study, the analysis process for both "Sentiment Analysis" and "IS Design
System" should be discussed.
# Analysis for Sentiment Analysis
Support Vector Classifier, as the most reliable-deemed model was used for training the study's
model that is using the training dataset that was previously got from Kaggle as its training
dataset. Then, this newly trained model, which has used the method that is deemed the most
reliable based on the comparison between other methods using F1 scores for each, was used on
the tweets that were fetched by using the Twitter-API, using a custom query specifying
geological boundaries and topics of which the tweets should be fetched. These tweets included
data on tweet content, date, location, likes, and retweets.
Using this newly trained model, on the fetched tweets dataset, a sentiment result of "negative",
"neutral", or "positive" was generated for each tweet. However, in order to add quantitativeness
to this data, the result "negative" is considered "0", "neutral" is considered "0.5", "positive" is
considered "1", and their weighted averages were taken on a category or country level to generate
an average sentiment score between 0 and 1, lower than 0.5 meaning average negative sentiment,
higher than 0.5. meaning average positive sentiment, while the distance between the average
score and the point 0.5 indicates how strongly negative or positive the average sentiment is
(Kolchyna et al., 2015
