# Natural Language Processing: Bernoulli Naive Bayes vs Random Forest
### By: Corina Lentz
---
## Problem Statement

More businesses are making the transition into online sales [*source*](https://fortune.com/2020/07/15/ecommerce-online-shopping-coronavirus-business-trends-covid/), which means an increased demand for predicting online shopping behavior, using methods such as Natural Language Processing (or NLP). The medical field is also turning to artificial intelligence to automate electronic medical and health records through NLP [*source*](https://healthtechmagazine.net/article/2021/07/how-can-healthcare-leverage-natural-language-processing-medical-records-perfcon). But which NLP model will give the most accurate predictions, given two possible outcomes? In this project we will compare the Bernoulli Naive Bayes and Random Forest NLP models, to see which one is the most accurate.

---
## Data Dictionary

|Feature|Description|
|---|---|
|**subreddit**|Indicates which subreddit the post is from ('horror' indicates the r/horror subreddit and 'fantasy' indicates the r/Fantasy subreddit). Note: This column is later binarized and renamed 'is_horror' (please see below).|
|**title**|The title of the Reddit post.|
|**selftext**|The subtitle of the Reddit post which is an optional description of the post itself. Note: If there is a NaN value in this column it means either the post did not have a subtitle or that the post's subtitle was removed by the author before the data was collected.|
|**title_selftext**|The text from the 'title' and 'selftext' columns merged into one single string.|
|**is_horror**|The 'subreddit' column has been binarized so that 1 indicates the r/horror subreddit and 0 would indicate the r/Fantasy subreddit. The column name has been updated to reflect this binary.|

---

## Summary of Analysis

I compiled 5,000 posts each from my chosen subreddits: r/horror and r/Fantasy. After compiling my data I saved the columns for the title ('title'), the subtitle('selftext'), and the subreddit the post came from ('subreddit') into a dataframe. During an initial review of the first/last five rows of the new dataframe I found that there were null values and the value [removed] in many of the subtitle rows. The [removed] value indicated that the post once had a a subtitle but it has since been removed. I converted [removed] to a null value and checked the counts for the null values in each row. I was relieved to find that only the subtitle column had null values (although they made up almost half of the column). Fortunately I was able to create a new column called 'title_selftext' that merged the text from the title and subtitle columns. This meant having null values in the subtitle column wouldn't stop me from completing my analysis and I didn't have to drop those rows. I binarized the subreddit column so that horror became a one and fantasy became a 0. 

I utilized stop_words 'english' stopwords list along with CountVectorizer() to prepare the data for EDA. However,  during my initial EDA I discovered there are common stopwords written with unusual characters that weren't being filtered out. I added these words to the 'english' stopwords list along with words written in a language I don't know. Unfortunately there are a wide range of numbers that act like strings so they need to be listed out individually to be removed. There are too many for me to remove them all due to computer limitations, but I was successful in thinning them out. In reviewing the top 15 high-frequency words for each class I discovered there were several words that both classes shared which unfortunately made it difficult to get a decent accuracy score. I also added those words to the custom stopwords list.

We have an even 50/50 split between our 0 and 1 classes since an equal number of r/horror and r/Fantasy posts were collected and we didn't have to drop any rows during cleaning. Stratifying won't be needed. This also gives us a fairly low baseline of 0.5 . It was interesting to see that the top 20 words for fantasy have higher frequency numbers than the top 20 words for horror. The top words for r/horror were centered around movies and movie content, while the r/fantasy top words center around books and book content. In the horror class the titles/subtitles with the lowest word counts are either empty, appear to be advertising something, or seem to be for some sort of technical integration. In the fantasy class the majority of the titles/subtitles with the lowest word counts appear to be advertising. For both classes the titles/subtitles with the largest word counts are from the author putting their entire post into the title/subtitle section. For the horror class the largest word counts include short stories, movie reviews, and movie recommendation lists. The largest word counts for the fantasy class are from book reviews, book recommendation lists, and short stories. This is consistent with the top words for both classes. The horror class has a lower concentration of words than the fantasy class, which is also consistent with the frequencies seen when reviewing their respective top words.

Once data cleaning, processing, and EDA was complete I started working with the Bernoulli Naive Bayes model paired with the CountVectorizer as the transformer. I used a pipeline to calculate the best hyperparameters for the CountVectorizer (I used the custom stopwords list for the stop_word hyperparameter). After getting the optimal hyperparameters setup I was able to fit the model and take a look at how it performed. The accuracy for the Bernoulli Naive Bayes model was 85.15% (well above our 50% baseline) and the variance was small (difference of 0.004332 between our training and testing scores). In reviewing the confusion matrix I found the model definitely had difficulty correctly predicting our fantasy (0) class with 406 False Positives vs only 70 False Negatives for our horror (1) class.

I also used the CountVectorizer with the Random Forest model. I used the pipeline to calculate the optimal hyperparameters for both the model and the transformer. Once those were setup I fit the model and took a look at its performance. With an accuracy score of 85.61% the Random Forest model definitely beat our baseline accuracy of 50% and did slightly better than the Bernoulli NB model. Random Forest has slightly more variance with 0.004332 difference between the training and testing sets versus 0.004265. The confusion matrix shows that Random Forest model did noticeably better at predicting our horror (1) class with only 53 False Negatives. However it did slightly worse in predicting our fantasy (0) class with 410 False Positives.

---
## Conclusion & Recommendations

Although the Random Forest model was more accurate than the Bernoulli Naive Bayes model, the difference was fairly small. Either of these models could be used to give excellent results in classification.

Going forward, I would recommend dedicating more time to building-out an even more robust stop words list to improve accuracy even more. I’d also recommending testing a Logistic Regression model for further comparison.

---
## Resources
- https://fortune.com/2020/07/15/ecommerce-online-shopping-coronavirus-business-trends-covid/
- https://healthtechmagazine.net/article/2021/07/how-can-healthcare-leverage-natural-language-processing-medical-records-perfcon
- https://en.wikipedia.org/wiki/Natural_language_processing 
- https://en.wikipedia.org/wiki/Random_forest
- https://en.wikipedia.org/wiki/Bayes'_theorem
- https://github.com/pushshift/api
- https://www.reddit.com/r/horror/
- https://www.reddit.com/r/Fantasy/


