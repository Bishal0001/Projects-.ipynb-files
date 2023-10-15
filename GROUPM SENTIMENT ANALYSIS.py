#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np 
import pandas as pd 
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import re
from selenium import webdriver
import joblib


# ## IMDB

# In[166]:


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time

imdb_ = []
driver = webdriver.Chrome('')
driver.get('https://www.imdb.com/title/tt8111088/reviews/?ref_=tt_ov_rt')

while True:
    try:
        load_more_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="load-more-trigger"]')))

        load_more_button.click()
    except Exception as e:
        break  # Exit the loop if the "load more" button is not found or clickable

    time.sleep(5)  # Give the page some time to load the additional reviews

    # Get the updated page source
    page_source = driver.page_source

    # Parse the page source with BeautifulSoup
    soup = BeautifulSoup(page_source, 'lxml')

    # Find and extract the reviews
    reviews = soup.find_all('div', {"class": "text show-more__control"})
    for review in reviews:
        imdb_.append(review.text)
        print(len(imdb_))




# ## YT
# 

# In[ ]:


data = []

url = 'https://www.youtube.com/watch?v=Znsa4Deavgg'


driver = webdriver.Chrome()
wait = WebDriverWait(driver,10)
driver.get(url)
driver.maximize_window()
o=0


for item in range(80):
    wait.until(EC.visibility_of_element_located((By.TAG_NAME, 'body'))).send_keys(Keys.END)
    o+=1
    print(o,end=' ')
    time.sleep(4)
for comments in wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR,'#comment #content-text'))):
    data.append(comments.text)

    print(len(data),end=' ')
        


# ## GOOGLE REVIEW
# 

# In[167]:


start_time = time.time()
data = []
# path = './driver/chromedriver'
# service = Service(executable_path=path)
url = 'https://www.google.com/search?q=the+mandalorian++googles+review&sca_esv=572781667&sxsrf=AM9HkKkLZjeuSB9AY2YipKKzPxQJgIAjeg%3A1697091455056&ei=f48nZZqIA9aLoATujajYDA&ved=0ahUKEwjaw9Gq7u-BAxXWBYgKHe4GCssQ4dUDCBA&uact=5&oq=the+mandalorian++googles+review&gs_lp=Egxnd3Mtd2l6LXNlcnAiH3RoZSBtYW5kYWxvcmlhbiAgZ29vZ2xlcyByZXZpZXcyBxAjGLACGCdInwlQ9wRY9wRwAXgBkAEAmAG_AaABvwGqAQMwLjG4AQPIAQD4AQHCAgoQABhHGNYEGLAD4gMEGAAgQYgGAZAGCA&sclient=gws-wiz-serp'

# with webdriver.Chrome(service=service) as driver:
driver = webdriver.Chrome()
wait = WebDriverWait(driver,10)
driver.get(url)
# driver.maximize_window()
o=0

load_more_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="kp-wp-tab-FilmReview"]/div[1]/div/div/div[2]/div[2]/div/div/div/div/g-more-link/div')))
load_more_button.click()
time.sleep(5)





load_more_button = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.tEJZ0b')))
for i in range(399):
    print(i,end=' ')
    load_more_button[i].click()
    time.sleep(4)
    
    if i == 390:
        time.sleep(1500)
        break
    # Check if you've clicked all buttons
    elif i == len(load_more_button) - 1:
        print('No more buttons to click, scrolling down.')
        load_more_button = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.tEJZ0b')))
        driver.execute_script("window.scrollBy(0, window.innerHeight);")
        time.sleep(4)  # Add a delay after scrolling


        
    
    
    
#     ****wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, '.U1vjCc'))).send_keys(Keys.END)
    
#     o+=1
#     print(o,end=' ')
#     time.sleep(4)
for reviews in wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR,'.T7nuU'))):
    data.append(reviews.text)
    time.sleep(1)

    print(len(data),end=' ')
end_time = time.time()        


# ##  Dumping and Loading Data 

# In[5]:


import joblib
joblib.dump(data,'google_review.joblib')


# In[4]:


import joblib
yt_comments = joblib.load('yt_comments.joblib')


# In[246]:


import joblib
imdb = joblib.load('imdb_data.joblib')
len(imdb)


# In[249]:


[i for i,x in enumerate(imdb) if x == '''We enjoyed seasons 1 and 2. Unfortunately season 3 seems to exist as the Mandalorian version of Star Wars, Return of the Jedi, the Ewok merchandise money grab. Basically it's a vehicle to market Baby Yoda merch. The writing and direction play as afterthoughts. The narrative is now just a jumble of pointless opportunities for the green puppet to do cute things. The first two seasons had that delightful space western feel. This feels like the longest Super Bowl ad ever made. A collection of quality actors who always bring their A-game, the production just abandoned them in favor of a padded out, pointless exercise in greed.''']


# In[250]:


imdb[164]


# In[ ]:





# In[ ]:





# In[15]:


len(set(joblib.load('yt_comments.joblib'))) + len(set(joblib.load('yt_comments2.joblib'))) +  len(set(joblib.load('imdb_data.joblib'))) +   len(set(joblib.load('google_review.joblib')))


# In[10]:


yt_comments = list(set(joblib.load('yt_comments.joblib')))


# In[12]:


len(set(joblib.load('yt_comments2.joblib')))


# In[13]:


len(set(joblib.load('imdb_data.joblib')))


# In[14]:


len(set(joblib.load('google_review.joblib')))


# ## DATA PREP

# In[141]:


yt_comments = list(set(joblib.load('yt_comments.joblib')))
yt_comments2 = list(set(joblib.load('yt_comments2.joblib')))
imdb_data = list(set(joblib.load('imdb_data.joblib')))
google_review = list(set(joblib.load('google_review.joblib')))


# In[142]:


import re

print((len(yt_comments)))
punc = '''{}[]()-'";!,:/\?><|_@#$%^&*~`+=.'''


c = []
translation_table = str.maketrans('', '', punc)

cleaned_comments = [comment.translate(translation_table) for comment in yt_comments]

for cleaned_comment in cleaned_comments:
    c.append(cleaned_comment)



yt_comments = [i for i in c if not re.findall("[^\u0000-\u05C0\u2100-\u214F]+",i)]

yt_comments = [i.replace('\n','') for i in yt_comments]
yt_comments = [i for i in yt_comments if i != '']
len(yt_comments)


# In[143]:


print(len(yt_comments2))

c1 = []
translation_table = str.maketrans('', '', punc)

cleaned_comments = [comment.translate(translation_table) for comment in yt_comments2]

for cleaned_comment in cleaned_comments:
    c1.append(cleaned_comment)


yt_comments2 = [i for i in c1 if not re.findall("[^\u0000-\u05C0\u2100-\u214F]+",i)]
yt_comments2 = [i.replace('\n','') for i in yt_comments2]
yt_comments2 = [i for i in yt_comments2 if i != '']
len(yt_comments2)


# In[144]:


print(len(imdb_data))

c2 = []
translation_table = str.maketrans('', '', punc)

cleaned_comments = [comment.translate(translation_table) for comment in imdb_data]

for cleaned_comment in cleaned_comments:
    c2.append(cleaned_comment)
    
imdb_data = [i for i in c2 if not re.findall("[^\u0000-\u05C0\u2100-\u214F]+",i)]
imdb_data = [i.replace('\n','') for i in imdb_data]
imdb_data = [i for i in imdb_data if i != '']
len(imdb_data)


# In[145]:


print(len(google_review))

c3 = []
translation_table = str.maketrans('', '', punc)

cleaned_comments = [comment.translate(translation_table) for comment in google_review]

for cleaned_comment in cleaned_comments:
    c3.append(cleaned_comment)

google_review = [i for i in c3 if not re.findall("[^\u0000-\u05C0\u2100-\u214F]+",i)]
google_review = [i.replace('\n','') for i in google_review]
google_review = [i for i in google_review if i != '']
len(google_review)


# In[146]:


add_on = google_review[-579:]
add_on += imdb_data[-196:]



google_review = google_review[:2632]



imdb_data = imdb_data[:2632]


# In[147]:


len(imdb_data)


# In[148]:


len(google_review)


# In[149]:


yt_comments+=yt_comments2


# In[150]:


yt_comments+=add_on


# In[151]:


len(yt_comments)


# In[ ]:


yt_comments,google_review,imdb_data


# In[152]:


data = {
    'yt_comments': yt_comments,
    'google_reviews': google_review,
    'imdb_reviews': imdb_data
}


# In[158]:


df = pd.DataFrame(data)


# In[163]:


df[:-831]


# In[164]:


df[-831:]


# In[165]:


df.to_csv('groupm_project')


# In[166]:


df


# In[2]:


df = pd.read_csv('final_groupm_project2.csv')


# In[3]:


df = df.iloc[:,1:]


# In[4]:


df.head()


# In[43]:


df


# In[165]:


# for i in range(len(df)):
#     df['yt_comments_label'][i] == 'NEGATIVE' and


# In[ ]:





# In[48]:


df.iloc[:,3:].head()


# In[49]:


def determine_final_label(row):
    num_negatives = (row == 'NEGATIVE').sum()
    return 0 if num_negatives in (2, 3) else 1


df['final_label'] = df.apply(determine_final_label, axis=1)


# In[ ]:





# In[59]:


columns_to_exclude = ['yt_comments_label', 'google_reviews_label', 'imdb_reviews_label']
df = df.drop(columns=columns_to_exclude)


# In[45]:


# df.to_csv('final_groupm_project1.csv')
df.to_csv('final_groupm_project2.csv')


# ---

# ---

# ---

# In[14]:


df = pd.read_csv('final_groupm_project1.csv')
df = df.iloc[:,1:]


# In[15]:


df.head()


# ### Quick EDA

# In[16]:


df.final_label.value_counts(normalize=True).plot(kind='bar')


# * ***Nearly 80% of reviews are positive*** 

# In[17]:


df.describe(include=object)


# * ***yt_comments column has 14 duplicates***

# In[18]:


df.info()


# ## VADER SCORE

# In[19]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# from tdqm import tdqm


# In[20]:


sia = SentimentIntensityAnalyzer()


# In[21]:


sia.polarity_scores('fuck you')


# In[22]:


def vedar_score_neg(text):
    res = sia.polarity_scores(text)
    return res['neg']
    
def vedar_score_neu(text):
    res = sia.polarity_scores(text)
    return res['neu']
    
def vedar_score_pos(text):
    res = sia.polarity_scores(text)
    return res['pos']
    
def vedar_score_compound(text):
    res = sia.polarity_scores(text)
    return res['compound']


# In[23]:


df['vader_yt_neg'] = df['yt_comments'].apply(vedar_score_neg)
df['vader_yt_pos'] = df['yt_comments'].apply(vedar_score_pos)
df['vader_yt_neu'] = df['yt_comments'].apply(vedar_score_neu)
df['vader_yt_compound'] = df['yt_comments'].apply(vedar_score_compound)



df['vader_gr_neg'] = df['google_reviews'].apply(vedar_score_neg)
df['vader_gr_pos'] = df['google_reviews'].apply(vedar_score_pos)
df['vader_gr_neu'] = df['google_reviews'].apply(vedar_score_neu)
df['vader_gr_compound'] = df['google_reviews'].apply(vedar_score_compound)




df['vader_ir_neg'] = df['imdb_reviews'].apply(vedar_score_neg)
df['vader_ir_pos'] = df['imdb_reviews'].apply(vedar_score_pos)
df['vader_ir_neu'] = df['imdb_reviews'].apply(vedar_score_neu)
df['vader_ir_compound'] = df['imdb_reviews'].apply(vedar_score_compound)


# In[24]:


df.columns


# In[25]:


df = df[['yt_comments','vader_yt_neg', 'vader_yt_pos', 'vader_yt_neu', 'vader_yt_compound', 'google_reviews',
    'vader_gr_neg', 'vader_gr_pos', 'vader_gr_neu', 'vader_gr_compound', 'imdb_reviews', 'vader_ir_neg',
    'vader_ir_pos', 'vader_ir_neu', 'vader_ir_compound', 'final_label']]


# In[26]:


df


# In[27]:


sns.barplot(data = df, y='vader_ir_compound', x='final_label')


# In[28]:


fig, axs = plt.subplots(1,3, figsize=(15, 5))
sns.barplot(data = df, y='vader_ir_pos', x='final_label', ax = axs[0])
sns.barplot(data = df, y='vader_ir_neu', x='final_label', ax = axs[1])
sns.barplot(data = df, y='vader_ir_neg', x='final_label', ax = axs[2])
axs[0].set_title('Positive_imdb')
axs[1].set_title('Neutral_imdb')
axs[2].set_title('Negative_imdb')
plt.tight_layout()
plt.show()



fig, axs = plt.subplots(1,3, figsize=(15, 5))
sns.barplot(data = df, y='vader_yt_pos', x='final_label', ax = axs[0])
sns.barplot(data = df, y='vader_yt_neu', x='final_label', ax = axs[1])
sns.barplot(data = df, y='vader_yt_neg', x='final_label', ax = axs[2])
axs[0].set_title('Positive_yt')
axs[1].set_title('Neutral_yt')
axs[2].set_title('Negative_yt')
plt.tight_layout()
plt.show()


fig, axs = plt.subplots(1,3, figsize=(15, 5))
sns.barplot(data = df, y='vader_gr_pos', x='final_label', ax = axs[0])
sns.barplot(data = df, y='vader_gr_neu', x='final_label', ax = axs[1])
sns.barplot(data = df, y='vader_gr_neg', x='final_label', ax = axs[2])
axs[0].set_title('Positive_google')
axs[1].set_title('Neutral_google')
axs[2].set_title('Negative_google')
plt.tight_layout()
plt.show()





# * ***It's evident that as the positive sentiment scores from VADER increase, the corresponding final label also tends to increase, reflecting a more positive classification. Conversely, when VADER assigns negative sentiment scores, the final label decreases, indicating a more negative classification. Hence it validates our data***

# ## Roberta Pretrained Model

# In[29]:


from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax


# In[30]:


import warnings
warnings.filterwarnings("ignore")

model = f'cardiffnlp/twitter-roberta-base-sentiment'
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForSequenceClassification.from_pretrained(model)


# In[31]:


encoded_text =tokenizer('fuck you',return_tensors='pt')
output = model(**encoded_text)
scores =output[0][0].detach().numpy()
scores = softmax(scores)
score_dict = {
    'roberta_neg':scores[0],
    'roberta_neu':scores[1],
    'roberta_pos':scores[2]
}

score_dict


# In[35]:


'jjjjjjjj'[:300]


# In[41]:


def roberta_scores_neg(text):
    try:
        encoded_text = tokenizer(text[:2000], return_tensors='pt')
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        score_dict = {
            'roberta_neg': scores[0],
            'roberta_neu': scores[1],
            'roberta_pos': scores[2]
        }
    except RuntimeError:
        encoded_text = tokenizer(text[:1000], return_tensors='pt')
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        score_dict = {
            'roberta_neg': scores[0],
            'roberta_neu': scores[1],
            'roberta_pos': scores[2]
        }

    return score_dict['roberta_neg']


def roberta_scores_neu(text):
    try:
        encoded_text =tokenizer(text[:2000],return_tensors='pt')
        output = model(**encoded_text)
        scores =output[0][0].detach().numpy()
        scores = softmax(scores)
        score_dict = {
            'roberta_neg':scores[0],
            'roberta_neu':scores[1],
            'roberta_pos':scores[2]
        }
    except RuntimeError:
        encoded_text = tokenizer(text[:1000], return_tensors='pt')
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        score_dict = {
            'roberta_neg': scores[0],
            'roberta_neu': scores[1],
            'roberta_pos': scores[2]
        }

    return score_dict['roberta_neu']




def roberta_scores_pos(text):
    try:
        encoded_text =tokenizer(text[:2000],return_tensors='pt')
        output = model(**encoded_text)
        scores =output[0][0].detach().numpy()
        scores = softmax(scores)
        score_dict = {
            'roberta_neg':scores[0],
            'roberta_neu':scores[1],
            'roberta_pos':scores[2]
        }
    except RuntimeError:
        encoded_text = tokenizer(text[:1000], return_tensors='pt')
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        score_dict = {
            'roberta_neg': scores[0],
            'roberta_neu': scores[1],
            'roberta_pos': scores[2]
        }
    return score_dict['roberta_pos']


# In[42]:


df['yt_comments'].apply(roberta_scores_neg)


# In[43]:


df['roberta_yt_neg'] = df['yt_comments'].apply(roberta_scores_neg)
df['roberta_yt_pos'] = df['yt_comments'].apply(roberta_scores_pos)
df['roberta_yt_neu'] = df['yt_comments'].apply(roberta_scores_neu)



df['roberta_gr_neg'] = df['google_reviews'].apply(roberta_scores_neg)
df['roberta_gr_pos'] = df['google_reviews'].apply(roberta_scores_pos)
df['roberta_gr_neu'] = df['google_reviews'].apply(roberta_scores_neu)




df['roberta_ir_neg'] = df['imdb_reviews'].apply(roberta_scores_neg)
df['roberta_ir_pos'] = df['imdb_reviews'].apply(roberta_scores_pos)
df['roberta_ir_neu'] = df['imdb_reviews'].apply(roberta_scores_neu)


# In[44]:


df


# In[46]:


df.columns


# In[69]:


df = df[['yt_comments', 'vader_yt_neg', 'vader_yt_pos', 'vader_yt_neu',
       'vader_yt_compound',
       'roberta_yt_neg', 'roberta_yt_pos', 'roberta_yt_neu', 'google_reviews', 'vader_gr_neg', 'vader_gr_pos',
       'vader_gr_neu', 'vader_gr_compound', 'roberta_gr_neg',
       'roberta_gr_pos', 'roberta_gr_neu', 'imdb_reviews', 'vader_ir_neg',
       'vader_ir_pos', 'vader_ir_neu', 'vader_ir_compound', 'roberta_ir_neg', 'roberta_ir_pos',
       'roberta_ir_neu', 'final_label']]
df.head(50)


# In[48]:


fig, axs = plt.subplots(1,3, figsize=(15, 5))
sns.barplot(data = df, y='roberta_ir_pos', x='final_label', ax = axs[0])
sns.barplot(data = df, y='roberta_ir_neu', x='final_label', ax = axs[1])
sns.barplot(data = df, y='roberta_ir_neg', x='final_label', ax = axs[2])
axs[0].set_title('Positive_imdb')
axs[1].set_title('Neutral_imdb')
axs[2].set_title('Negative_imdb')
plt.tight_layout()
plt.show()



fig, axs = plt.subplots(1,3, figsize=(15, 5))
sns.barplot(data = df, y='roberta_yt_pos', x='final_label', ax = axs[0])
sns.barplot(data = df, y='roberta_yt_neu', x='final_label', ax = axs[1])
sns.barplot(data = df, y='roberta_yt_neg', x='final_label', ax = axs[2])
axs[0].set_title('Positive_yt')
axs[1].set_title('Neutral_yt')
axs[2].set_title('Negative_yt')
plt.tight_layout()
plt.show()


fig, axs = plt.subplots(1,3, figsize=(15, 5))
sns.barplot(data = df, y='roberta_gr_pos', x='final_label', ax = axs[0])
sns.barplot(data = df, y='roberta_gr_neu', x='final_label', ax = axs[1])
sns.barplot(data = df, y='roberta_gr_neg', x='final_label', ax = axs[2])
axs[0].set_title('Positive_google')
axs[1].set_title('Neutral_google')
axs[2].set_title('Negative_google')
plt.tight_layout()
plt.show()





# * ***Clearly, when RoBERTa assigns higher positive sentiment scores, the final label tends to be more positive, affirming a positive classification. Conversely, when RoBERTa assigns lower sentiment scores, the final label tends to be more negative, indicating a negative classification. By observing negative, positive and neutral sentiment data it suggests that the data aligns more closely with the RoBERTa model, indicating a higher level of generalization compared to the Vader model.***

# In[ ]:





# In[66]:


# sns.histplot(data=df,x='vader_yt_neu')
sns.histplot(data=df[df.vader_yt_pos > 0.5],x='vader_yt_pos')
sns.histplot(data=df[df.roberta_yt_pos > 0.5],x='roberta_yt_pos')


# In[ ]:


df


# In[62]:


df.info()


# In[80]:


df[['vader_yt_neg', 'vader_yt_neu', 'vader_yt_pos']].idxmax(axis=1).value_counts(normalize=True).plot(kind='bar')


# * ***Upon analyzing the results of the Vader sentiment analysis model, it is evident that a significant portion of the data (approximately 90%) is classified as neutral. This suggests that the Vader model may not perform well in accurately predicting sentiment for this particular dataset. As a result, it is intended not to proceed with further analysis based on the Vader model's predictions.***

# In[90]:


yt_senti_plot = df[['roberta_yt_neg', 'roberta_yt_neu', 'roberta_yt_pos']].idxmax(axis=1).value_counts(normalize=True)


# In[93]:


gr_senti_plot = df[['roberta_gr_neg', 'roberta_gr_neu', 'roberta_gr_pos']].idxmax(axis=1).value_counts(normalize=True)


# In[94]:


ir_senti_plot = df[['roberta_ir_neg', 'roberta_ir_neu', 'roberta_ir_pos']].idxmax(axis=1).value_counts(normalize=True)


# In[116]:


gr_senti_plot = gr_senti_plot.reset_index().rename(columns={'index':'google',0:'percentage'})
yt_senti_plot = yt_senti_plot.reset_index().rename(columns={'index':'youtube',0:'percentage'})
ir_senti_plot = ir_senti_plot.reset_index().rename(columns={'index':'imdb',0:'percentage'})


# In[120]:


gr_senti_plot = gr_senti_plot.iloc[:,1:]
ir_senti_plot = ir_senti_plot.iloc[:,1:]
yt_senti_plot = yt_senti_plot.iloc[:,1:]


# In[ ]:





# In[188]:


import matplotlib.pyplot as plt

x = ['positive', 'negative', 'neutral']
youtube = [0.648176, 0.099544, 0.252280]
google = [0.862082, 0.100304, 0.037614]
imdb = [0.681231, 0.232523, 0.086246]

# Define the width of the bars
plt.figure(figsize=(13, 6))
bar_width = 0.2
vertical_offset = 0.02  # Adjust this value for the desired gap

# Define the x-axis positions for each group of bars
x_pos = range(len(x))

# Create the barplot for 'youtube'
plt.bar([i - bar_width for i in x_pos], youtube, width=bar_width, label='youtube')
# Create the barplot for 'google'
plt.bar(x_pos, google, width=bar_width, label='google')
# Create the barplot for 'imdb'
plt.bar([i + bar_width for i in x_pos], imdb, width=bar_width, label='imdb')

# Set the x-axis labels
plt.xticks(x_pos, x)

# Add labels on top of the bars with a gap
for i, value in enumerate(youtube):
    plt.annotate(f'{round(value * 100, 2)}%', (i - bar_width, value + vertical_offset), ha='center')
for i, value in enumerate(google):
    plt.annotate(f'{round(value * 100, 2)}%', (i, value + vertical_offset), ha='center')
for i, value in enumerate(imdb):
    plt.annotate(f'{round(value * 100, 2)}%', (i + bar_width, value + vertical_offset), ha='center')

# Add a legend
plt.legend()

# Set labels and title
plt.xlabel('Sentiment')
plt.ylabel('Percentage')
plt.title('Sentiment Distribution by Platforms')

# Show the plot
plt.show()


# **```Comprehensive Sentiment Analysis of "The Mandalorian" Series Across Platforms```**
# 
# In our analysis of audience sentiment for "The Mandalorian" series, we collected and examined a total of 2,632 rows of data from YouTube comments, IMDb reviews, and Google reviews. Our goal was to gain insights into the viewers' sentiments and how they perceive the series.
# 
# **Dominance of Positive Sentiment:**
# 
# * ***Google Reviews***: Remarkably, Google reviews stood out as the platform with the highest prevalence of positive sentiment, where approximately ***87%*** of reviews conveyed enthusiasm and admiration. This robust positive reception on Google reflects a substantial and appreciative viewership, highlighting the series' appeal on this platform.
# 
# * ***YouTube Comments***: Equally noteworthy, YouTube comments exhibited a strong positive sentiment, accounting for around ***65%*** of the interactions. This significant positivity indicates that "The Mandalorian" enjoys immense popularity on YouTube, resonating with a vast audience of fans.
# 
# * ***IMDb Reviews***: IMDb, too, contributed to the prevailing positive sentiment, with roughly ***67%*** of reviews reflecting favorable opinions. The series has garnered extensive acclaim and recognition on IMDb, showcasing its impact on the film and television community.
# 
# **Variation in Negative Sentiment:**
# 
# * ***Google Reviews***: While Google reviews predominantly portrayed positivity, it's worth acknowledging that approximately ***10%*** of the reviews expressed negative sentiment. This data implies that while the series is highly regarded, it may not align with the preferences of all Google reviewers.
# 
# * ***YouTube Comments***: On YouTube, the percentage of negative comments was around ***10%***. This suggests that while the series enjoys a substantial following, it is not immune to critical viewpoints.
# 
# * ***IMDb Reviews***: IMDb exhibited the highest proportion of negative sentiment, with approximately ***24%*** of reviews expressing dissatisfaction. Even though the series is celebrated, it remains polarizing on IMDb, with a significant number of users holding critical views.
# 
# **Presence of Neutral Sentiment:**
# 
# * ***Google Reviews***: A noteworthy ***4%*** of Google reviews exhibited a neutral sentiment, signaling the diversity of opinions within the Google user base.
# 
# * ***YouTube Comments***: On YouTube, roughly ***26%*** of comments struck a neutral tone, reflecting the presence of diverse perspectives among the platform's audience.
# 
# * ***IMDb Reviews***: IMDb contributed approximately ***9%*** of neutral reviews, reinforcing the notion of mixed opinions within its user community.
# 
# **Platform-Specific Insights:**
# 
# ```Notably, our analysis revealed two key platform-specific observations. Google reviews received the highest number of positive ratings, underscoring its significance as a platform for positive sentiment expression. In contrast, IMDb garnered a higher count of negative sentiment expressions, indicating a relatively higher degree of critical reviews on this platform.```
# 
# ```In summary, "The Mandalorian" has achieved a remarkable level of positive sentiment across Google, YouTube, and IMDb, attracting a diverse and appreciative audience. However, it is essential to acknowledge the varying degrees of negative and neutral sentiment. The series has successfully captured the attention and admiration of a broad audience, but it is not without its critics. These insights can be invaluable for understanding the reception of the series across different online platforms and tailoring marketing and engagement strategies accordingly.```

# In[ ]:





# In[ ]:





# In[164]:


# date


# In[47]:


import joblib
date = joblib.load('dates_imdb.joblib')


# In[48]:


d1 = pd.DataFrame(date).rename(columns={0:'date'})


# In[49]:


d1 = d1.astype('datetime64')


# In[50]:


d1.set_index('date', inplace=True)


# In[51]:


d1 = d1[::-1]


# In[69]:


# Plot the index (date)

plt.figure(figsize=(12,5))

plt.plot(d1.index, range(len(d1.index)))  # Plot the index as numeric values
plt.xlabel('Date')
plt.ylabel('Index')

# Customize the x-axis ticks
plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(prune='both'))  # Show all x-axis dates
plt.show()


# In[ ]:





# In[89]:


dates = pd.to_datetime(date, format='%d %B %Y')


# In[93]:


d_2020 = dates[dates<'2021-01-01']


# In[96]:


d_2021 = dates[(dates>='2021-01-01') & (dates<'2022-01-01')]


# In[98]:


d_2022 = dates[(dates>='2022-01-01') & (dates<'2023-01-01')]


# In[99]:


d_2023 = dates[(dates>='2023-01-01') & (dates<'2024-01-01')]


# In[121]:


d_2020 = pd.DataFrame(d_2020).rename(columns={0:'date'})
d_2020 = d_2020.astype('datetime64')
d_2020.set_index('date', inplace=True)
d_2020 = d_2020[::-1]


d_2021 = pd.DataFrame(d_2021).rename(columns={0:'date'})
d_2021 = d_2021.astype('datetime64')
d_2021.set_index('date', inplace=True)
d_2021 = d_2021[::-1]


d_2022 = pd.DataFrame(d_2022).rename(columns={0:'date'})
d_2022 = d_2022.astype('datetime64')
d_2022.set_index('date', inplace=True)
d_2022 = d_2022[::-1]



d_2023 = pd.DataFrame(d_2023).rename(columns={0:'date'})
d_2023 = d_2023.astype('datetime64')
d_2023.set_index('date', inplace=True)
d_2023 = d_2023[::-1]


# In[120]:





# In[ ]:


d1 = pd.DataFrame(date).rename(columns={0:'date'})


# In[108]:


import matplotlib.pyplot as plt

# Create a 2x2 grid of subplots (2 rows, 2 columns)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

# Plot for 2020
axes[0, 0].plot(d_2020.index, range(len(d_2020.index)))
axes[0, 0].set_title('2020')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Index')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].xaxis.set_major_locator(plt.MaxNLocator(prune='both'))

# Plot for 2021
axes[0, 1].plot(d_2021.index, range(len(d_2021.index)))
axes[0, 1].set_title('2021')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Index')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].xaxis.set_major_locator(plt.MaxNLocator(prune='both'))

# Plot for 2022
axes[1, 0].plot(d_2022.index, range(len(d_2022.index)))
axes[1, 0].set_title('2022')
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('Index')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].xaxis.set_major_locator(plt.MaxNLocator(prune='both'))

# Plot for 2023
axes[1, 1].plot(d_2023.index, range(len(d_2023.index)))
axes[1, 1].set_title('2023')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Index')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].xaxis.set_major_locator(plt.MaxNLocator(prune='both'))

# Adjust the layout
plt.tight_layout()

# Show the figure
plt.show()


# In[109]:


import matplotlib.pyplot as plt

# Create a single figure
plt.figure(figsize=(15, 6))

# Plot for 2020
plt.plot(d_2020.index, range(len(d_2020.index)), label='2020')

# Plot for 2021
plt.plot(d_2021.index, range(len(d_2021.index)), label='2021')

# Plot for 2022
plt.plot(d_2022.index, range(len(d_2022.index)), label='2022')

# Plot for 2023
plt.plot(d_2023.index, range(len(d_2023.index)), label='2023')

# Customize labels and titles
plt.xlabel('Date')
plt.ylabel('Index')
plt.title('Line Plots for Different Years')

# Add a legend to distinguish between the lines
plt.legend()

# Customize the x-axis ticks
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(prune='both'))

# Show the plot
plt.show()


# In[144]:


import matplotlib.pyplot as plt

# Create a single figure
plt.figure(figsize=(15, 6))

# Plot histograms for each year on the same plot
plt.hist(d_2020.index, bins=50, alpha=0.5, color='blue', label='2020')
plt.hist(d_2021.index, bins=50, alpha=0.5, color='green', label='2021')
plt.hist(d_2022.index, bins=40, alpha=0.5, color='red', label='2022')
plt.hist(d_2023.index, bins=40, alpha=0.5, color='purple', label='2023')

# Customize labels and title with increased font size
plt.xlabel('Date', fontsize=14)  # Increase font size for the x-label
plt.ylabel('Counts', fontsize=14)  # Increase font size for the y-label
plt.title('Distribution of Year-wise Review Counts', fontsize=16)  # Increase font size for the title

# Add a legend to distinguish between the years with increased font size
plt.legend(fontsize=12)

# Increase the font size of the x-tick labels
plt.xticks(fontsize=12)

# Increase the font size of the y-tick labels
plt.yticks(fontsize=12)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# The Mandalorian, a popular series, has had its fair share of reviews on IMDb over the past few years. Analyzing the review trends, we can draw some noteworthy insights:
# 
# **1. Consistent Popularity in 2020:** In the year 2020, "The Mandalorian" garnered immense attention from its viewers and fans, leading to an impressive count of nearly 2500 reviews on IMDb. This remarkable achievement underlines the show's broad appeal and its ability to captivate a vast audience.
# 
# The primary catalyst behind this significant success in 2020 can be traced back to the show's strategic episode releases in late 2019 and late 2020. This clever scheduling approach resulted in the release of episodes twice in just over a year and two months.
# 
# **2. A Dip in Viewer Engagement in 2021:** The following year, 2021, saw a notable decrease in the number of reviews, which dropped to 400. This decline suggests a temporary lull in viewer engagement. The reasons for this dip could range from the show's availability or a potential hiatus between seasons.
# 
# **3. A Further Drop in 2022:** Unfortunately, the trend of declining reviews continued into 2022, with the count decreasing by more than half, reaching an absolute count of just 175 reviews. The significant drop in reviews raises questions about the series' reception and popularity during that time.
# 
# **4. Signs of a Revival in 2023:** Despite the challenging trends observed in the past two years, there is a glimmer of hope in 2023. As of the current year, 2023, we have already seen 200 reviews, suggesting that the series is making a comeback in terms of viewer engagement. While this number might not be as high as in 2020, it's a positive sign that the show is finding its footing once again.
# 
# In conclusion, the journey of viewer engagement with The Mandalorian has been a rollercoaster, with peaks and troughs. The series remains popular and is showing signs of resurgence in 2023, proving its lasting appeal to audiences. The reasons behind these fluctuations could be multifaceted, including factors related to the show's release schedule, viewer preferences, and more. It will be interesting to observe how the series continues to evolve and engage its audience in the future.

# In[163]:


import matplotlib.pyplot as plt
import pandas as pd

# Define the data
dates = [ 'November 15, 2019','December 27, 2019','October 30, 2020','December 18, 2020','March 1, 2023','April 19, 2023']

# Convert date strings to datetime objects
date_objects = pd.to_datetime(dates)

# Create a single figure
plt.figure(figsize=(15, 6))

# Plot the histogram for 2020
plt.hist(d_2020.index, bins=20, alpha=0.5, color='blue', label='2020')

# # Plot the histogram for 2021
plt.hist(d_2021.index, bins=20, alpha=0.5, color='green', label='2021')

# # Plot the histogram for 2022
plt.hist(d_2022.index, bins=20, alpha=0.5, color='red', label='2022')

# # Plot the histogram for 2023
plt.hist(d_2023.index, bins=20, alpha=0.5, color='purple', label='2023')

# Customize labels and title
plt.xlabel('Date', fontsize=14)  # Increase font size for the x-label
plt.ylabel('Counts', fontsize=14)  # Increase font size for the y-label
plt.title('Time-span within which an entire season got released', fontsize=16)  # Increase font size for the title

# Add a legend to distinguish between the years
plt.legend()

# Set custom x-tick positions and labels
x_positions = date_objects
x_labels = [date.strftime('%b %d, %Y') for date in date_objects]

# Set the x-ticks and labels with increased font size
plt.gca().set_xticks(x_positions)
plt.gca().set_xticklabels(x_labels, rotation=45, ha='right', fontsize=12)  # Increase font size for x-tick labels

# Increase the font size of the y-tick labels
plt.yticks(fontsize=12)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# Certainly, let's break down the insights into a concise, point-wise narrative:
# 
# **1. Review Counts and Release Dates**: The data reveals distinct spikes in review counts for "The Mandalorian" on IMDb. These spikes occurred around specific dates: '**November 22, 2019**,' '**November 20, 2020**,' and '**March 22, 2023**.'
# 
# **2. New Season Releases**: Intriguingly, these dates align perfectly with the release dates of new seasons of the series. This alignment suggests a strong connection between the release of fresh content and heightened audience engagement.
# 
# **3. Prompt Audience Participation**: The correlation underscores the immediate enthusiasm of the show's fanbase. Fans not only eagerly await new season releases but also promptly express their thoughts and opinions by posting reviews on IMDb.
# 
# **4. Dedicated Fan Community**: The data reflects a dedicated community of fans actively engaging with the series. These review spikes are a testament to the enthusiasm and commitment of "The Mandalorian" fanbase.
# 
# **5. Amplified Anticipation**: The review spikes are not mere data points; they signify the anticipation and excitement surrounding each new season of the series. They showcase how new content can effectively drive audience interaction and engagement.
# 
# ```In summary, the data tells a compelling story of how new content releases, in this case, the new seasons of "The Mandalorian," play a pivotal role in energizing audience engagement. Each spike in reviews reflects the passion and prompt participation of dedicated fans, making it a noteworthy example of how fresh content can ignite and sustain audience enthusiasm.```

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




