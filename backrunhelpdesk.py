import warnings
import numpy as np
warnings.filterwarnings("ignore")
import pandas as pd
main_df = pd.read_excel(r'C:\Users\DELL\OneDrive\Desktop\Internship\df_withoutdup_final_origi1.xlsx')
dftemp=main_df.copy()
main_df_new = pd.DataFrame()
main_df_new["combined data"] = "-"
main_df_new["combined data"] = main_df["Subject"] + " " + main_df["Message"]
main_df_new.head()
import string
import nltk
import re
from nltk.stem import WordNetLemmatizer
from string import punctuation
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def sent_tokens_func(text):
  return nltk.sent_tokenize(text)

def word_tokens_func(text):
  return nltk.word_tokenize(text)  

def to_lower(text):
  if not isinstance(text,str):
    text = str(text)
  return text.lower()

def number_omit_func(text):
  output = ''.join(c for c in text if not c.isdigit())
  return output

def remove_punctuation(text):
  return ''.join(c for c in text if c not in punctuation) 

def stopword_remove_func(sentence):
  stop_words = stopwords.words('english')
  return ' '.join([w for w in nltk.word_tokenize(sentence) if not w in stop_words])

def lemmatize(text):
          wordnet_lemmatizer = WordNetLemmatizer()
          lemmatized_word = [wordnet_lemmatizer.lemmatize(word)for word in nltk.word_tokenize(text)]
          return " ".join(lemmatized_word)

def preprocess(text):
        lower_text = to_lower(text)
        sentence_tokens = sent_tokens_func(lower_text)
        word_list = []
        for each_sent in sentence_tokens:
            lemmatizzed_sent = lemmatize(each_sent)
            clean_text = number_omit_func(lemmatizzed_sent)
            clean_text = remove_punctuation(clean_text)
            clean_text = stopword_remove_func(clean_text)
            word_tokens = word_tokens_func(clean_text)
            for i in word_tokens:
                word_list.append(i)
        return " ".join(word_list)

sample_data = main_df_new['combined data']
sample_data = sample_data.apply(preprocess)
sample_data_new=main_df['Solution']
sample_data_new = sample_data_new.apply(preprocess)
main_df_new['preprocessed combined data'] = sample_data
main_df['Solution']=sample_data
main_df_new.to_csv("Preprocessed_data.csv",index=False)
df_all_rows = pd.concat([main_df['Solution'],main_df_new],axis=1)
df_all_rows.to_csv("Preprocessed__data.csv",index=False)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

sentences=main_df_new['preprocessed combined data'].values.tolist()
sentence_embeddings = np.array(model.encode(sentences))