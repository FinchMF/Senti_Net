import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer


lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()



def preprocess_tweet(text):

    clean_tw = [char for char in text if char not in string.punctuation]
    clean_tw = ''.join(clean_tw)
    clean_tw = clean_tw.lower()
    # remove URLs
    clean_tw = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', clean_tw)
    clean_tw = re.sub(r'http\S+', '', clean_tw)
    # remove usernames
    clean_tw = re.sub(r'@[^\s]+', '', clean_tw)
    # remove the # in #hashtag
    clean_tw = re.sub(r'#([^\s]+)', r'\1', clean_tw)
    # remove repeated characters
    clean_tw = word_tokenize(clean_tw)
    # remove stopwords from final word list
    tw = [tw for tw in clean_tw if tw not in stopwords.words('english')]
    lem_tw = [lemmatizer.lemmatize(i) for i in tw]
    joined_stem_text = ' '.join([stemmer.stem(i) for i in lem_tw])
    return joined_stem_text


