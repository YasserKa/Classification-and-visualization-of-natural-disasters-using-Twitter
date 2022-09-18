# %%
import string
import re
import nltk

def clean_text(text):
    nltk.download('stopwords')
    stopword = nltk.corpus.stopwords.words('english')

    # Remove URLs/Mentions/Hashtags/new lines
    text = re.sub("((www.[^s]+)"
            "|(https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*))"
            "|(@[a-zA-Z0-9]*)"
            "|(#[a-zA-Z0-9]*)"
            "|\n)", ' ', text)
    # Remove stopwords
    text = " ".join([word for word in str(text).split() if word not in stopword])
    # Remove punctuation
    text = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    # Remove Emojis
    emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text) 
    # Lower case
    text = text.lower()

    return text
