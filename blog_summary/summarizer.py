import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from heapq import nlargest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Ensure you have the required NLTK data
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def get_article_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    article_text = ''
    for p in paragraphs:
        article_text += p.get_text()
    return article_text

def summarize(text, num_sentences=5):
    # Tokenize sentences and words
    sentence_tokens = sent_tokenize(text)
    word_tokens = word_tokenize(text.lower())

    # Calculate word frequencies and filter out stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    word_frequencies = FreqDist(w for w in word_tokens if w not in stopwords)

    # Rank sentences based on frequency of words in the text
    ranking = {}
    for i, sentence in enumerate(sentence_tokens):
        ranking[i] = sum([word_frequencies[word.lower()] for word in word_tokenize(sentence) if word.lower() not in stopwords])

    # Select the top sentences
    top_sentence_indices = nlargest(num_sentences, ranking, key=ranking.get)
    summary_sentences = [sentence_tokens[index] for index in sorted(top_sentence_indices)]

    return summary_sentences

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Tokenize the text and remove stop words
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalnum() and token.lower() not in stop_words]

    # Lemmatize the tokens
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return ' '.join(lemmatized_tokens)

def lsa_summarization(text, num_sentences=5):
    # Preprocess the text
    preprocessed_text = preprocess_text(text)

    # Tokenize sentences and create the TF-IDF matrix
    sentence_tokens = sent_tokenize(text)
    preprocessed_sentence_tokens = sent_tokenize(preprocessed_text)

    if len(preprocessed_sentence_tokens) <= 1:
        return preprocessed_sentence_tokens

    vectorizer = TfidfVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(preprocessed_sentence_tokens)

    # Perform LSA using Truncated SVD
    lsa = TruncatedSVD(n_components=1, random_state=42)
    lsa.fit(matrix)

     # Get the most important sentences
    scores = lsa.components_[0]

    # Limit the num_sentences to the actual number of sentences
    num_sentences = min(num_sentences, len(preprocessed_sentence_tokens))

    idx = scores.argsort()[-num_sentences:][::-1]
    summary_sentences = [sentence_tokens[i] for i in idx]

    return summary_sentences[:num_sentences]







