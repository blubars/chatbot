import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import random
import string
import numpy as np
from nltk import data, word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

def read_corpus(path):
    # detect type, open
    f = open(path, 'r', errors='ignore')
    if path.split('.')[-1] == "json":
        flat_msgs = []
        convos = json.load(f)
        for convo in convos:
            msgs = [msg['text'] for msg in convo]
            flat_msgs += msgs
        return flat_msgs
    else:
        raw = f.read()
        sent_detector = data.load('tokenizers/punkt/english.pickle')
        return sent_detector.tokenize(raw)


class TextGenerator:
    def __init__(self, corpus):
        print("Learning how to speak...  ", end="")
        self.greeting_responses = [
            "hi", "hey", "*nods*", "hi there", "hello", 
            "I am glad! You are talking to me"
        ]
        self.remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
        # WordNet is a semantically-oriented dictionary of English included in NLTK.
        self.lemmer = WordNetLemmatizer()
        #sentences = self.preprocess(corpus)
        #self.train(X)
        self.sentences = read_corpus(corpus)
        self.vectorizer = TfidfVectorizer(tokenizer=self.tokenize, stop_words='english')
        self.tfidf = self.vectorizer.fit_transform(self.sentences)
        print("done!")

    def preprocess(corpus):
        self.sentences = []
        #raw = f.read().lower()
        sent_detector = data.load('tokenizers/punkt/english.pickle')
        # incremental training of tokenizer? for abbreviations.
        #PunktTrainer(train_text=corpus)
        #sent_tokens = nltk.sent_tokenize(raw) # converts to list of sentences
        sent_detector = data.load('tokenizers/punkt/english.pickle')
        sentences = sent_detector.tokenize(corpus)
        for S in sentences:
            tokens = word_tokenize(S)
            tokens = self.normalize(tokens)
            tokens = self.lemmatize(tokens)
            self.sentences.append((S, tokens))
            print("{}\n  --> {}".format(S, tokens))
        #word_tokens = nltk.word_tokenize(raw) # converts to list of words
        vect = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
        vect.fit_transform(sent_tokens)
        return self.sentences

    def tokenize(self, sentence):
        tokens = word_tokenize(sentence)
        tokens = self.normalize(tokens)
        return self.lemmatize(tokens)

    def lemmatize(self, tokens):
        return [self.lemmer.lemmatize(token) for token in tokens]

    def normalize(self, tokens):
        norm_tokens = []
        for token in tokens:
            t = token.lower().translate(self.remove_punct_dict)
            if len(t) > 0:
                norm_tokens.append(t)
        return norm_tokens

    # types: thanks, greeting
    def get_response(self, resp_type=None, user_input=None):
        if resp_type is 'yourewelcome':
            response = self.get_yourewelcome()
        elif resp_type is 'greeting':
            response = self.get_greeting()
        elif resp_type is 'query':
            response = self.query_response(user_input)
        else:
            response = "Not sure how to respond to that. ({})".format(resp_type)
        return response

    def get_greeting(self):
        return random.choice(self.greeting_responses)

    def get_yourewelcome(self):
        return random.choice(("You're welcome", "No problem!", "Sure thing."))

    def query_response(self, user_input):
        response = "Sorry, I don't understand you"

        x = self.vectorizer.transform([user_input])
        vals = cosine_similarity(self.tfidf, x)
        idx = np.argmax(vals, axis=0)[0]
        match_val = sum(vals[idx])
        if match_val > 0:
            response = self.sentences[idx]

        return response

if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('wordnet')

