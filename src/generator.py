import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import random
import string
import numpy as np
import spacy
#from nltk import data, word_tokenize
#from nltk.stem import WordNetLemmatizer
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import spacy

# cheap thread-unsafe singleton
nlp = None

class ConvoMsg:
    def __init__(self, text, speaker=None):
        self.text = text
        self.speaker = speaker
        self.doc = self.parse()

    def parse(self):
        return nlp(self.text)

    def vectorize(self):
        # spacy's default: average of token vectors.
        return self.doc.vector

class Conversation:
    def __init__(self, msgs):
        print("parsing convo, {} msgs".format(len(msgs)))
        self.msgs = list((ConvoMsg(m['text'], m['name']) for m in msgs))
        self.parsed_docs = self.parse()
        self.X = self.vectorize()

    def parse(self):
        return [nlp(msg.text) for msg in self.msgs]

    def flat_vectorize(self):
        # average conversation vectors
        return np.mean(self.vectorize(), axis=0)

    def vectorize(self):
        # array of doc vectors.
        return np.array([msg.vectorize() for msg in self.msgs])

    def get_similar_msg(self, x, n=1):
        sim = cosine_similarity(x.reshape(1,-1), self.X)[0]
        if n == 1:
            idx = np.argmax(sim)
            return self.msgs[idx]
        # TODO: n>1

    def get_similar_rsp(self, x, n=1):
        sim = cosine_similarity(x.reshape(1,-1), self.X)[0]
        if n == 1:
            idx = np.argmax(sim)
            #print("Argmax id is {}, len is {}".format(idx, len(self.msgs)))
            speaker = self.msgs[idx].speaker
            while (idx < (len(self.msgs))) and speaker is self.msgs[idx].speaker:
                idx += 1
            if idx < len(self.msgs):
                return self.msgs[idx]
            else:
                return None
        # TODO: n>1

    def test(self):
        print(self.msgs)
        print("vectorize: shape:{}".format(self.vectorize().shape))
        print("flat_vect shape:{}".format(self.flat_vectorize().shape))


def load_spacy_model():
    global nlp
    print("Loading spacy lang model... ", end="")
    nlp = spacy.load('en_core_web_md')
    print("done")

def read_corpus(path):
    load_spacy_model()
    # detect type, open
    f = open(path, 'r', errors='ignore')
    if path.split('.')[-1] == "json":
        convos = []
        raw_convos = json.load(f)
        for raw_convo in raw_convos:
            convo = Conversation(raw_convo)
            convos.append(convo)
            #msgs = [msg['text'] for msg in convo]
            #flat_msgs += msgs
        #convos[0].test()
        return convos
    else:
        raw = f.read()
        doc = nlp(raw)
        #sent_detector = data.load('tokenizers/punkt/english.pickle')
        #return sent_detector.tokenize(raw)
        return [sent.text for sent in doc.sents]


class TextGenerator:
    def __init__(self, corpus, verbosity=0):
        load_spacy_model()
        self.greeting_responses = [
            "hi", "hey", "*nods*", "hi there", "hello", 
            "I am glad! You are talking to me"
        ]
        self.verbosity = verbosity
        self.remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
        # WordNet is a semantically-oriented dictionary of English included in NLTK.
        #self.lemmer = WordNetLemmatizer()
        self.convos = read_corpus(corpus)
        self.flat = True
        self.X = self.vectorize_conversations(self.convos, flat=self.flat)
        #self.vectorizer = TfidfVectorizer(tokenizer=self.tokenize, stop_words='english')
        #self.tfidf = self.vectorizer.fit_transform(self.sentences)

    def tokenize(self, sentence):
        tokens = word_tokenize(sentence)
        tokens = self.normalize(tokens)
        return self.lemmatize(tokens)

    def vectorize_conversations(self, convos, flat=False):
        self.X_convo = []
        if flat:
            size = 0
            for i,c in enumerate(convos):
                self.X_convo = self.X_convo + [i for _ in range(len(c.X))]
                size += len(c.X)
            X = np.zeros((size, convos[0].X.shape[1]))
            size = 0
            for c in convos:
                X[size:size + len(c.X)] = c.X
                size += len(c.X)
        else:
            X = [c.flat_vectorize() for c in convos]
        return np.array(X)

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

    def get_similar_conversation(self, x, n=1):
        sim = cosine_similarity(x.reshape(1,-1), self.X)[0]
        if n == 1:
            idx = np.argmax(sim)
            try:
                return self.convos[idx]
            except:
                return None

    def get_similar_response(self, x):
        resp = "Sorry, not sure how to respond to that."
        if self.flat:
            sim = cosine_similarity(x.reshape(1,-1), self.X)[0]
            idx = np.argmax(sim)
            try:
                convo = self.convos[self.X_convo[idx]]
                msg = convo.get_similar_rsp(x)
                resp = msg.text
            except:
                if self.verbosity > 0:
                    print("Couldn't find a similar resp in convo")
        else:
            convo = self.get_similar_conversation(x)
            if convo:
                #msg = convo.get_similar_msg(x)
                try:
                    msg = convo.get_similar_rsp(x)
                    resp = msg.text
                except:
                    if self.verbosity > 0:
                        print("Couldn't find a similar resp in convo")
            elif self.verbosity > 0:
                print("Couldn't find a similar conversation")
        return resp

    def vectorize_response(self, user_input):
        return ConvoMsg(user_input).vectorize()


    def query_response(self, user_input):
        response = "Sorry, I don't understand you"

        #x = self.vectorizer.transform([user_input])
        #vals = cosine_similarity(self.tfidf, x)
        #idx = np.argmax(vals, axis=0)[0]
        #match_val = sum(vals[idx])
        #if match_val > 0:
        #    response = self.sentences[idx]
        x = self.vectorize_response(user_input)
        if self.verbosity > 0:
            print(sum(x))
        response = self.get_similar_response(x)
        return response


