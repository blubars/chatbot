
class Understander:
    def __init__(self, verbosity=1):
        self.greeting_inputs = ("hello", "hi", "greetings", "sup", "what's up", "hey")
        self.verbosity = verbosity
        # WordNet is a semantically-oriented dictionary of English included in NLTK.
        #self.lemmer = WordNetLemmatizer()
        #sentences = self.preprocess(corpus)
        #self.train(X)

    def train(self, X):
        # incremental training
        # TODO
        pass

    def process_input(self, user_input):
        input_type = self.get_input_type(user_input)

    def get_input_type(self, user_input):
        # TODO: replace with real model rather than rule-based.
        input_type = 'thanks'
        if user_input == 'thanks' or user_input == 'thank you':
            input_type = 'thanks'
        elif self.is_greeting(user_input):
            input_type = 'greeting'
        else:
            input_type = 'query'
        return input_type

    def is_greeting(self, user_input):
        return user_input in self.greeting_inputs


