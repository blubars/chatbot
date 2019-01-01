import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
from argparse import ArgumentParser
from understander import Understander
from generator import TextGenerator
from time import sleep

# initial version based on:
# https://medium.com/analytics-vidhya/building-a-simple-chatbot-in-python-using-nltk-7c8c8215ac6e

class DialogContext:
    # state track/subject change; track conversation topics for followups
    def __init__(self):
        self.acts = []

    def update(self, user_text="", chatbot_text=""):
        self.acts.append((user_text, chatbot_text))

    def __str__(self):
        s = ""
        for i,act in enumerate(self.acts):
            s += "  [{}] {} : {}\n".format(i, act[0], act[1])
        return s

# keep track of nested contexts
class ContextStack:
    def __init__(self, capacity=10):
        self.contexts = []
        self.capacity = capacity

    def push(self, context):
        if len(self.contexts) < self.capacity:
            self.contexts.append(context)
            return True
        else:
            return False

    def pop(self):
        if len(self.contexts) > 0:
            return self.contexts.pop()
        else:
            return None

class Chatbot:
    # dialog manager and task manager.
    def __init__(self, corpus, name="HARRY BOTTER", verbosity=0):
        self.name = name.upper()
        self.history = []
        self.context_stack = ContextStack()
        self.commands = ("help")
        print("Learning how to understand... ", end="")
        self.understander = Understander(verbosity=verbosity)
        print("done")
        print("Learning how to speak... ", end="")
        self.generator = TextGenerator(corpus, verbosity=verbosity)
        print("done")
        self.verbosity = verbosity

    def run(self):
        # initial greeting
        print("\n------------------------------")
        print("{}: Hi! I will answer your queries about shit you should know but clearly don't. If you want to exit, type 'Bye'!".format(self.name))
        alive = True
        # loop until told to frack off
        while(alive):
            # let user pick context.
            #context = DialogContext()
            user_input = self.get_user_input()
            sleep(0.2)
            if self.is_exit_request(user_input):
                alive = False
                print("{}: Bye! Take care.".format(self.name))
            else:
                response = self.handle_user_input(user_input)
                print("{}: {}".format(self.name, response))
                self.history.append((user_input, response))
        if self.verbosity > 0:
            self.print_history()

    def print_history(self):
        print("Printing conversation history")
        for i,C in enumerate(self.history):
            print("[{}]: '{}'\n       --> '{}'".format(i, C[0], C[1]))

    def get_user_input(self):
        print(" > ", end='')
        user_input = input()
        return user_input.lower()

    def is_exit_request(self, user_input):
        exit_commands = ('bye', 'quit')
        return user_input.lower() in exit_commands

    def handle_user_input(self, user_input):
        # map input type to response type
        input_type = self.understander.get_input_type(user_input)
        resp_type = None
        if input_type is 'thanks':
            resp_type = 'yourewelcome'
        elif input_type is 'greeting':
            resp_type = 'greeting'
        elif user_input in self.commands:
            self.handle_command(user_input)
        elif input_type is 'query':
            resp_type = 'query'
        return self.generator.get_response(resp_type, user_input)

    def handle_command(self, command):
        if command is 'help':
            self.display_help()

    def display_help(self):
        print("{}: Help - ".format(self.name))
        print("   - EXIT: 'bye' or 'quit'")


if __name__ == "__main__":
    parser = ArgumentParser(prog="chatbot")
    parser.add_argument('corpus', help='corpus to train chatbot with')
    parser.add_argument('--verbosity', '-v', type=int, help='verbosity', default=0)
    args = parser.parse_args()

    bot = Chatbot(args.corpus, verbosity=args.verbosity)
    bot.run()

