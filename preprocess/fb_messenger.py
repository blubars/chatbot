# convert FB messenger json logs into corpus for use in chatbot training
# strips names, replaces with 'A' and 'B'
    # e.g., {
    #  "sender_name": "First Last",
    #  "timestamp_ms": 1539825241822,
    #  "content": "Sounds good man will see you then",
    #  "type": "Generic"
    #},

import sys
import os
import json
import re

CONVO_MS_THRESH = 6*60*60*1000 # 6hrs*60min/hr*6sec/min*1000ms/sec
MAX_PARTICIPANTS = 2
MIN_CONVO_LEN = 2

def write_convos(fname, convo):
    with open(fname, 'w') as f:
        json.dump(convo, f, indent=0)

def visit_dir(path, convos, conv_limit):
    print("Visiting {}".format(path))
    for child in os.scandir(path):
        if len(convos) >= conv_limit:
            break
        elif child.is_dir():
            convos = visit_dir(child.path, convos, conv_limit)
        elif child.is_file() and re.search('\.json$', child.path):
            convos += process_file(child.path)
    print("Processed {} conversations".format(len(convos)))
    return convos

def process_file(fname):
    print("processing file {}".format(fname))
    convos = []
    with open(fname, 'r') as f:
        data = json.load(f)
        names = parse_participants(data)
        if len(names) <= MAX_PARTICIPANTS:
            anon_ids = get_anonymous_name_dict(names)
            convos = parse_messages(data, aliases=anon_ids)
    return convos

def parse_participants(data):
    # return list of people (strings)
    return [p['name'] for p in data['participants']]

def get_anonymous_name_dict(people):
    start_id = ord('A')
    ids = [chr(char) for char in range(start_id, start_id + len(people))]
    return {p:v for (p,v) in zip(people, ids)}

def parse_messages(data, aliases=None, filter_type=None):
    # return list of message txt
    if filter_type:
        f = lambda msg: msg['type'] is filter_type
    else:
        f = lambda msg: True

    convos = []
    msgs = []
    prev_ts = 0
    for msg in reversed(list(filter(f, data['messages']))):
        ts = msg['timestamp_ms']
        if (ts - prev_ts) > CONVO_MS_THRESH:
            # new conversation
            if len(msgs) >= MIN_CONVO_LEN:
                convos.append(msgs)
                msgs = []
        prev_ts = ts

        try:
            text = msg['content']
            person = aliases[msg['sender_name']]
            msgs.append({'name':person, 'text':text})
        except:
            pass
    if len(msgs) >= MIN_CONVO_LEN:
        convos.append(msgs)
    return convos


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Error: usage: fb_messenger.py <fb_messenger_root_dir> <output_file> <conversation limit>")
        sys.exit(1)
    elif len(sys.argv) < 4:
        conv_limit = 100
    else:
        conv_limit = sys.argv[3]

    # recursively parse all the messages in the given directory
    msg_dir = sys.argv[1]
    outfile = sys.argv[2]
    convos = visit_dir(msg_dir, [], conv_limit)
    write_convos(outfile, convos)
    print("Done!")

