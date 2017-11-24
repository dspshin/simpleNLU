#!/usr/bin/python
#-*- coding: utf-8 -*-

import json
from pprint import pprint

INPUT_JSON = 'simple.json'


#dts loading / parsing
f = open(INPUT_JSON, 'rt')
raw_json = f.read()
f.close()

dts = json.loads(raw_json)
#print('loaded json:', dts)

# parse concepts
for concept in dts['concepts']:
    print(concept)

# init dialog stack
dialog_stack = []
dialog_stack.append( dts['children'][0] ) # add root agency

# build expectation agenda


# prompt callback function
def cbPrompt(msg):
    pass
# input callback function
def cbInput():
    return ""

def inform(agent):
    print('processing inform agent:', agent)
    cbPrompt(agent['prompt'])

def request(agent):
    print('processing request agent:', agent)
    cbPrompt(agent['prompt'])

    # go to input phase
    nlu_result = cbInput()
    pprint(nlu_result)

    # concept mapping


def execute(agent):
    print('processing execute agent:', agent)

agentRunner = {
    'inform':inform,
    'request':request,
    'execute':execute
}

def execute_agent(agent):
    agentRunner.get(agent['agentType'], lambda x:"agent type mismatched")(agent)

"""execute agent on top of the stack"""
def execute_top_agent():
    global dialog_stack
    agent = dialog_stack.pop() # eliminate completed agents from stack

    type = agent['type']
    print('try to execute:', agent['name'], type)

    if type == 'agency':
        # if agency, check subagents and append reversely
        dialog_stack += agent['children'][::-1]
    elif type == 'agent':
        execute_agent(agent)
    else:
        print('Unknown type !')

# execution phase
def execution_phase():

    while len(dialog_stack)>0:
        # execute agent on top of the stack
        execute_top_agent()

        # error handling
        # push focus claiming

    print('dialog_stack is empty.')
