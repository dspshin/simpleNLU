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

# execution plan
exec_stack = []
exec_stack.append( dts['children'][0] ) # add root agency


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

    # show dialog stack
    print('dialog stack:', [ dialog['name'] for dialog in dialog_stack ])

    # concept mapping
    # build expectation agenda


def execute(agent):
    print('processing execute agent:', agent)

    # do api call or sth

agentRunner = {
    'inform':inform,
    'request':request,
    'execute':execute
}

def execute_agent(agent):
    agentRunner.get(agent['agentType'], lambda x:"agent type mismatched")(agent)

"""execute agent(cy) on top of the stack"""
def execute_top_agent():
    global exec_stack
    agent = exec_stack.pop()

    dialog_stack.append(agent)

    type = agent['type']
    print('try to execute:', agent['name'], type)

    if type == 'agency':
        # if agency, check subagents and append reversely
        exec_stack += agent['children'][::-1]
        execute_top_agent()
    elif type == 'agent':
        execute_agent(agent)
    else:
        print('Unknown type !')

    # eliminate completed agents from stack
    dialog_stack.pop()

# execution phase
def execution_phase():

    while len(exec_stack)>0:
        # execute agent on top of the stack
        execute_top_agent()

        # error handling
        # push focus claiming

    print('stack is empty.')
