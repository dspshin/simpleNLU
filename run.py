#!/usr/bin/python
#-*- coding: utf-8 -*-

import intent
import telepot
import sys
import time
from pprint import pprint
import numpy as np
from intent import log
import dm

def handleMessage(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)
    if content_type != 'text':
        sendMessage(chat_id, '난 텍스트 이외의 메시지는 처리하지 못해요.')
        return

    try:
        name = msg["from"]["last_name"] + msg["from"]["first_name"]
    except:
        name = ""

    text = msg['text']
    log( chat_id, name, text )

    nlu_result = intent.parse(text)
    pprint( nlu_result )

	#bot.sendMessage(chat_id, msg)

def handlePrompt(msg):
	print('>', msg)

def handleInput():
	sentence = input('>>> ')
	nlu_result = intent.parse( sentence )
	return nlu_result

if __name__=='__main__':

    # arg가 있으면 토큰이 있다고 생각하고, 없으면 그냥 input으로 동작.
    if len(sys.argv)>1:
        TOKEN = sys.argv[1]
        print('received token :', TOKEN)

        bot = telepot.Bot(TOKEN)
        pprint( bot.getMe() )

        bot.message_loop(handleMessage)
        print('Listening...')
        while 1:
            time.sleep(10)
    else:
	    dm.cbPrompt = handlePrompt
	    dm.cbInput = handleInput
	    dm.execution_phase()