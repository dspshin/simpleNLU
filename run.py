#!/usr/bin/python
#-*- coding: utf-8 -*-

from nlu import rule
from konlpy.utils import pprint


if __name__=='__main__':
    while True:
        sentence = input('> ')
        print( '입력:', sentence )

        nlu_result = rule.check( sentence )
        pprint( nlu_result )