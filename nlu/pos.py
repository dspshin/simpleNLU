#!/usr/bin/python
#-*- coding: utf-8 -*-

from konlpy.tag import Kkma, Twitter, Hannanum, Komoran
from konlpy.utils import pprint


# Kkma is a morphological analyzer and natural language processing system written in Java, developed by the Intelligent Data Systems (IDS) Laboratory at SNU.
# http://kkma.snu.ac.kr/
kkma = Kkma()

# Twitter Korean Text is an open source Korean tokenizer written in Scala, developed by Will Hohyon Ryu.
twitter = Twitter()

# JHannanum is a morphological analyzer and POS tagger written in Java, and developed by the Semantic Web Research Center (SWRC) at KAIST since 1999.
hannanum = Hannanum()

# KOMORAN is a relatively new open source Korean morphological analyzer written in Java, developed by Shineware, since 2013.
#komoran = Komoran()
# komoran은 현재 에러 발생하네. 사용안함.
# /NLU/nlu-env/lib/python3.5/site-packages/konlpy/tag/_komoran.py", line 60, in pos
#     result = self.jki.analyzeMorphs3(phrase, self.dicpath).toString()
# jpype._jexception.OutOfMemoryErrorPyRaisable: java.lang.OutOfMemoryError: GC overhead limit exceeded

def run_all(sentence):
    return {
        'kkma':kkma.pos(sentence),
        'twitter':twitter.pos(sentence),
        'hannanum':hannanum.pos(sentence)
    }
    # pprint( komoran.pos(sentence) )