#!/usr/bin/python
#-*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import pandas as pd
from konlpy.tag import Kkma, Twitter, Hannanum, Komoran
from konlpy.tag import Mecab
from matplotlib.image import imread, imsave 
import matplotlib.pyplot as plt  
from gensim.models import word2vec


# Kkma is a morphological analyzer and natural language processing system written in Java, developed by the Intelligent Data Systems (IDS) Laboratory at SNU.
# http://kkma.snu.ac.kr/
kkma = Kkma()

# Twitter Korean Text is an open source Korean tokenizer written in Scala, developed by Will Hohyon Ryu.
twitter = Twitter()

# JHannanum is a morphological analyzer and POS tagger written in Java, and developed by the Semantic Web Research Center (SWRC) at KAIST since 1999.
hannanum = Hannanum()

if __name__=='__main__':

	# raw_data = []

	# with open('data.tsv') as f:
	# 	for item in f.readlines():
	# 		parsed = item.split('\t')
	# 		raw_data.append( (parsed[1], parsed[0]) )

	# print( 'data.tsv row size:', len(raw_data) )


	df = pd.DataFrame.from_csv('data.tsv', sep='\t', header=None)
	
	# speech_act distinct count check
	count = df.groupby(0).count()
	print( count )
	y_size = count.size

	for index, row in df.iterrows():
		x = row[1]
		y = index
		print(x, y)
		print(kkma.pos(x))
		print(twitter.pos(x))
		print(hannanum.pos(x))
		input()
		# 3개를 다 쓸 필요는 없을듯.
		# 명사와 무언가를 더 뽑아서 벡터화 해야할듯.
		

	# one hot encoding

