#!/usr/bin/python
#-*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import pandas as pd
from konlpy.tag import Mecab
from matplotlib.image import imread, imsave
import matplotlib.pyplot as plt
from gensim.models import word2vec
import os, sys


DEBUG = True

# 300으로 가도 큰 차이없음.
# 200 : accuracy 0.946, 0.964(epoch 50)
# 100 : accuracy 0.927, 0.953(epoch 50)
vector_size = 200

# encode max len
#  7 : 0.965
#  10 : 0.965
encode_length = 7
#label_size = 5

# onehot과 wordvec 사용시의 learning rate는 전혀 다름.
learning_rate = 1e-3

#embed_type = "onehot"
embed_type = "wordvec"

# Choose multi test
filter_type = "multi"

#filter_sizes = [1,2,3,4,1,2,3,4,1,2,3,4] #0.965
filter_sizes = [1,2,3,4,5,1,2,3,4,5,1,2,3,4,5] #0.971
#filter_sizes = [1,2,3,4,5,2,3,4,5,3,4,5,4,5,2,3,4,5] #0.972
#filter_sizes = [1,2,3,4,5,2,3,4,5,2,3,4,5] #0.969
num_filters = len(filter_sizes)


def log(*args):
	if DEBUG:
		print(*args)

def train_vector_model(data_list):
    mecab = Mecab('/usr/local/lib/mecab/dic/mecab-ko-dic')
    str_buf = data_list['encode']
    pos1 = mecab.pos(''.join(str_buf))
    log("pos1:", pos1)
    pos2 = ' '.join(list(map(lambda x : '\n' if x[1] in ['SF'] else x[0], pos1))).split('\n')
    log("pos2:", pos2)

    morphs = list(map(lambda x : mecab.morphs(x) , pos2))

    # morphs = []
    # for sentence in pos2:
    #     for pos in mecab.pos(sentence):
    #         morphs.append('/'.join(pos))

    log("morphs:", morphs)

    model = word2vec.Word2Vec(size=vector_size, window=5, min_count=5, sg=1) #skipgram이 훨씬 성능이 우세

    # data.tsv만으로는 워드벡터가 빈약한거 같아.
    model.build_vocab(morphs)

    # 실험결과 : epochs - accuracy
    #  10 - 0.900
    #  20 - 0.943
    #  30 - 0.947
    #  50 - 0.953
    #  100 - 0.954
    model.train(morphs, total_examples=len(str_buf), epochs=50)
    return model

def load_csv(data_path):
    df_csv_read = pd.DataFrame(data_path)
    return df_csv_read

def extract_features(text):
    mecab = Mecab('/usr/local/lib/mecab/dic/mecab-ko-dic')
    res=[]
    raw_pos = mecab.pos(text)
    log('pos:', raw_pos)
    for pos in raw_pos:
        # https://docs.google.com/spreadsheets/d/1OGAjUvalBuX-oZvZ_-9tEfYD2gQe7hTGsgUpiiBSXI8/edit#gid=0
        # 명사와 동사면 충분할듯?
        # if pos[1] in ['NNG', 'NNP', 'VV', 'VX']:
        # 	res.append(pos[0])
        # elif pos[1].startswith('V'):
        # 	res.append(pos[0])

        res.append(pos[0]) # 그냥 모두 다 넣자.
        #res.append( '/'.join(pos) )
    return res

def embed(data) :
	inputs = []
	labels = []
	for encode_raw in data['encode']:
		encode_raw = extract_features(encode_raw)
		encode_raw = list(map(lambda x : encode_raw[x] if x < len(encode_raw) else '#', range(encode_length)))
		log('morphs:', encode_raw)

		if(embed_type == 'onehot') :
			bucket = np.zeros(vector_size, dtype=float).copy()
			input = np.array(list(map(lambda x : onehot_vectorize(bucket, x) if x in model.wv.index2word else np.zeros(vector_size,dtype=float) , encode_raw)))
		else :
			input = np.array(list(map(lambda x : model[x] if x in model.wv.index2word else np.zeros(vector_size,dtype=float) , encode_raw)))
		inputs.append(input.flatten())

	for decode_raw in data['decode']:
		# onehot
		label = np.zeros(label_size, dtype=float)
		np.put(label, decode_raw, 1)
		labels.append(label)
	return inputs, labels

def onehot_vectorize(bucket, x):
    np.put(bucket, model.wv.index2word.index(x),1)
    return bucket

def inference_embed(data) :
    encode_raw = extract_features(data)
    encode_raw = list(map(lambda x : encode_raw[x] if x < len(encode_raw) else '#', range(encode_length)))
    log( 'morphs:', encode_raw )
    if(embed_type == 'onehot') :
        bucket = np.zeros(vector_size, dtype=float).copy()
        input = np.array(list(map(lambda x : onehot_vectorize(bucket, x) if x in model.wv.index2word else np.zeros(vector_size,dtype=float) , encode_raw)))
    else :
        input = np.array(list(map(lambda x : model[x] if x in model.wv.index2word else np.zeros(vector_size,dtype=float) , encode_raw)))
    return input

def create_m_graph(train=True):
    # placeholder is used for feeding data.
    x = tf.placeholder("float", shape=[None, encode_length * vector_size], name = 'x')
    y_target = tf.placeholder("float", shape=[None, label_size], name = 'y_target')

    # reshape input data
    x_image = tf.reshape(x, [-1,encode_length,vector_size,1], name="x_image")
    # Keeping track of l2 regularization loss (optional)
    l2_loss = tf.constant(0.0)

    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            # Convolution Layer
            filter_shape = [filter_size, vector_size, 1, num_filters]
            W_conv1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b_conv1 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

            conv = tf.nn.conv2d(
                x_image,
                W_conv1,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")

            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b_conv1), name="relu")
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, encode_length - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            pooled_outputs.append(pooled)

    # Combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    # Add dropout
    keep_prob = 1.0
    if(train) :
        keep_prob = tf.placeholder("float", name="keep_prob")
        h_pool_flat = tf.nn.dropout(h_pool_flat, keep_prob)

    # Final (unnormalized) scores and predictions
    W_fc1 = tf.get_variable(
        "W_fc1",
        shape=[num_filters_total, label_size],
        initializer=tf.contrib.layers.xavier_initializer())
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[label_size]), name="b")
    l2_loss += tf.nn.l2_loss(W_fc1)
    l2_loss += tf.nn.l2_loss(b_fc1)
    y = tf.nn.xw_plus_b(h_pool_flat, W_fc1, b_fc1, name="scores")
    predictions = tf.argmax(y, 1, name="predictions")

    # CalculateMean cross-entropy loss
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_target)
    cross_entropy = tf.reduce_mean(losses)

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    # Accuracy
    correct_predictions = tf.equal(predictions, tf.argmax(y_target, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    return accuracy, x, y_target, keep_prob, train_step, y, cross_entropy, W_conv1

def show_layer(weight_list) :
    if(filter_type == 'multi') :
        show = np.array(weight_list).reshape(num_filters, filter_sizes[np.argmax(filter_sizes)], vector_size)
        for i, matrix in enumerate(show) :
            fig = plt.figure()
            plt.imshow(matrix)
        plt.show()
    else :
        show = np.array(weight_list).reshape(32, 2, 2)
        for i, matrix in enumerate(show) :
            fig = plt.figure()
            plt.imshow(matrix)
        plt.show()

def get_test_data():
    train_data, train_label = embed(load_csv(train_data_list))
    test_data, test_label = embed(load_csv(train_data_list))
    return train_label, test_label, train_data, test_data

def predict(test_data) :
    try :
        # reset Graph
        tf.reset_default_graph()
        # Create Session
        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth =True)))
        # create graph
        if(filter_type == 'single') :
            _, x, _, _, _, y, _, _ = create_s_graph(train=False)
        else :
            _, x, _, _, _, y, _, _ = create_m_graph(train=False)

        # initialize the variables
        sess.run(tf.global_variables_initializer())

        # set saver
        saver = tf.train.Saver()

        # Restore Model
        path = './model/'
        if os.path.exists(path):
            saver.restore(sess, path)
            log("model restored")

        # training the MLP
        #print("input data : {0}".format(test_data))
        y = sess.run([y], feed_dict={x: np.array([test_data])})
        log("result : {0}".format(y))
        log("result : {0}".format(np.argmax(y)))

    except Exception as e :
        raise Exception ("error on training: {0}".format(e))
    finally :
        sess.close()

    return np.argmax(y)


if __name__=='__main__':
    df = pd.DataFrame.from_csv('data.tsv', sep='\t', header=None)

    # speech_act distinct count check
    count = df.groupby(0).count()
    print( count )
    y_size = count.size

    train_data_list = {
        'encode':[],
        'decode':[]
    }
    y_dic = {}
    y_inv_list = []
    for index, row in df.iterrows():
        x = row[1]
        y = index

        # $제거 - 여기서는 별로 쓸모없으니
        x = x.replace('$', '')

        # ,로 여러개가 입력되면 뒤에걸로 mapping하자
        if y.find(',')>-1:
            y = y.split(',')[-1]
        print(x, y)
        try:
            y_dic[y] += 1
        except KeyError:
            y_dic[y] = 1
            y_inv_list.append(y)

        train_data_list['encode'].append(x)
        train_data_list['decode'].append(y_inv_list.index(y))

    label_size = len(y_inv_list)
    print( y_inv_list )

    # train_data_list = {
    # 	'encode' : ['판교에 오늘 피자 주문해줘',
    # 		'오늘 날짜에 호텔 예약 해줄레',
    # 		'바이킹스와프 예약할래',
    # 		'내일 집에서 쉴래',
    # 		'모래 날짜에 판교 여행 정보 알려줘'],
    # 	'decode' : [0, 1, 2, 3, 4]
    # }
    # log( train_data_list.get('encode') )

    model = train_vector_model(train_data_list)
    log(model)

    try:
        # get Data
        labels_train, labels_test, data_filter_train, data_filter_test = get_test_data()
        # log(labels_train)
        # log(labels_test)
        # log(data_filter_train)
        # log(data_filter_test)

        # reset Graph
        tf.reset_default_graph()

        # Create Session
        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth =True)))
        # create graph
        if(filter_type == 'single') :
            accuracy, x, y_target, keep_prob, train_step, y, cross_entropy, W_conv1 = create_s_graph(train=True)
        else :
            accuracy, x, y_target, keep_prob, train_step, y, cross_entropy, W_conv1 = create_m_graph(train=True)

        # set saver
        saver = tf.train.Saver(tf.all_variables())
        # initialize the variables
        sess.run(tf.global_variables_initializer())

        # training the MLP
        for i in range(500):
            sess.run(train_step, feed_dict={x: data_filter_train, y_target: labels_train, keep_prob: 0.7})
            if i%10 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={x:data_filter_train, y_target: labels_train, keep_prob: 1})
                log("step %d, training accuracy: %.3f"%(i, train_accuracy))

        # for given x, y_target data set
        log("test accuracy: %g"% sess.run(accuracy, feed_dict={x:data_filter_test, y_target: labels_test, keep_prob: 1}))

        # show weight matrix as image
        weight_vectors = sess.run(W_conv1, feed_dict={x: data_filter_train, y_target: labels_train, keep_prob: 1.0})
        #show_layer(weight_vectors)

        # Save Model
        path = './model/'
        if not os.path.exists(path):
            os.makedirs(path)
            log("path created")
        saver.save(sess, path)
        log("model saved")
    except Exception as e:
        raise Exception ("error on training: {0}".format(e))
    finally:
        sess.close()

    while True:
        text = input("> ")
        if not text:
            break
        y_index = predict(np.array(inference_embed(text)).flatten())
        print( y_inv_list[y_index] )
