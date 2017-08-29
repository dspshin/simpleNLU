#!/usr/bin/python
#-*- coding: utf-8 -*-

from nlu import pos
from konlpy.utils import pprint

TAGGER_MAPPING_TABLE = {
    'kkma':'k',
    'twitter':'t',
    'hannanum':'h'
}

RULES = [
    # tragger, command, syntax, output
    # command - c:contains
    ('k', 'c',  '에어컨|NNG', {'speech_act':'aircon'})
]


def check(sentence):
    pos_res = pos.run_all(sentence)
    pprint( pos_res )

    res = {}

    # 여기서 룰첵을 한 결과를 반환해야 함.
    # 알맞은 형태소분석기별로 문법을 따로 적용하는게 나을듯함.

    for tagger, tag_res in pos_res.items():
        tagger = TAGGER_MAPPING_TABLE.get(tagger, 'unknown')
        if tagger=='unknown':
            res['speech_act']='not_understanding'
            return res

        # join tag_res
        morps = []
        for morp in tag_res:
            morps.append( '|'.join(morp) )

        # 이하는 rule이 많아지면 혹은 db화되면 효율적으로 혹은 select로 변경되어야 함.
        for rule in RULES:
            # tagger가 동일해야함.
            r_tag, command, syntax, output = rule
            if tagger!=r_tag:
                continue

            # syntax check
            # contain인 경우
            for morp in morps:
                if morp == syntax and command == 'c':
                    res = {**res, **output} #join


    # 결과가 없으면
    if not res:
        res['speech_act']='not_understanding'

    res['sent'] = sentence
    return res