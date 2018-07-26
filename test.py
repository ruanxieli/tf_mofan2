# -*- coding: utf-8 -*-
# @Time    : 2018/6/6 下午9:20
# @Author  : Xieli Ruan
# @Site    : 
# @File    : test.py
# @Software: PyCharm

def randSeq(a,m):
    Xn=1
    i=1
    while True:
        Xn= (a*Xn)%m
        print('%d: %d'  % (i,Xn))
        if Xn==1:
            break
        i=i+1
    return 1


#
# randSeq(6,13)
# randSeq(7,13)


def emod(a,p):
    for i in range(p):
        print('%d ^ %d mod %d=%d' %(a, i, p,pow(a, i, p)))

emod(3,17)