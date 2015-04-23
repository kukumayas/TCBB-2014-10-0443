#coding:utf8
from GDEPhandler import transGDEP2graphFormat_v2
from Graph2vec import Graph2Vec




'''
First, Parse text8_splited using GDEP
Second, run following code to train word vector using NNGM.
'''

transGDEP2graphFormat_v2(['text8_dep'], 'text8_for_NNGM', False, merge=False)
model = Graph2Vec('text8_for_NNGM', size=400, workers=1, iter=1)
model.save_word2vec_format('GraphVector_d400.bin', binary=True)

'''
Last, evaluate GraphVector_d400.bin using compute-accuracy provided by Word2Vec
'''