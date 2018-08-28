import argparse
import nltk
import numpy as np


def get_args(argv=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=2000)
    parser.add_argument('--n_hidden', type=int, default= 128)

    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0: raise SystemExit('Unknown argument: {}'.format(unparsed))
    

    return parser.parse_args(argv)

class BLEUcalculator():
	#format
	#hypo = ['It', 'is', 'a', 'cat', 'at', 'room']
	#ref = ['It', 'is', 'a', 'cat', 'inside', 'the', 'room']
	#self.pair = [[hypothesis, reference], [hypothesis,reference], ....]
	#self.score_lst = list(map(lambda pair : nltk.translate.bleu_score.sentence_bleu([pair[1]],pair[0] ), lst ))
	#avg = np.mean(np.array(lst))

	def __init__(self):
		self.data_num = 0
		self.pair=[]

	def add_pair(self, hypos, refs):
		for i in range(len(hypos)):
			self.pair.append([hypos[i], refs[i]])

	def calculate(self):
		self.score_lst = list(map(lambda pair : nltk.translate.bleu_score.sentence_bleu([pair[1]],pair[0] ), self.pair ))
		self.avg = np.mean(np.array(self.score_lst))

