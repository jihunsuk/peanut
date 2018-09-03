import numpy as np
import nltk


class BLEUcalculator():
    # format
    # hypo = ['It', 'is', 'a', 'cat', 'at', 'room']
    # ref = ['It', 'is', 'a', 'cat', 'inside', 'the', 'room']
    # self.pair = [[hypothesis, reference], [hypothesis,reference], ....]
    # self.score_lst = list(map(lambda pair : nltk.translate.bleu_score.sentence_bleu([pair[1]],pair[0] ), lst ))
    # avg = np.mean(np.array(lst))

    def __init__(self):
        self.data_num = 0
        self.pair = []

    def add_pair(self, hypos, refs):
        for i in range(len(hypos)):
            self.pair.append([hypos[i], refs[i]])

    def read_txt(self):
        hypos = refs= []

        hypo_path =""
        txt_file = open(hypo_path,'r')
        for line in txt_file.readlines():
            line = line.strip()
            hypos.append(line.split(' '))

        ref_path =""
        txt_file = open(ref_path, "r")
        for line in txt_file.readlines():
            refs.append(line.strip().split(' '))
        self.add_pair(hypos, refs)


    def calculate(self):
        self.score_lst = list(map(lambda pair: nltk.translate.bleu_score.sentence_bleu([pair[1]], pair[0]), self.pair))
        self.avg = np.mean(np.array(self.score_lst))

#hypothesis
expect_path = '/Users/leehayeon/Downloads/expect.txt'
predict_path = '/Users/leehayeon/Downloads/predict.txt'

hypothesis =[]
reference =[]

expect_file = open(expect_path, "r")
predict_file = open(predict_path, "r")

for line in expect_file.readlines():
	new_line = line.strip().replace('  ', '*')
	new_line = new_line.replace(' ', '')
	new_line = new_line.replace('*', ' ')
	hypothesis.append(new_line.split(' '))

for line in predict_file.readlines():
        new_line = line.strip().replace('  ', '*')
        new_line = new_line.replace(' ', '')
        new_line = new_line.replace('*', ' ')
        reference.append(new_line.split(' '))

print(hypothesis[0])
print(reference[0])

bleu = BLEUcalculator()
bleu.add_pair(hypothesis, reference)
bleu.calculate()
print(bleu.avg)
