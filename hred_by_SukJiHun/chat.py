import tensorflow as tf
import sys
import numpy as np

from Dialogue import Dialogue
from model import Hred
from train import train

class chatbot:
    def __init__(self, voc_path):
        self.dialogue = Dialogue(voc_path)
        train(self.dialogue, epoch=1000) # train.py와 마찬가지로 train을 하지 않고 바로 모델만 불러와서 하면 잘 되지 않는다(원인불명)
        self.model = Hred(self.dialogue.input_max_len, self.dialogue.output_max_len, self.dialogue.voc_size, False)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # 모델 불러오기
        ckpt = tf.train.get_checkpoint_state('./model')
        self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def run(self):
        sys.stdout.write("> ")
        sys.stdout.flush()

        setences = []
        line = sys.stdin.readline()
        setences.append(line.strip())

        while line:
            print(self.get_replay(setences)) # 응답

            sys.stdout.write("\n> ")
            sys.stdout.flush()

            line = sys.stdin.readline()
            setences.append(line.strip())

    def _decode(self, enc_input, dec_input):
        if type(dec_input) is np.ndarray:
            dec_input = dec_input.tolist()

        enc_len = []
        dec_len = []

        enc_batch = []
        dec_batch = []
        for i in range(0, len(enc_input)):
            enc, dec, _ = self.dialogue.transform(enc_input[i], dec_input[i], self.dialogue.input_max_len, self.dialogue.output_max_len)
            enc_batch.append(enc)
            dec_batch.append(dec)
            enc_len.append(len(enc_input[i]))
            dec_len.append(len(dec_input[i])+1)

        return self.model.predict(self.sess, enc_batch, enc_len, dec_batch, dec_len, len(enc_batch))

    # msg에 대한 응답을 반환
    def get_replay(self, sentences):

        enc_input = [self.dialogue.tokens_to_ids(self.dialogue.tokenizer(sentence)) for sentence in sentences]
        dec_input = enc_input

        outputs = self._decode(enc_input, dec_input)
        reply = self.dialogue.decode([outputs[len(enc_input)-1]], True)
        reply = self.dialogue.cut_eos(reply)

        return reply

path = './data/chat.log'

def main(_):
    print("깨어나는 중 입니다. 잠시만 기다려주세요...\n")

    Chatbot = chatbot(path)
    Chatbot.run()

if __name__ == "__main__":
    tf.app.run()