import tensorflow as tf
import sys
import math

from Dialogue import Dialogue
from model import Hred

class chatbot:
    def __init__(self, voc_path):
        self.dialogue = Dialogue()
        self.dialogue.load_vocab(voc_path)
        self.model = Hred(self.dialogue.voc_size, False)
        self.sess = tf.Session()

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
        enc_len = []
        dec_len = []

        enc_batch = []
        dec_batch = []

        input_len = int(math.ceil((len(enc_input) + 1) * 1.5)) # 버킷 사용

        for i in range(0, len(enc_input)):
            enc, dec, _ = self.dialogue.transform(enc_input[i], dec_input[i], input_len, input_len)
            enc_batch.append(enc)
            dec_batch.append(dec)
            enc_len.append(len(enc_input[i]))
            dec_len.append(len(dec_input[i])+1)
        # dec_batch, dec_len은 원래 필요하지않는데 model에서 placeholder로 해놔서 그냥 enc랑 동일하게 주고있습니다.
        return self.model.predict(self.sess, enc_batch, enc_len, dec_batch, dec_len, len(enc_batch))

    # msg에 대한 응답을 반환
    def get_replay(self, sentences):

        enc_input = [self.dialogue.tokens_to_ids(self.dialogue.tokenizer(sentence)) for sentence in sentences]
        dec_input = enc_input

        outputs = self._decode(enc_input, dec_input)
        reply = self.dialogue.decode([outputs[len(enc_input)-1]], True)
        reply = self.dialogue.cut_eos(reply)

        return reply

data_path = './data/chat.log'
vocab_path = './data/chat.voc'

def main(_):

    print("깨어나는 중 입니다. 잠시만 기다려주세요...\n")

    Chatbot = chatbot(vocab_path)
    Chatbot.run()

if __name__ == "__main__":
    tf.app.run()