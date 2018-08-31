import tensorflow as tf
import math
import sys
import random
import numpy as np

sys.path.insert(0, '/')

from model import Hred
from Dialogue import Dialogue

def train(dialog, batch_size=10, epoch=100):

    model = Hred(dialog.voc_size, True, 10)
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('./model')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("다음 파일에서 모델을 읽는 중 입니다..", ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("새로운 모델을 생성하는 중 입니다.")
            sess.run(tf.global_variables_initializer())

        # 학습시작.
        total_batch = int(math.ceil(len(dialog.seq_data)/(float(batch_size)*10)))
        for step in range(total_batch * epoch):

            enc_inputs = []
            enc_lengths = []
            dec_inputs = []
            dec_lengths = []
            targets = []

            for i in range(10):
                enc_input, enc_length, dec_input, dec_length, target = dialog.next_batch(batch_size)

                enc_inputs.append(enc_input)
                enc_lengths.append(enc_length)
                dec_inputs.append(dec_input)
                dec_lengths.append(dec_length)
                targets.append(target)

            max_dec_lengths = np.max(dec_lengths, 1)
            context_size = len(enc_input)

            _, loss = sess.run([model.train_op, model.cost],
                               feed_dict={model.enc_input_idx: enc_inputs,
                                          model.dec_input_idx: dec_inputs,
                                          model.enc_length: enc_lengths,
                                          model.dec_length: dec_lengths,
                                          model.targets: targets,
                                          model.dec_max_length: max_dec_lengths,
                                          model.context_size: context_size})

            if (step+1) % 100 == 0:
                print('Step:', '%06d' % model.global_step.eval(),
                      'cost =', '{:.6f}'.format(loss))

            if (step+1) % 500 == 0:
                model.saver.save(sess, './model/conversation.ckpt', global_step=model.global_step)

        print('최적화 완료!')

def test(batch_size=10):
    data_path = './data/dict_idx_char_test.npy'
    vocab_path = './data/char_dictionary.txt'

    dialog = Dialogue()

    dialog.load_vocab(vocab_path)
    dialog.load_data(data_path)

    print("\n=== 예측 테스트 ===")
    model = Hred(dialog.voc_size, False, 1)

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('./model')
        print("다음 파일에서 모델을 읽는 중 입니다..", ckpt.model_checkpoint_path)

        model.saver.restore(sess, ckpt.model_checkpoint_path)

        all_expect = []
        all_predict = []

        total_batch = int(math.ceil(len(dialog.seq_data) / float(batch_size)))
        for step in range(total_batch):
            enc_input, enc_length, dec_input, dec_length, targets = dialog.next_batch(batch_size)

            expects, outputs = sess.run([model.targets, model.outputs],
                                        feed_dict={model.enc_input_idx: [enc_input],
                                                   model.dec_input_idx: [dec_input],
                                                   model.enc_length: [enc_length],
                                                   model.dec_length: [dec_length],
                                                   model.targets: [targets],
                                                   model.dec_max_length: [np.max(dec_length, 0)],
                                                   model.context_size: len(enc_input)})


            for i in range(len(outputs)):
                all_expect.append(dialog.cut_eos(dialog.decode([expects[0][i]], True)))
                all_predict.append(dialog.cut_eos(dialog.decode([outputs[i]], True)))

        # all_expect : 실제값 저장되어있음
        # all_predict: 예측값 저장되어있음
        # for i in range(len(all_predict)):
        #     print("실제값:", all_expect[i])
        #     print("예측값:", ''.join(all_predict[i]))


def main(_):
    data_path = './data/dict_idx_char_training.npy'
    vocab_path = './data/char_dictionary.txt'

    dialog = Dialogue()

    dialog.load_vocab(vocab_path)
    dialog.load_data(data_path)

    train(dialog, epoch=100)   # 학습
    #test()               # 테스트

if __name__ == "__main__":
    tf.app.run()
