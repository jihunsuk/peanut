import tensorflow as tf
import math
import sys
import random

sys.path.insert(0, '/')

from model import Hred
from Dialogue import Dialogue

def train(dialog, batch_size=20, epoch=100):

    model = Hred(dialog.voc_size, True)
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('./model')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("다음 파일에서 모델을 읽는 중 입니다..", ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("새로운 모델을 생성하는 중 입니다.")
            sess.run(tf.global_variables_initializer())

        # 학습시작.
        total_batch = int(math.ceil(len(dialog.seq_data)/float(batch_size)))
        for step in range(total_batch * epoch):
            enc_input, enc_length, dec_input, dec_length, targets = dialog.next_batch(batch_size)

            _, loss = sess.run([model.train_op, model.cost],
                        feed_dict={model.enc_input_idx: enc_input,
                                   model.dec_input_idx: dec_input,
                                   model.enc_length: enc_length,
                                   model.dec_length: dec_length,
                                   model.targets: targets,
                                   model.batch_size: len(enc_input)})

            if (step+1) % 100 == 0:
                print('Step:', '%06d' % model.global_step.eval(),
                      'cost =', '{:.6f}'.format(loss))

        model.saver.save(sess, './model/conversation.ckpt', global_step=model.global_step)

        print('최적화 완료!')

def test(dialog, batch_size=20):
    print("\n=== 예측 테스트 ===")
    model = Hred(dialog.voc_size, False)

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('./model')
        print("다음 파일에서 모델을 읽는 중 입니다..", ckpt.model_checkpoint_path)

        model.saver.restore(sess, ckpt.model_checkpoint_path)

        enc_input, enc_length, dec_input, dec_length, targets = dialog.next_batch(batch_size)

        expect, outputs, accuracy = sess.run([model.targets, model.outputs, model.accuracy],
                                             feed_dict={model.enc_input_idx: enc_input,
                                                        model.dec_input_idx: dec_input,
                                                        model.enc_length: enc_length,
                                                        model.dec_length: dec_length,
                                                        model.targets: targets,
                                                        model.batch_size: len(enc_input)})

        expect = dialog.decode(expect)
        outputs = dialog.decode(outputs)

        pick = random.randrange(0, len(expect) / 2)
        input = dialog.decode([dialog.seq_data[pick * 2]], True)
        expect = dialog.decode([dialog.seq_data[pick * 2 + 1]], True)
        outputs = dialog.cut_eos(outputs[pick])

        print("\n정확도:", accuracy)
        print("랜덤 결과\n")
        print("    입력값:", input)
        print("    실제값:", expect)
        print("    예측값:", ' '.join(outputs))


def main(_):
    data_path = './data/chat.log'
    vocab_path = './data/chat.voc'

    dialog = Dialogue()

    dialog.load_vocab(vocab_path)
    dialog.load_data(data_path)

    #train(dialog, epoch=1000)   # 학습
    test(dialog)               # 테스트


if __name__ == "__main__":
    tf.app.run()
