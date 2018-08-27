import tensorflow as tf
import math
import sys
import os
import random

sys.path.insert(0, '/')

from model import Seq2Seq
from Dialogue import Dialogue

def train(dialog, batch_size=100, epoch=100):
    model = Seq2Seq(dialog.voc_size)

    with tf.Session() as sess:

        # 모델을 읽어온다. 없으면 새로 만든다.
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
            enc_input, dec_input, targets = dialog.next_batch(batch_size)
            _, loss = model.train(sess, enc_input, dec_input, targets)
            if step % 100 == 0:
                print('cost = ', loss)

        # 학습된 모델을 저장한다.
        checkpoint_path = os.path.join('./model', 'conversation.ckpt')
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)

        print('최적화 완료!')

def test(dialog, batch_size=100):
    print("\n=== 예측 테스트 ===")

    model = Seq2Seq(dialog.voc_size)

    with tf.Session() as sess:
        # 모델을 읽어온다.
        ckpt = tf.train.get_checkpoint_state('./model')
        print("다음 파일에서 모델을 읽는 중 입니다..", ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)

        enc_input, dec_input, targets = dialog.next_batch(batch_size)

        expect, outputs, accuracy = model.test(sess, enc_input, dec_input, targets)

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
    dialog = Dialogue('./data/chat.log')
    train(dialog, epoch=2000)   # 학습
    #test(dialog)               # 테스트

if __name__ == "__main__":
    tf.app.run()
