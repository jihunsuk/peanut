import numpy as np
import tensorflow as tf   
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from model.hred import hred
from utils.Dialogue import Dialogue
import utils.utils as config

#temp data
enc_input_data = [[10, 5, 4, 7, 8], [8, 4, 6, 7, 0], [7, 10, 4, 0, 0], [4, 8, 0, 0, 0]]
dec_input_data = [[1, 8, 6, 4, 5], [1, 10, 4, 8, 0], [1, 7, 5, 0, 0], [1, 5, 0, 0, 0]]
target_data = [[8, 6, 4, 5, 2], [10, 4, 8, 2, 0], [7, 5, 2, 0, 0], [5, 2, 0, 0, 0]]
enc_seq_len = [5, 4, 3, 2]
dec_seq_len = [5, 4, 3, 2]
max_len = 5

args = config.get_args(sys.argv[1:])

total_epoch =  args.n_epochs
num_units = args.n_hidden
learning_rate = 0.001
voc_size = 11

hred_model = hred(voc_size, learning_rate, num_units)
fdtr = {hred_model.enc_inputs: enc_input_data, 
		 hred_model.dec_inputs: dec_input_data, 
		 hred_model.targets: target_data, 
		 hred_model.enc_seq_length: enc_seq_len, 
		 hred_model.dec_seq_length: dec_seq_len, 
		 hred_model.max_length: max_len}

sys.path.insert(0, '/')

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(total_epoch):
	loss = hred_model.train(sess, fdtr)
	print(epoch, loss)


def train(dialog, batch_size=100, epoch=100):
    model = Seq2Seq(dialog.voc_size)

    with tf.Session() as sess:

        # 모델을 읽어온다. 없으면 새로 만든다.
        ckpt = tf.train.get_checkpoint_state('./log')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("다음 파일에서 모델을 읽는 중 입니다..", ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("새로운 모델을 생성하는 중 입니다.")
            sess.run(tf.global_variables_initializer())

        # 학습된 모델을 저장한다.
        checkpoint_path = os.path.join('./log', 'conversation.ckpt')
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)

        print('최적화 완료!')

def test(dialog, batch_size=100):
    print("\n=== 예측 테스트 ===")

    model = Seq2Seq(dialog.voc_size)

    with tf.Session() as sess:
        # 모델을 읽어온다.
        ckpt = tf.train.get_checkpoint_state('./log')
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
'''
if __name__ == "__main__":
    tf.app.run()
'''