import tensorflow as tf
import math
import sys
import random

sys.path.insert(0, '/')

from model import Hred
from Dialogue import Dialogue


def train(dialog, batch_size=100, epoch=100):

    model = Hred(dialog.input_max_len, dialog.output_max_len, dialog.voc_size, True)

    with tf.Session() as sess:
        # 모델을 읽어온다. 없으면 새로 만든다.
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state('./model')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("다음 파일에서 모델을 읽는 중 입니다..", ckpt.model_checkpoint_path)

            loader = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta')
            loader.restore(sess, tf.train.latest_checkpoint('./model'))

            total_batch = int(math.ceil(len(dialog.seq_data) / float(batch_size)))
            for step in range(total_batch * epoch):
                enc_input, enc_length, dec_input, dec_length, targets = dialog.next_batch(batch_size)

                _, loss = sess.run(['train_op:0', 'cost:0'],
                                   feed_dict={'enc_input_idx:0': enc_input,
                                              'dec_input_idx:0': dec_input,
                                              'enc_length:0': enc_length,
                                              'dec_length:0': dec_length,
                                              'targets:0': targets,
                                              'batch_size:0': len(enc_input)})
                if (step + 1) % 100 == 0:
                    print('Step:', '%06d' % sess.run('global_step:0'),
                          'cost =', '{:.6f}'.format(loss))

            model.saver.save(sess, './model/conversation.ckpt', global_step=sess.run('global_step:0'))

        else:
            print("새로운 모델을 생성하는 중 입니다.")

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

def test(dialog, batch_size=100):
    print("\n=== 예측 테스트 ===")
    model = Hred(dialog.input_max_len, dialog.output_max_len, dialog.voc_size, False)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 모델을 읽어온다.
        ckpt = tf.train.get_checkpoint_state('./model')
        print("다음 파일에서 모델을 읽는 중 입니다..", ckpt.model_checkpoint_path)

        model.saver.restore(sess, tf.train.latest_checkpoint('./model'))

        enc_input, enc_length, dec_input, dec_length, targets = dialog.next_batch(batch_size)

        expect2, outputs2, accuracy2 = sess.run([model.targets, model.outputs, model.accuracy],
                                             feed_dict={model.enc_input_idx: enc_input,
                                                        model.dec_input_idx: dec_input,
                                                        model.enc_length: enc_length,
                                                        model.dec_length: dec_length,
                                                        model.targets: targets,
                                                        model.batch_size: len(enc_input)})


        expect2 = dialog.decode(expect2)
        outputs2 = dialog.decode(outputs2)

        pick2 = random.randrange(0, len(expect2) / 2)
        input2 = dialog.decode([dialog.seq_data[pick2 * 2]], True)
        expect2 = dialog.decode([dialog.seq_data[pick2 * 2 + 1]], True)
        outputs2 = dialog.cut_eos(outputs2[pick2])

        print("\n정확도:", accuracy2)
        print("랜덤 결과\n")
        print("    입력값:", input2)
        print("    실제값:", expect2)
        print("    예측값:", ' '.join(outputs2))



# 다른 방법 시도
        # graph = tf.get_default_graph()
        # t = graph.get_tensor_by_name("targets:0")
        # o = graph.get_tensor_by_name("outputs:0")
        # a = graph.get_tensor_by_name("accuracy:0")
        # m = graph.get_operation_by_name("embeddings")


        # expect, outputs, accuracy = sess.run(['targets:0', 'outputs:0', 'accuracy:0'],
        # expect, outputs, accuracy = sess.run([t, o, a],
        #                                      feed_dict={'enc_input_idx:0': enc_input,
        #                                                 'dec_input_idx:0': dec_input,
        #                                                 'enc_length:0': enc_length,
        #                                                 'dec_length:0': dec_length,
        #                                                 'targets:0': targets})

        # expect = dialog.decode(expect)
        # outputs = dialog.decode(outputs)
        #
        # pick = random.randrange(0, len(expect) / 2)
        # input = dialog.decode([dialog.seq_data[pick * 2]], True)
        # expect = dialog.decode([dialog.seq_data[pick * 2 + 1]], True)
        # outputs = dialog.cut_eos(outputs[pick])
        #
        # print("\n정확도:", accuracy)
        # print("랜덤 결과\n")
        # print("    입력값:", input)
        # print("    실제값:", expect)
        # print("    예측값:", ' '.join(outputs))


def main(_):
    dialog = Dialogue('./data/chat.log')        # train과 test를 동시에 실행했을 경우, 정확도가 좋게 나온다. 하지만 test만 실행했을 경우 정확도가 엄청낮다. (원인불명, 학습된 모델이 제대로 적용되지 않는것같다)
    #train(dialog, epoch=1000)   # 학습
    test(dialog)               # 테스트


if __name__ == "__main__":
    tf.app.run()
