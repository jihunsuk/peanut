import re
import numpy as np
import math

class Dialogue:

    def __init__(self, path):
        self.PRE_DEFINED = ['_P_', '_S_', '_E_', '_U_']
        self.sentences = self.load_data(path)
        self.voc_arr = self.PRE_DEFINED + self.make_voc()
        self.voc_dict = {voc: i for i, voc in enumerate(self.voc_arr)}
        self.voc_size = len(self.voc_dict)
        self.seq_data = self.make_seq_data()
        self.max_output_len = 20    # 출력값 최대길이

        self.index_in_epoch = 0

    # 파일로부터 문장을 읽어온다.
    def load_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f]
        return sentences

    # 문자열로 바꾸어준다.
    def decode(self, indices, string=False):
        tokens = [[self.voc_arr[i] for i in dec] for dec in indices]

        if string:
            return self._decode_to_string(tokens[0])
        else:
            return tokens

    def _decode_to_string(self, tokens):
        text = ' '.join(tokens)
        return text.strip()

    # E 가 있는 전까지 반환
    def cut_eos(self, indices):
        eos_idx = indices.index('_E_')
        return indices[:eos_idx]

    # E 인지 아닌지 검사
    def is_eos(self, voc_id):
        return voc_id == 2

    # 미리 정의한 (E, U, S, P)인지 검사
    def is_defined(self, voc_id):
        return voc_id in self.PRE_DEFINED

    # 정규화식에서 빼고 띄워쓰기 기준으로 자른다.
    def tokenizer(self, sentence):
        sentence = re.sub("[.,!?\"':;)(]", ' ', sentence)
        tokens = sentence.split()
        return tokens

    # 토큰을 id로 바꿔 반환한다.
    def tokens_to_ids(self, tokens):
        ids = [self.voc_dict[token] if token in self.voc_arr else self.voc_dict['_U_'] for token in tokens]
        return ids

    # id를 토큰으로 바꿔 반환한다.
    def ids_to_tokens(self, ids):
        tokens = [self.voc_arr[id] for id in ids]
        return tokens

    # max_len 만큼 패딩 추가.
    def pad(self, seq, max_len, start=None, eos=None):
        if start:
            padded_seq = [1] + seq  # 1은 시작 심볼
        elif eos:
            padded_seq = seq + [2]  # 2는 끝 심볼
        else:
            padded_seq = seq

        if len(padded_seq) < max_len:
            return padded_seq + ([0] * (max_len - len(padded_seq))) # 0은 패딩 심볼
        else:
            return padded_seq

    # 사전을 만든다.
    def make_voc(self):
        voc_set = set()
        for sentence in self.sentences:
            voc_set.update(self.tokenizer(sentence))
        return list(voc_set)

    # 문자열을 숫자(index) 배열로 만든다.
    def make_seq_data(self):
        seq_data = [self.tokens_to_ids(self.tokenizer(sentence)) for sentence in self.sentences]

        return seq_data

    # batch_set에 있는 입력과 출력의 최대 길이를 반환
    def max_len(self, batch_set):
        max_len_input = 0
        max_len_output = 0

        for i in range(0, len(batch_set), 2):
            len_input = len(batch_set[i])
            len_output = len(batch_set[i+1])
            if len_input > max_len_input:
                max_len_input = len_input
            if len_output > max_len_output:
                max_len_output = len_output

        return max_len_input, max_len_output + 1

    # batct_size만큼 입력데이터를 반환
    def next_batch(self, batch_size):
        enc_batch = []
        dec_batch = []
        target_batch = []

        start = self.index_in_epoch

        if self.index_in_epoch + batch_size < len(self.seq_data) - 1:
            self.index_in_epoch = self.index_in_epoch + batch_size
        else:
            self.index_in_epoch = 0

        batch_set = self.seq_data[start:start + batch_size]
        max_len_input, max_len_output = self.max_len(batch_set)
        for i in range(0, len(batch_set) - 1, 2):
            enc, dec, tar = self.transform(batch_set[i], batch_set[i+1], max_len_input, max_len_output)

            enc_batch.append(enc)
            dec_batch.append(dec)
            target_batch.append(tar)

        return enc_batch, dec_batch, target_batch

    # 입력과 출력을 변환
    def transform(self, input, output, max_len_input, max_len_output):

        # 각각의 길이만큼 심볼 추가
        enc_input = self.pad(input, max_len_input)
        dec_input = self.pad(output, max_len_output, start=True)
        target = self.pad(output, max_len_output, eos=True)

        # 인코더의 입력을 뒤집는다.
        enc_input.reverse()

        enc_input = np.eye(self.voc_size)[enc_input]
        dec_input = np.eye(self.voc_size)[dec_input]

        return enc_input, dec_input, target

