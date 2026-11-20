# coding: utf-8
'''교재 ch04/train.py 를 음악 말뭉치용으로 고친 것.

바뀐 점
  * 데이터: ptb.load_data() 대신 music_data.load_data()
  * 네거티브 샘플링: 교재 UnigramSampler 의 CPU 경로는 샘플마다 np.random.choice 를 호출해
    어휘가 3만 개인 이 말뭉치에서는 너무 느리다. 그래서 교재 코드의 GPU 경로와 같은
    "배치 전체를 한 번에 뽑는(타깃이 섞일 수 있음)" 방식을 CPU에서도 쓰도록 바꿨다.
    (fast_negative_sampling 참고. 분포는 교재와 같은 unigram^0.75)

사용 예) python train_music.py --tokenizer kiwi --model cbow --window 5 --hidden 100 --epoch 5
'''
import sys, os, time, argparse, pickle, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common.trainer import Trainer
from common.optimizer import Adam
from common.util import create_contexts_target
import negative_sampling_layer
from cbow import CBOW
from skip_gram import SkipGram
from music_data import load_data


def fast_negative_sampling(self, target):
    batch_size = target.shape[0]
    return np.random.choice(self.vocab_size, size=(batch_size, self.sample_size),
                            replace=True, p=self.word_p)


negative_sampling_layer.UnigramSampler.get_negative_sample = fast_negative_sampling


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokenizer', default='kiwi')
    ap.add_argument('--model', default='cbow')
    ap.add_argument('--window', type=int, default=5)
    ap.add_argument('--hidden', type=int, default=100)
    ap.add_argument('--batch', type=int, default=100)
    ap.add_argument('--epoch', type=int, default=5)
    ap.add_argument('--min_count', type=int, default=5)
    ap.add_argument('--frac', type=float, default=1.0)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    np.random.seed(a.seed)

    corpus, word_to_id, id_to_word = load_data(a.tokenizer, a.min_count, a.frac)
    vocab_size = len(word_to_id)
    contexts, target = create_contexts_target(corpus, a.window)
    print('토큰 %d, 어휘 %d, 학습 샘플 %d' % (len(corpus), vocab_size, len(target)), flush=True)

    Model = CBOW if a.model == 'cbow' else SkipGram
    model = Model(vocab_size, a.hidden, a.window, corpus)
    trainer = Trainer(model, Adam(lr=a.lr))
    t0 = time.time()
    trainer.fit(contexts, target, a.epoch, a.batch, eval_interval=max(1, len(target) // a.batch // 10))  # 에폭당 10번 손실 기록
    elapsed = time.time() - t0

    name = a.out or '%s_%s_w%d_h%d_mc%d_f%g' % (a.model, a.tokenizer, a.window, a.hidden, a.min_count, a.frac)
    os.makedirs('results', exist_ok=True)
    params = {'word_vecs': model.word_vecs.astype(np.float16), 'word_to_id': word_to_id,
              'id_to_word': id_to_word, 'loss_list': trainer.loss_list, 'args': vars(a),
              'tokens': len(corpus), 'time_sec': elapsed}
    with open(os.path.join('results', name + '.pkl'), 'wb') as f:
        pickle.dump(params, f, -1)
    print('저장: results/%s.pkl  (%.0f초)' % (name, elapsed))


if __name__ == '__main__':
    main()
