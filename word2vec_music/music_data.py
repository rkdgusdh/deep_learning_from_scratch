# coding: utf-8
'''음악 말뭉치(.tok)를 읽어 교재의 (corpus, word_to_id, id_to_word) 형태로 바꾼다.'''
import os
import collections
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')


def prepare():
    '''저장소에 들어 있는 정제본(music_clean.txt.gz)에서 토큰 파일(*.tok)을 만든다(처음 한 번, 약 1분).'''
    import gzip, shutil, sys
    clean = os.path.join(DATA_DIR, 'music_clean.txt')
    if not os.path.exists(clean):
        with gzip.open(clean + '.gz', 'rb') as fi, open(clean, 'wb') as fo:
            shutil.copyfileobj(fi, fo)
    sys.path.insert(0, DATA_DIR)
    import tokenize_music
    tokenize_music.main(clean, DATA_DIR)


def load_docs(tokenizer='kiwi'):
    '''[(제목, [토큰, ...]), ...] — 문서 단위로 토큰을 이어 붙인다.'''
    path = os.path.join(DATA_DIR, 'music_%s.tok' % tokenizer)
    if not os.path.exists(path):
        prepare()
    docs = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('## '):
                docs.append((line[3:].split('|')[0].strip(), []))
            elif line:
                docs[-1][1].extend(line.split())
    return docs


def load_data(tokenizer='kiwi', min_count=5, frac=1.0):
    '''min_count번 미만 나온 단어는 <unk>로 바꾼다(PTB가 1만 단어 밖을 <unk>로 바꾼 것과 같은 처리).
    frac < 1 이면 앞쪽 문서 일부만 쓴다(문서 순서는 수집 단계에서 이미 무작위로 섞었다).'''
    docs = load_docs(tokenizer)
    docs = docs[:int(len(docs) * frac)]
    words = [w for _, toks in docs for w in toks]
    counts = collections.Counter(words)
    vocab = ['<unk>'] + [w for w, c in counts.most_common() if c >= min_count]
    word_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_word = {i: w for w, i in word_to_id.items()}
    unk = word_to_id['<unk>']
    corpus = np.array([word_to_id.get(w, unk) for w in words], dtype=np.int32)
    return corpus, word_to_id, id_to_word
