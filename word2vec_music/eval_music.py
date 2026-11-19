# coding: utf-8
'''학습된 단어 벡터(results/*.pkl)를 평가한다.

  1) 유추 정확도: eval/music_analogy.txt 의 a:b = c:? 문제를 교재 analogy()와 같은 방식
     (b - a + c 와 코사인 유사도가 가장 큰 단어, a·b·c 제외)으로 풀어 top-1 / top-5 정답률
  2) 범주 군집 정확도: eval/music_categories.txt 의 단어마다, 평가 단어 전체 중 가장 가까운 5개가
     같은 범주인 비율(precision@5)
'''
import os, sys, pickle, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))


def load_params(path):
    with open(path, 'rb') as f:
        p = pickle.load(f)
    W = p['word_vecs'].astype(np.float32)
    W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-8
    p['W'] = W
    return p


def load_analogy(path=os.path.join(HERE, 'eval', 'music_analogy.txt')):
    qs, cat = [], None
    for line in open(path, encoding='utf-8'):
        if line.startswith(':'):
            cat = line[1:].strip()
        elif line.strip():
            qs.append((cat, line.split()))
    return qs


def load_categories(path=os.path.join(HERE, 'eval', 'music_categories.txt')):
    cats = {}
    for line in open(path, encoding='utf-8'):
        name, ws = line.split(':', 1)
        cats[name.strip()] = ws.split()
    return cats


def solve_analogy(p, a, b, c, top=5):
    w2i, i2w, W = p['word_to_id'], p['id_to_word'], p['W']
    q = W[w2i[b]] - W[w2i[a]] + W[w2i[c]]
    q /= np.linalg.norm(q)
    sim = W @ q
    for w in (a, b, c):
        sim[w2i[w]] = -np.inf
    sim[w2i['<unk>']] = -np.inf
    return [i2w[i] for i in np.argsort(-sim)[:top]]


def eval_analogy(p, qs=None):
    qs = qs or load_analogy()
    w2i = p['word_to_id']
    res = collections.defaultdict(lambda: [0, 0, 0, 0])   # 전체, 풀 수 있음, top1, top5
    for cat, (a, b, c, d) in qs:
        r = res[cat]; r[0] += 1
        if not all(w in w2i for w in (a, b, c, d)):
            continue
        r[1] += 1
        pred = solve_analogy(p, a, b, c)
        r[2] += pred[0] == d
        r[3] += d in pred
    tot = [sum(r[i] for r in res.values()) for i in range(4)]
    res['전체'] = tot
    return {k: dict(total=v[0], covered=v[1], top1=v[2] / max(v[1], 1), top5=v[3] / max(v[1], 1))
            for k, v in res.items()}


def eval_categories(p, cats=None, k=5):
    cats = cats or load_categories()
    w2i, W = p['word_to_id'], p['W']
    words, labels = [], []
    for name, ws in cats.items():
        for w in ws:
            if w in w2i:
                words.append(w); labels.append(name)
    X = W[[w2i[w] for w in words]]
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    labels = np.array(labels)
    prec = collections.defaultdict(list)
    for i in range(len(words)):
        nn = np.argsort(-S[i])[:k]
        prec[labels[i]].append(np.mean(labels[nn] == labels[i]))
    out = {name: float(np.mean(v)) for name, v in prec.items()}
    out['전체'] = float(np.mean([x for v in prec.values() for x in v]))
    return out


def most_similar(p, query, top=5):
    w2i, i2w, W = p['word_to_id'], p['id_to_word'], p['W']
    if query not in w2i:
        return []
    sim = W @ W[w2i[query]]
    sim[w2i[query]] = -np.inf
    return [(i2w[i], float(sim[i])) for i in np.argsort(-sim)[:top]]


if __name__ == '__main__':
    for path in sys.argv[1:]:
        p = load_params(path)
        a = eval_analogy(p); c = eval_categories(p)
        print('%s  유추 top1 %.3f top5 %.3f (%d/%d문항)  범주 P@5 %.3f' % (
            os.path.basename(path), a['전체']['top1'], a['전체']['top5'], a['전체']['covered'],
            a['전체']['total'], c['전체']))
