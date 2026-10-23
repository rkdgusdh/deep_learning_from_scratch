"""저장된 가중치로 좌우 반전 TTA(test-time augmentation) 예측을 다시 계산한다. 재학습 없음.
반전한 눈 이미지(yaw, head-pose yaw 부호 반전)의 예측을 다시 반전해 원래 예측과 평균한다.
사용법: python tta_eval.py --protocol lopo L_aug L_augreg  -> runs/<name>_<fold>_predtta.npy 저장"""
import argparse, json, os
import numpy as np
import nn, models
from train import load_data, split, normalize, predict, HERE

ap = argparse.ArgumentParser()
ap.add_argument('names', nargs='+')
ap.add_argument('--protocol', default='lopo')
a = ap.parse_args()
nf, suf = (15, 'p{:02d}') if a.protocol == 'lopo' else (5, 'f{}')
X, G, H, P = load_data('none')
for name in a.names:
    plain, tta = [], []
    for k in range(nf):
        run = f'{name}_{suf.format(k)}'
        meta = json.load(open(os.path.join(HERE, 'runs', run + '.json')))['args']
        tr, te, _ = split(P, k, a.protocol)
        stats = (X[tr].mean(), X[tr].std())
        m = models.build(meta['model'], np.random.default_rng(0), **json.loads(meta['model_kw']))
        nn.load_state(m, dict(np.load(os.path.join(HERE, 'runs', run + '_weights.npz'))))
        Xte = normalize(X[te], meta['norm'], stats)
        p = predict(m, Xte, H[te], tta_flip=True)
        np.save(os.path.join(HERE, 'runs', run + '_predtta.npy'), p)
        e0 = nn.angular_error_deg(np.load(os.path.join(HERE, 'runs', run + '_pred.npy')), G[te]).mean()
        e1 = nn.angular_error_deg(p, G[te]).mean()
        plain.append(e0); tta.append(e1)
        print(f'{run}: plain {e0:.3f}  flip-TTA {e1:.3f}', flush=True)
    print(f'== {name}: plain {np.mean(plain):.3f}  flip-TTA {np.mean(tta):.3f}', flush=True)
