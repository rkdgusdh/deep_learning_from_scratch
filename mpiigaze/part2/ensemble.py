"""여러 실험의 테스트 예측(pitch, yaw)을 평균해 앙상블 오차를 계산한다.
사용법: python ensemble.py --protocol 5fold res_aug res_aug_s1 res_aug_s2
각 이름은 fold 접미사(_f0, _p03 등)를 뺀 실험 이름. 모든 실험에 공통으로 있는 fold만 쓴다."""
import argparse, os, json
import numpy as np
import nn
from train import load_data, split, HERE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('names', nargs='+')
    ap.add_argument('--protocol', default='5fold')
    ap.add_argument('--label', default=None)
    ap.add_argument('--tta', action='store_true', help='tta_eval.py가 만든 좌우 반전 TTA 예측 사용')
    a = ap.parse_args()
    nf, suf = (15, 'p{:02d}') if a.protocol == 'lopo' else (5, 'f{}')
    _, G, _, P = load_data('none')
    rows, all_err = [], []
    for k in range(nf):
        paths = [os.path.join(HERE, 'runs', f'{n}_{suf.format(k)}_pred{"tta" if a.tta else ""}.npy') for n in a.names]
        if not all(os.path.exists(p) for p in paths):
            continue
        _, te, _ = split(P, k, a.protocol)
        preds = [np.load(p) for p in paths]
        ens = np.mean(preds, 0)
        err = nn.angular_error_deg(ens, G[te])
        singles = [nn.angular_error_deg(p, G[te]).mean() for p in preds]
        rows.append((k, err.mean(), np.mean(singles)))
        all_err.append(err)
    for k, e, s in rows:
        print(f'fold {k:2d}: ensemble {e:.3f}  (single mean {s:.3f})')
    if rows:
        tot = np.concatenate(all_err).mean()
        print(f'{len(rows)}/{nf} folds  ensemble mean {tot:.3f}  single mean {np.mean([r[2] for r in rows]):.3f}')
        out = os.path.join(HERE, 'results_ensemble.jsonl')
        with open(out, 'a') as f:
            f.write(json.dumps(dict(label=a.label or '+'.join(a.names), protocol=a.protocol, tta=a.tta, names=a.names,
                                    folds=[r[0] for r in rows], ensemble=round(float(tot), 3),
                                    single_mean=round(float(np.mean([r[2] for r in rows])), 3))) + '\n')


if __name__ == '__main__':
    main()
