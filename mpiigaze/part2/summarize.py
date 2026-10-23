"""results.csv를 실험(이름에서 _f<k> 제거)별로 묶어 fold 평균을 표로 출력하고 results_summary.md에 저장.
fold마다 테스트 샘플 수가 같으므로(9000) fold 평균 = 45,000장 전체 평균이다."""
import csv, os, re, json
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, 'results.csv'))))
groups = defaultdict(dict)
order = []
for r in rows:
    base = re.sub(r'_(f\d|p\d\d)$', '', r['name'])
    if base not in groups:
        order.append(base)
    groups[base][int(r['fold'])] = r

lines = ['| 실험 | 프로토콜 | 모델 | 파라미터 | fold별 테스트 오차(°) | 평균(°) | 완료 fold |', '|---|---|---|---|---|---|---|']
for b in order:
    g = groups[b]
    errs = [float(g[k]['test_err']) for k in sorted(g)]
    r0 = next(iter(g.values()))
    folds = ' / '.join(f"f{k}:{float(g[k]['test_err']):.2f}" for k in sorted(g))
    nf = 15 if r0.get('protocol') == 'lopo' else 5
    full = f'{len(g)}/{nf}'
    lines.append(f"| {b} | {r0.get('protocol', '5fold')} | {r0['model']} {r0['model_kw'] if r0['model_kw'] != '{}' else ''} | {int(r0['params']):,} | {folds} | {np.mean(errs):.2f} | {full} |")
txt = '\n'.join(lines)
print(txt)
open(os.path.join(HERE, 'results_summary.md'), 'w').write(txt + '\n')
