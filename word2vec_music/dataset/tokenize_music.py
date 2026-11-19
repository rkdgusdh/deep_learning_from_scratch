"""정제된 음악 말뭉치를 두 가지 토크나이저로 토큰화해 저장한다.

  rule : common/korean.py 의 규칙 기반 조사 분리 (김소월 예제와 같은 방식)
  kiwi : Kiwi 형태소 분석기 (pip install kiwipiepy)
         - 명사·부사 등은 그대로, 동사/형용사 어간은 '부르다'처럼 '다'를 붙여 기본형으로
         - 조사·접미사는 '-를', '-적'처럼 '-'를 붙여 구분 (rule 과 같은 표기)
         - 어미(-었, -고 ...)는 버린다

두 방식 모두 문장부호는 버린다. 출력: 한 줄에 한 문장, 토큰은 공백으로 구분.
문서 경계는 '## 제목' 줄로 남겨 말뭉치 크기 실험에서 문서 단위로 자를 수 있게 한다.
사용법: python dataset/tokenize_music.py dataset/music_clean.txt dataset/
"""
import os, re, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.korean import tokenize as rule_tokenize

WORD = re.compile(r'[가-힣A-Za-z0-9]')


def kiwi_token(t):
    tag = t.tag
    if tag.startswith('E') or tag.startswith('S'):   # 어미, 문장부호/기호는 버림 (SL 영어, SN 숫자는 아래서 살림)
        if tag in ('SL', 'SN', 'SH'):
            return t.form
        return None
    if tag in ('VV', 'VA', 'VX', 'XSV', 'XSA', 'VV-I', 'VA-I', 'VV-R', 'VA-R'):
        return t.form + '다'
    if tag in ('VCP', 'VCN'):
        return '-' + t.form + '다'
    if tag.startswith('J') or tag.startswith('XS') or tag == 'XPN':
        return '-' + t.form
    return t.form


def main(src, out_dir):
    from kiwipiepy import Kiwi
    kiwi = Kiwi(num_workers=4)
    docs, cur = [], None
    for line in open(src, encoding='utf-8'):
        line = line.rstrip('\n')
        if line.startswith('## '):
            cur = [line, []]; docs.append(cur)
        elif line:
            cur[1].append(line)
    sents = [s for _, ss in docs for s in ss]
    kiwi_out = iter([' '.join(w for w in (kiwi_token(t) for t in toks) if w) for toks in kiwi.tokenize(sents)])
    with open(os.path.join(out_dir, 'music_rule.tok'), 'w', encoding='utf-8') as fr, \
         open(os.path.join(out_dir, 'music_kiwi.tok'), 'w', encoding='utf-8') as fk:
        for head, ss in docs:
            fr.write(head + '\n'); fk.write(head + '\n')
            for s in ss:
                fr.write(' '.join(w for w in rule_tokenize(s) if WORD.search(w)) + '\n')
                fk.write(next(kiwi_out) + '\n')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
