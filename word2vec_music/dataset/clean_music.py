"""음악 문서 정제: 괄호 속 원어·한자·날짜 제거, 한글 위주 문장만 남김, 문장 단위 분리."""
import re, sys, random
src, dst = sys.argv[1], sys.argv[2]
PAREN = re.compile(r'\([^()]*\)|（[^（）]*）|\[[^\[\]]*\]')
HANGUL = re.compile(r'[가-힣]')
LETTER = re.compile(r'[A-Za-z가-힣ぁ-んァ-ン一-龥]')
SENT = re.compile(r'(?<=[다요까])\.\s*')
docs = []
for block in open(src, encoding='utf-8').read().split('\n## '):
    block = block.lstrip('# ')
    title, _, body = block.partition('\n')
    sents, seen = [], set()
    for line in body.split('\n'):
        if line.startswith('= =') or len(line) < 15: continue
        line = PAREN.sub('', PAREN.sub('', line))
        for s in SENT.split(line):
            s = re.sub(r'\s+', ' ', s).strip()
            letters = LETTER.findall(s)
            if len(s) < 10 or not letters: continue
            if len(HANGUL.findall(s)) / len(letters) < 0.6: continue
            s = s if s.endswith('.') else s + '.'
            if not s.endswith('다.') or s in seen: continue  # 서술문만, 중복 제거(목록·표 줄 배제)
            seen.add(s); sents.append(s)
    if len(sents) >= 2:
        docs.append((title.strip(), sents))
random.seed(0); random.shuffle(docs)
with open(dst, 'w', encoding='utf-8') as f:
    for t, ss in docs:
        f.write('## %s | 한국어 위키백과\n%s\n\n' % (t, '\n'.join(ss)))
print('문서', len(docs), '문장', sum(len(s) for _, s in docs), '글자', sum(len(x) for _, s in docs for x in s))
