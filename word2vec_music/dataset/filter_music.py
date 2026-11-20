"""kowikitext에서 음악 관련 문서를 골라낸다.

위키백과 문서의 첫 문장은 '《First Love》는 이루마의 두 번째 정규 음반이다.'처럼 정의문이다.
정의문의 서술부(끝 40글자)에 음악 관련 명사가 있으면 음악 문서로 본다.
사용법: python filter_music.py kowikitext_20200920.train music_raw.txt
"""
import re
import sys

TITLE = re.compile(r'^ = ([^=].*?) =\s*$')   # ' = 문서 제목 =' 줄 (소제목은 ' = = 개요 = =')
HEAD = re.compile(r'(가수|음반|앨범|싱글|노래|곡|밴드|그룹|듀오|작곡가|작사가|음악가|뮤지션|피아니스트|바이올리니스트|'
                  r'첼리스트|기타리스트|드러머|베이시스트|키보디스트|지휘자|성악가|소프라노|테너|바리톤|래퍼|프로듀서|DJ|'
                  r'악기|음악 장르|장르|오케스트라|관현악단|교향악단|합창단|가곡|교향곡|협주곡|소나타|오페라|뮤지컬|'
                  r'사운드트랙|음악 축제|음악상|레이블|음악)(이다|다|이며|이자|이었다|였다|[ ,·]|$)')


def docs(path):
    title, lines = None, []
    with open(path, encoding='utf-8') as f:
        for line in f:
            m = TITLE.match(line.rstrip('\n'))
            if m:
                if title:
                    yield title, lines
                title, lines = m.group(1).strip(), []
            elif title is not None:
                lines.append(line.rstrip('\n'))
    if title:
        yield title, lines


def main(src, dst):
    n = kept = chars = 0
    with open(dst, 'w', encoding='utf-8') as out:
        for title, lines in docs(src):
            n += 1
            if ':' in title:          # 분류:, 틀:, 위키백과: 문서 제외
                continue
            body = [l.strip() for l in lines if l.strip()]
            if not body or body[0].startswith('= ='):
                continue
            definition = re.split(r'(?<=다)\.', body[0])[0]
            if HEAD.search(definition[-40:]):
                kept += 1
                chars += sum(len(l) for l in body)
                out.write('## %s\n' % title)
                out.write('\n'.join(body) + '\n\n')
    print('전체 문서', n, '음악 문서', kept, '글자 수', chars)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
