#!/bin/bash
# 음악 말뭉치를 처음부터 다시 만든다. (word2vec_music 폴더에서 실행)
#  1) kowikitext(한국어 위키백과 2020-09-20 덤프를 문서 단위 텍스트로 정리한 공개 데이터, CC BY-SA)를 내려받고
#  2) 정의문의 서술부가 음악 관련 명사(가수, 음반, 노래, 작곡가, 밴드, 교향곡 ...)인 문서만 고른 뒤
#  3) 괄호·목록·중복 줄을 지우고 서술문만 남겨
#  4) 규칙 기반 / Kiwi 두 가지로 토큰화한다.
# 필요: pip install kiwipiepy   (내려받는 파일 약 500MB, 압축 해제 1.7GB)
set -e
cd "$(dirname "$0")"
URL=https://github.com/lovit/kowikitext/releases/download/kowikitext.20200920.v2/kowikitext_20200920.train.zip
[ -f kowikitext_20200920.train ] || { curl -L -O "$URL"; unzip -o kowikitext_20200920.train.zip; rm kowikitext_20200920.train.zip; }
python3 filter_music.py kowikitext_20200920.train music_raw.txt
python3 clean_music.py music_raw.txt music_clean.txt
python3 tokenize_music.py music_clean.txt .
