# coding: utf-8
'''한글 문서 처리를 위한 공통 유틸리티

교재(common/util.py)의 preprocess()는 영어 문장을 공백 기준으로 나누지만,
한글은 '영화를', '영화는', '영화가'처럼 명사 뒤에 조사가 붙어 있어서
공백만으로 나누면 같은 단어가 서로 다른 어휘로 취급됩니다.

이 모듈은 외부 형태소 분석기 없이, 간단한 규칙만으로
    1) 어절(공백 단위)로 나눈 뒤
    2) 어절 끝의 자주 쓰이는 조사를 떼어내는
토크나이저를 제공합니다. 떼어낸 조사는 '-를', '-는'처럼 앞에 '-'를 붙여 구분합니다.

완벽한 형태소 분석은 아니지만(예: '나이' -> '나', '-이'), 교재의 코드를
그대로 사용해 한글 말뭉치를 실습해 보기에는 충분합니다.
더 정확한 분석이 필요하면 tokenize() 대신 KoNLPy 등의 형태소 분석기를 사용하면 됩니다.
'''
import sys
sys.path.append('..')
import re
from common.np import *


# 어절 끝에서 분리할 조사 목록(긴 것부터 검사해야 하므로 길이 순으로 정렬해 사용)
JOSA = [
    '에서는', '에서도', '에서만', '으로는', '으로도', '으로만', '으로서', '으로써', '에게는', '에게도', '에게서',
    '께서는', '께서도', '까지는', '까지도', '까지만', '부터는', '부터도', '보다는', '보다도',
    '이라고', '이라는', '이라도', '이나마', '만으로', '한테는', '한테도',
    '에서', '으로', '에게', '께서', '까지', '부터', '보다', '처럼', '마다', '밖에', '만큼',
    '조차', '마저', '이나', '이든', '이란', '이라', '라고', '라는', '라도', '한테',
    '에는', '에도', '에만', '와는', '과는', '와도', '과도', '로는', '로도', '로만', '로서', '로써',
    '은', '는', '이', '가', '을', '를', '의', '에', '로', '와', '과',
    '도', '만', '나', '랑', '께',
]
JOSA = sorted(set(JOSA), key=len, reverse=True)

# 한 글자 어간이라도 조사를 분리해도 안전한 단어들(대명사, 자주 쓰이는 한 글자 명사)
_ONE_CHAR_STEMS = set('나너우리저그이것내네제뭐왜누구무엇여기거기저기'
                      '눈돈집물불말밤비꿈몸손발피땀입코귀목등팔배맘낯뺨턱혀뼈숨넋혼힘복덕죄벌상벽탑절종'
                      '밥술쌀떡국엿죽장꽃풀잎싹씨숲산강바다섬들길논밭땅흙돌모래물새벌뱀쥐범곰소개닭말'
                      '옷신갓띠칼총활창책글붓종이방문창벽담뜰마루집절성궐궁왕님놈애형님분놈삶뜻값끝'
                      '앞뒤위밑옆안밖속겉쪽편짝밖때곳분원번살개명년월일달해별빛밤낮봄철겨울봄여름가을'
                      '일꾼짓탓맛멋빚빛잠병약독술춤노래')

# '-는'/'-은'이 조사가 아니라 동사·형용사의 어미(관형형)일 가능성이 높은 어간 끝 글자
#   예) 하는, 되는, 있는, 없는, 않는, 같은, 좋은, 많은, 작은
_VERB_ENDINGS = {
    '는': ('하', '되', '있', '없', '않', '같', '싶'),
    '은': ('좋', '많', '작', '높', '낮', '깊', '늙', '젊', '붉', '검', '밝', '굵', '얇', '짧', '넓', '좁',
          '짙', '밉', '곱', '먹', '잡', '찾', '받', '읽', '앉', '웃', '죽', '놓', '같', '있', '없', '않'),
}

# 한글 / 자모 / 한자 / 영문 / 숫자 / 문장부호 단위로 쪼개는 정규표현식
_TOKEN_RE = re.compile(r'[가-힣]+|[ㄱ-ㅎㅏ-ㅣ]+|[\u4e00-\u9fff]+|[a-zA-Z]+|[0-9]+|\.+|!+|\?+|~+|[^\s\w]')

JOSA_MARK = '-'


def split_josa(eojeol):
    '''어절 하나를 (어간, 조사)로 나눈다. 조사가 없으면 어간만 돌려준다.

    :param eojeol: 한글 어절(공백이 없는 한글 문자열)
    :return: 토큰 리스트. 예) '영화를' -> ['영화', '-를'],  '좋아한다' -> ['좋아한다']
    '''
    for josa in JOSA:  # 긴 조사부터 검사
        if not eojeol.endswith(josa):
            continue
        stem = eojeol[:-len(josa)]
        if len(stem) == 0:
            return [eojeol]
        if len(stem) == 1 and len(josa) == 1 and stem not in _ONE_CHAR_STEMS:
            return [eojeol]  # '나이' -> '나' + '-이' 같은 잘못된 분리를 피한다
        if josa in _VERB_ENDINGS and stem.endswith(_VERB_ENDINGS[josa]):
            return [eojeol]  # '하는', '좋은'은 조사가 아니라 어미
        return [stem, JOSA_MARK + josa]
    return [eojeol]


def tokenize(text, split_josa_flg=True):
    '''한글 문장을 토큰(단어) 리스트로 나눈다.

    :param text: 문장(문자열)
    :param split_josa_flg: True면 어절 끝의 조사를 '-조사' 형태로 분리한다.
    :return: 토큰 리스트
    '''
    tokens = []
    for tok in _TOKEN_RE.findall(text):
        if split_josa_flg and '가' <= tok[0] <= '힣':
            tokens += split_josa(tok)
        else:
            tokens.append(tok)
    return tokens


def tokenize_chars(text):
    '''음절(글자) 단위로 나눈다. 공백은 '␣'로 표시한다.'''
    return ['␣' if c == ' ' else c for c in text.strip()]


def detokenize(tokens):
    '''토큰 리스트를 다시 읽기 좋은 문장으로 합친다.
    '-조사' 토큰과 문장부호는 앞 단어에 붙여 쓴다.
    '''
    text = ''
    for tok in tokens:
        if tok == '<eos>':
            text += '\n'
        elif tok.startswith(JOSA_MARK) and len(tok) > 1:
            text += tok[1:]
        elif tok and tok[0] in '.,!?~)':
            text += tok
        elif text == '' or text.endswith('\n'):
            text += tok
        else:
            text += ' ' + tok
    return text


def preprocess(text, split_josa_flg=True):
    '''common/util.py의 preprocess()와 같은 역할을 하는 한글 버전

    :param text: 한글 문장(들)
    :return: (corpus, word_to_id, id_to_word)
    '''
    words = tokenize(text, split_josa_flg)

    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word


# --------------------------------------------------------------------------
# 숫자 <-> 한글 숫자(7장, 8장 데이터셋 생성에 사용)
# --------------------------------------------------------------------------
_DIGITS = ['', '일', '이', '삼', '사', '오', '육', '칠', '팔', '구']
_UNITS = ['', '십', '백', '천']


def number_to_hangul(n):
    '''0 이상 99999 이하의 정수를 한자어 숫자 읽기로 바꾼다.
    예) 0 -> '영', 12 -> '십이', 105 -> '백오', 2019 -> '이천십구', 12000 -> '만이천'
    '''
    if not (0 <= n < 100000):
        raise ValueError('0 <= n < 100000 이어야 합니다.')
    if n == 0:
        return '영'

    result = ''
    man, rest = divmod(n, 10000)
    if man > 0:
        result += (_DIGITS[man] if man > 1 else '') + '만'

    s = str(rest).zfill(4)
    for i, ch in enumerate(s):
        d = int(ch)
        if d == 0:
            continue
        unit = _UNITS[3 - i]
        if d == 1 and unit != '':
            result += unit          # '일십' 대신 '십'
        else:
            result += _DIGITS[d] + unit
    return result


# --------------------------------------------------------------------------
# matplotlib 한글 폰트 설정
# --------------------------------------------------------------------------
_FONT_CANDIDATES = [
    'AppleGothic', 'Apple SD Gothic Neo',          # macOS
    'Malgun Gothic',                               # Windows
    'NanumGothic', 'Nanum Gothic', 'NanumBarunGothic', 'NanumGothicCoding',
    'Noto Sans CJK KR', 'Noto Sans KR', 'UnDotum', 'Baekmuk Dotum',  # Linux / Colab
]


def set_korean_font(verbose=True):
    '''matplotlib에서 한글이 깨지지 않도록 사용 가능한 한글 폰트를 찾아 설정한다.

    Colab(리눅스)에서는 먼저 아래 명령으로 나눔 폰트를 설치해야 합니다.
        !apt-get -qq install fonts-nanum
        import matplotlib.font_manager as fm; fm._load_fontmanager(try_read_cache=False)
    :return: 설정된 폰트 이름(없으면 None)
    '''
    import matplotlib
    import matplotlib.font_manager as fm

    available = set(f.name for f in fm.fontManager.ttflist)
    for name in _FONT_CANDIDATES:
        if name in available:
            matplotlib.rcParams['font.family'] = name
            matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
            if verbose:
                print('matplotlib 한글 폰트: %s' % name)
            return name

    print('[경고] 한글 폰트를 찾을 수 없어 그래프의 한글이 깨질 수 있습니다.')
    print('       Colab에서는 "!apt-get -qq install fonts-nanum" 실행 후 런타임을 다시 시작하세요.')
    return None
