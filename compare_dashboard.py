import streamlit as st
import pandas as pd
from openai import OpenAI
import re
import plotly.express as px
import plotly.graph_objects as go
import os

# =================================================================
# 0. API 설정
# =================================================================
# Streamlit Secrets 또는 환경변수에서 API 키 로드 (GitHub에 노출되지 않음)
API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not API_KEY:
    st.error("❌ OpenAI API 키가 설정되지 않았습니다.\n\nStreamlit Cloud Secrets에 OPENAI_API_KEY를 입력해주세요.")
    st.stop()

client = OpenAI(api_key=API_KEY)

# =================================================================
# 1. 전처리 함수 (기존 대시보드 코드 그대로 재활용)
# =================================================================
def normalize_text(text: str) -> str:
    """텍스트를 소문자로 바꾸고 특수문자를 공백으로 정리하는 함수"""
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s가-힣+/%]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compact_text(text: str) -> str:
    """공백까지 모두 제거한 텍스트 (띄어쓰기 차이를 무시하고 검색하기 위함)"""
    return re.sub(r"\s+", "", normalize_text(text))


# =================================================================
# 1-2. 자극 언급 긍/부정 분리 분류기
# =================================================================
# 자극 관련 씨앗 단어 (이게 있어야 자극 리뷰 후보)
IRRITATION_SEEDS = ['자극', '따갑', '따가', '가렵', '가려', '쓰라', '뒤집', '좁쌀', '붉어', '아파', '아프']
IRR_WINDOW = 15  # 씨앗 단어 이후 탐색 범위(글자). 라로슈포제 실제 리뷰 검증으로 결정.

# ① 부정어/긍정 맥락: 자극 근처에 있으면 '불만 아님' (오히려 무자극 칭찬)
_POS_OVERRIDE = re.compile(
    r'(없|않|무자극|전혀|하나도|1도|별로\s*안|안\s*느|못\s*느|못느|안\s*따가|덜|진정|순하|순한|순해|편안|괜찮)'
)
# ② 효능·용도 맥락: "자극받은 피부 진정" → 제품이 자극을 해결하는 것 (제품 탓 아님)
_EFFICACY = re.compile(r'자극[가-힣\s]{0,4}(받|간|난)')
# ③ 진짜 불만 신호
_NEG_CUE = re.compile(
    r'(심하|심해|강하|강해|많아|가게|불호|거슬|따가워|따가웠|따갑다|쓰라|시려|시림|느껴|느꼈|느낀)'
)


def classify_irritation(text):
    """
    자극 언급의 성격을 판정한다.
    반환: 'neg'(진짜 불만) / 'pos'(무자극 칭찬·진정용도) / 'neutral'(애매) / 'none'(언급없음)
    핵심: 부정어·진정맥락(pos)을 먼저 배제한 뒤 진짜 불만(neg)을 찾는다. 순서 중요!
    """
    t = normalize_text(text)
    if not any(seed in t for seed in IRRITATION_SEEDS):
        return 'none'

    found_neg = False
    found_pos = False

    for seed in ['자극', '따가', '따갑', '가려', '가렵', '쓰라']:
        for m in re.finditer(seed, t):
            s = m.start()
            win = t[max(0, s - 2): s + len(seed) + IRR_WINDOW]
            # ① 긍정·부정어 우선
            if _POS_OVERRIDE.search(win) or re.search(r'자극\s*안', win):
                found_pos = True
                continue
            # ② 효능·용도 맥락
            if _EFFICACY.search(win) or '외부자극' in win or '외부 자극' in win:
                found_pos = True
                continue
            # ③ 진짜 불만
            if _NEG_CUE.search(win):
                found_neg = True
                continue

    if found_neg:
        return 'neg'
    if found_pos:
        return 'pos'
    return 'neutral'


def is_negative_irritation(text):
    """진짜 자극 불만이면 True. 기존 .str.contains() 자리에 .apply() 로 사용."""
    return classify_irritation(text) == 'neg'


# =================================================================
# 1-3. 트러블/여드름 긍(케어)/부(유발) 분리 분류기
# =================================================================
TROUBLE_SEEDS = ['트러블', '여드름', '좁쌀', '뾰루지', '뾰로지', '화농', '뾰루치']
TR_WIN_BACK = 16
TR_WIN_FWD  = 20

# 케어(긍정)·부정어: 트러블이 줄었다·사라졌다·안 생겼다·진정됐다
_TR_POS = re.compile(
    r'(진정|가라앉|잠잠|들어가|들어갔|들어간|사라|옅어|줄어|줄었|줄였|개선|완화|잡아|케어|회복|복구|효과|좋아|깨끗|'
    r'안\s*나|안나|안\s*생|안생|안\s*올라|나지\s*않|없어졌|없앤|없네|없어요|덜\s*나)'
)
# 사용 맥락(조건절): "트러블 올라올 때/났을 때 바르면" → 제품 탓 아님
_TR_CONDITIONAL = re.compile(r'(올라올|올라오려|올라오면|올라올랑|날\s*때|났을\s*때|생겼을\s*때|생기면|날때|때마다|올라왔을)')
# 걱정·가정 (실제 발생 아님)
_TR_WORRY = re.compile(r'(걱정|어떡|할까봐|날까봐|올라올까|뒤집힐까|날 것 같|날것 같|폭발할까)')
# 명시적 악화·유발
_TR_WORSEN = re.compile(r'(오히려|더\s*나게|폭탄|우다다|유발|악화|더\s*심|더\s*올라|더\s*뒤집)')
# 실망·부적합 ('안맞'은 타제품 혼동 많아 제외)
_TR_DISAPPOINT = re.compile(r'(아쉬|다른\s*제품\s*(사용|써|씀|으로|쓰)|버림|버렸|후회)')
# 제품 원인 + 결과적 발생
_TR_CAUSE = re.compile(r'(이거|이걸|이\s*제품|발라서|발랐더니|바르고\s*나|쓰고\s*나|쓰니|쓰면|사용하니|올리니|바르니)')
_TR_APPEAR = re.compile(r'(났어|났는|났네|났습|올라왔|올라와요|생겼어|생겼는|생겼습|남\b|나더라|폭발)')


def classify_trouble(text):
    """
    트러블/여드름 언급의 성격을 판정.
    'pos'(트러블 케어=장점) / 'neg'(트러블 유발·악화=단점) / 'neutral' / 'none'
    핵심: '여드름 올라올 때 바르면 진정'(사용맥락)·'날까봐 걱정'(가정)은 부정이 아니다.
    """
    t = normalize_text(text)
    if not any(seed in t for seed in TROUBLE_SEEDS):
        return 'none'
    found_pos = False
    found_neg = False
    for seed in TROUBLE_SEEDS:
        for m in re.finditer(seed, t):
            s = m.start()
            win = t[max(0, s - TR_WIN_BACK): s + len(seed) + TR_WIN_FWD]
            # ① 케어 신호·부정어 → 긍정 (최우선)
            if _TR_POS.search(win):
                found_pos = True
                continue
            # ② 조건절(사용맥락)·걱정(가정) → 부정 아님
            if _TR_CONDITIONAL.search(win) or _TR_WORRY.search(win):
                continue
            # ③ 악화·실망 → 부정
            if _TR_WORSEN.search(win) or _TR_DISAPPOINT.search(win):
                found_neg = True
                continue
            # ④ 제품 원인 + 결과 발생 → 부정
            if _TR_CAUSE.search(win) and _TR_APPEAR.search(win):
                found_neg = True
                continue
    if found_neg:
        return 'neg'
    if found_pos:
        return 'pos'
    return 'neutral'


# =================================================================
# 1-4. 분류기 기반 '계산형' 키워드를 장점/단점 카운트에 합쳐주는 함수
# =================================================================
def add_special_counts(series, counts, which):
    """
    키워드 매칭(get_kw_count)으로는 못 잡는 '맥락 분류' 항목을 합산해 추가.
    which='pros' → 트러블케어  /  which='cons' → 자극(부정), 트러블유발
    """
    if which == 'pros':
        counts['트러블케어'] = int(series.apply(lambda x: classify_trouble(x) == 'pos').sum())
    elif which == 'cons':
        counts['자극(부정)']  = int(series.apply(is_negative_irritation).sum())
        counts['트러블유발'] = int(series.apply(lambda x: classify_trouble(x) == 'neg').sum())
    return counts


def get_kw_count(series, kw_dict):
    """
    리뷰 텍스트(series)에서 키워드 사전(kw_dict)에 해당하는 표현이
    몇 건 등장하는지 '대표 이름'별로 합산해서 돌려주는 함수.
    기존 get_kw_df 와 동일한 로직이지만, 정렬/상위15개 자르기를 빼서
    비교용으로 쓰기 편하게 만들었습니다. 반환값은 {대표이름: 건수} 딕셔너리.
    """
    norm_series = series.astype(str).apply(normalize_text)
    comp_series = series.astype(str).apply(compact_text)

    # 같은 대표이름끼리 검색어를 묶기
    display_groups = {}
    for search_term, display_name in kw_dict.items():
        display_groups.setdefault(display_name, []).append(search_term)

    result = {}
    for display_name, search_terms in display_groups.items():
        k1_patterns = [re.escape(normalize_text(kw)) for kw in search_terms if normalize_text(kw)]
        k2_patterns = [re.escape(compact_text(kw)) for kw in search_terms if compact_text(kw)]
        p1 = '|'.join(k1_patterns)
        p2 = '|'.join(k2_patterns)
        count = (norm_series.str.contains(p1, na=False) |
                 comp_series.str.contains(p2, na=False)).sum()
        result[display_name] = int(count)
    return result


# =================================================================
# 2. 키워드 사전 (기존 대시보드 코드 그대로 재활용)
# =================================================================
PROS_KW_DICT = {
    '수분': '수분감', '촉촉': '수분감', '수분감': '수분감', '촉촉함': '수분감', '촉촉하다': '수분감', '수분공급': '수분감', '수분충전': '수분감',
    '보습': '보습력', '보습력': '보습력', '보습감': '보습력', '고보습': '보습력', '속보습': '보습력',
    '진정': '진정효과', '진정효과': '진정효과', '진정됨': '진정효과', '쿨링': '진정효과',
    '순하': '순함', '순함': '순함', '순하다': '순함', '저자극': '순함', '무자극': '순함',
    '장벽': '장벽강화', '장벽강화': '장벽강화', '피부장벽': '장벽강화', '재생': '장벽강화',
    '탄력': '탄력/주름', '쫀쫀': '탄력/주름', '탱탱': '탄력/주름', '주름': '탄력/주름',
    '미백': '미백/톤업', '톤업': '미백/톤업', '광채': '미백/톤업', '윤기': '미백/톤업', '환해': '미백/톤업',
    '발림': '발림성', '발림성': '발림성', '부드럽': '발림성', '매끈': '발림성',
    '흡수': '흡수력', '흡수력': '흡수력', '스며': '흡수력',
    '밀착': '밀착력', '밀착력': '밀착력',
    '지속력': '지속력', '오래감': '지속력',
    # ── Step 4 신규 추가 ──
    '화잘먹': '화잘먹', '화장이 잘 먹': '화잘먹', '화장 잘': '화잘먹', '메이크업잘': '화잘먹',
    '쿨링감': '쿨링감', '시원': '쿨링감', '시원함': '쿨링감', '청량': '쿨링감',
    '가벼움': '가벼움', '가볍': '가벼움', '가벼': '가벼움', '산뜻': '가벼움',
    # ── 향 긍정: 무향·저자극향 언급 (장점으로 트래킹) ──
    '무취무향': '무향/저자극향', '무향': '무향/저자극향', '무취': '무향/저자극향',
    '향이 없': '무향/저자극향', '향 없': '무향/저자극향', '향 안 나': '무향/저자극향',
    '향이 안 나': '무향/저자극향', '향 심하지 않': '무향/저자극향',
    '향 첨가하지않': '무향/저자극향', '향을 첨가하지않': '무향/저자극향',
    '인공향 무첨가': '무향/저자극향', '향료 무첨가': '무향/저자극향',
    '향 거슬리지 않': '무향/저자극향', '향이 거슬리지 않': '무향/저자극향',
    '향 무난': '무향/저자극향',
}

CONS_KW_DICT = {
    '무거': '무거움', '무거움': '무거움',
    '묽': '묽음', '너무묽': '묽음',
    '비싸': '비쌈', '비쌈': '비쌈', '가격부담': '비쌈',
    '건조': '건조함', '당김': '건조함', '속건조': '건조함',
    '끈적': '끈적임', '끈적임': '끈적임',
    '번들': '번들거림', '유분': '번들거림', '유분기': '번들거림',
    '밀림': '화장밀림', '겉돌': '화장밀림',
    # '자극/트러블' 묶음은 제거 → 자극(부정)·트러블유발 로 분리 계산 (add_special_counts)
    # ── 향 단점: 부정·호불호 맥락만 정밀 포착 ('향' 단독 키워드 제거) ──
    # [제거 이유] '향' 한 글자는 '의향', '취향', '성향' 등 무관한 단어를 대량 오탐
    # [추가 원칙] 반드시 부정 맥락이 포함된 구절만 등록
    '향이 별로': '향(호불호)', '향 별로': '향(호불호)',
    '향이 강': '향(호불호)', '향이 독': '향(호불호)', '향이 진': '향(호불호)',
    '향이 안좋': '향(호불호)', '향이 싫': '향(호불호)',
    '향이 거슬': '향(호불호)', '향 거슬': '향(호불호)',
    '향 적응': '향(호불호)', '향에 적응': '향(호불호)',       # "향 적응이 안 돼요"
    '향 때문에': '향(호불호)',                                # "향 때문에 거부감"
    '향에 실망': '향(호불호)',
    '향이 실망': '향(호불호)',
    '호불호': '향(호불호)',                                   # "향 호불호 갈림"
    '불호': '향(호불호)',                                     # "대부분 불호일 거예요"
    '향기 불호': '향(호불호)', '향이 불호': '향(호불호)',
    '향이 문제': '향(호불호)', '향만 아니라면': '향(호불호)',
    '향만 아니면': '향(호불호)',
    '고무냄새': '향(호불호)', '고무 냄새': '향(호불호)',      # 라로슈포제 실제 리뷰 기반
    '목공풀': '향(호불호)',                                   # 실제 리뷰에 등장한 표현
    '약냄새': '향(호불호)', '약 냄새': '향(호불호)',
    '연고냄새가': '향(호불호)', '연고 냄새가': '향(호불호)',   # "연고냄새가 나서 별로"
    '냄새가 강': '향(호불호)', '냄새가 독': '향(호불호)',
    '냄새가 싫': '향(호불호)', '냄새 별로': '향(호불호)',
    '냄새가 별로': '향(호불호)', '냄새 때문에': '향(호불호)',
    '냄새가 거슬': '향(호불호)',
    '인공향': '향(호불호)',                                   # "인공향 첨가"는 부정 맥락
    '향료': '향(호불호)',                                     # "향료 들어있어서"
    '답답': '답답함', '답답함': '답답함'
}

LOYALTY_KW_DICT = {
    '다회구매': '다회구매(n통)', 'n통': '다회구매(n통)', '몇통': '다회구매(n통)',
    '상비약': '상비약/필수템', '필수템': '상비약/필수템',
    '신뢰': '신뢰/만족', '믿고씀': '신뢰/만족', '만족': '신뢰/만족', '꾸준히': '신뢰/만족',
}

TRIGGER_KW_DICT = {
    '추천': '지인추천', '지인추천': '지인추천', '친구추천': '지인추천', '가족추천': '지인추천', '추천받': '지인추천',
    '기획': '기획상품', '기획세트': '기획상품', '1+1': '기획상품', '세트': '기획상품',
    '리뷰': '후기', '후기': '후기',
    '유튜버': '유튜버/인플루언서 추천', '인플루언서': '유튜버/인플루언서 추천',
    '광고': '광고',
    '샘플': '샘플체험', '테스트': '샘플체험',
    '세일': '할인/세일', '할인': '할인/세일',
    '성분': '성분', '가성비': '가성비'
}

TPO_KW_DICT = {
    '메이크업 전': '화장전', '화장 전': '화장전', '화장전': '화장전',
    '밤': '밤', '자기 전': '밤', '자기전': '밤', '취침': '밤', '잠들기 전': '밤',
    '아침': '아침', '점심': '점심', '저녁': '저녁',
    '환절기': '환절기', '겨울': '겨울', '여름': '여름',
    '세안 후': '세안 후', '샤워 후': '세안 후',
    '운동': '운동 후', '수영': '운동 후'
}

# =================================================================
# 3. AI 시스템 프롬프트 (비교분석용으로 살짝 수정)
# =================================================================
SYSTEM_PROMPT = """
너는 올리브영 카테고리 MD + BM 컨설턴트 듀얼 역할을 수행하는 뷰티 데이터 전문가야.
MD 관점: 어떤 제품을 어떤 고객에게 추천할지 결정.
BM 관점: 우리 브랜드가 이 카테고리에서 어디에 포지셔닝할지 전략 수립.
전달받은 [정확한 수치 데이터]를 바탕으로 '경쟁 제품 비교분석 리포트'를 작성해줘.

━━━━━━━━━━━━━━━━━━━━━━━━━━
[절대 규칙 — 어기면 리포트 무효]
━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 제목 체계: 대제목 #, 중제목 ##, 소제목 ####. 모든 제목 한국어. 별표(**) 금지.
2. 수치 3종 세트: 모든 추천·분석에는 반드시
   [제품명 + 핵심수치 + 카테고리평균 대비] 3가지를 함께 표기.
   예: 에스네이처 (수분감 64.2%, 카테고리 평균 +19.1%p)
   수치는 반드시 <span style="color:#777777">(수치)</span> 로 감싸.
3. 상위 3개 병기: "1등 제품" 만 쓰지 말고 "1등(n%) > 2등(n%) > 3등(n%)" 형태로 항상 3개 병기.
4. 전수 언급: 리포트 전체에서 입력된 모든 제품이 최소 1회 이상 언급되어야 함.
   한 번도 안 나온 제품은 마지막에 '미언급 제품 및 사유' 섹션에서 별도 설명.
5. Actionability: 모든 전략 제안은 "타깃 / 메시지 / 채널 / 예상 KPI" 4요소 포함.
   "~을 고려해야 한다" 같은 모호한 표현 금지. "~한다" 명령형으로 작성.
6. 표 강제: 매트릭스 섹션(피부타입·피부고민·TPO·장점·단점별 추천)은
   반드시 마크다운 표(| 헤더 | 헤더 |) 형식으로 출력.
7. 인사말 생략: 바로 # [경쟁 제품 비교분석 리포트] 제목부터 시작.

━━━━━━━━━━━━━━━━━━━━━━━━━━
[리포트 구조]
━━━━━━━━━━━━━━━━━━━━━━━━━━
# [경쟁 제품 비교분석 리포트]

## 1. 시장 한눈에 보기
아래 두 항목 모두 반드시 마크다운 표(| 헤더 | 헤더 |) 형식으로 출력.

#### 종합 순위 Top 3
표 형식으로 출력. 컬럼: 순위 | 제품명 | 평균평점 | 재구매율(%) | 가격만족도
(평균평점 50% + 재구매율 30% + 가격만족도 20% 복합점수 기준. 전달받은 [종합 순위 Top 3] 데이터를 그대로 사용할 것.)

#### 카테고리 전체 평균
표 형식으로 출력. 컬럼: 평균평점 | 재구매율(%) | 가격만족도

## 2. 피부타입별 추천 매트릭스
표 형식으로 출력. 컬럼: 피부타입 | 1순위(근거) | 2순위(근거) | 3순위(근거) | 비추(근거)
건성 / 지성 / 복합성 / 민감성 / 트러블성 모두 작성.

[선정 기준 — 절대 다른 지표 혼용 금지]
- 1~3순위: 전달받은 [제품×피부타입 5점 비율(%)] 에서 해당 피부타입 값이 높은 순서대로 선정.
  근거 표기 형식 → 제품명 (해당피부타입 5점비율 XX.X%)
  예: 에스네이처 (건성 5점비율 78.2%)
- 비추: 전달받은 [제품×피부타입 1~2점 비율(%)] 에서 해당 피부타입 값이 가장 높은 제품.
  근거 표기 형식 → 제품명 (해당피부타입 1~2점비율 XX.X%)
  예: 라로슈포제 (건성 1~2점비율 8.3%)

## 3. 피부고민별 추천 매트릭스
표 형식으로 출력. 컬럼: 피부고민 | 추천 제품(근거 수치)
잡티 / 미백 / 주름 / 트러블 / 모공 / 건조함 포함.

[선정 기준 — 절대 다른 지표 혼용 금지]
- 전달받은 [제품×피부고민 5점 비율(%)] 에서 해당 피부고민 값이 가장 높은 제품.
  근거 표기 형식 → 제품명 (해당고민 5점비율 XX.X%)
  예: 바이오힐보 (건조함 5점비율 82.1%)

## 4. TPO별 추천 매트릭스
표 형식으로 출력. 컬럼: 상황 | 추천 제품(근거 수치)
아침 / 밤 / 화장 전 / 환절기 / 여름 / 겨울 포함.

[선정 기준 — 절대 다른 지표 혼용 금지]
- 전달받은 [제품별 TPO 언급률(%)] 에서 해당 상황 값이 가장 높은 제품.
  근거 표기 형식 → 제품명 (해당TPO 언급률 XX.X%)
  예: 토리든 (밤 언급률 18.3%)

## 5. 장점 키워드별 1등 제품
표 형식으로 출력. 컬럼: 장점 항목 | 1등 제품(언급률%) | 2등 제품(언급률%) | 카테고리 평균(%)

## 6. 단점 키워드별 최다 언급 제품
표 형식으로 출력. 컬럼: 단점 항목 | 최다 언급 제품(언급률%) | 최소 언급 제품(언급률%) | 카테고리 평균(%)

## 7. 데이터 기반 전략 제안
#### 신제품 포지셔닝 공백 분석 (어느 피부타입×고민 조합이 공략되지 않았는가)
#### 카테고리 1위 추격 차별화 전략 3가지
각 전략마다 아래 4요소를 반드시 포함:
- 타깃:
- 메시지:
- 채널:
- 예상 KPI:
"""

# =================================================================
# 4. 페이지 기본 설정
# =================================================================
st.set_page_config(page_title="크림 Top 10 리뷰 비교 분석 대시보드", page_icon="🆚", layout="wide")


# =================================================================
# 5. 멀티 파일 → 하나의 통합 DataFrame 만들기
# =================================================================
def load_and_merge(uploaded_files):
    """
    업로드된 여러 CSV를 하나로 합치는 함수.
    각 파일에 'product'(제품명) 컬럼을 새로 붙여서 제품을 구분합니다.
    제품명은 파일 이름에서 .csv 를 떼어내 사용합니다.
    """
    frames = []
    for f in uploaded_files:
        product_name = f.name.replace('.csv', '')
        one_df = pd.read_csv(f)
        # 같은 제품 내 중복 리뷰 제거
        one_df = one_df.drop_duplicates(subset=['content'], keep='first').copy()
        one_df['product'] = product_name  # 제품 구분용 컬럼 추가
        frames.append(one_df)
    merged = pd.concat(frames, ignore_index=True)
    merged['rating'] = pd.to_numeric(merged['rating'], errors='coerce')
    return merged


def make_trouble_df(df):
    """
    skinTrouble 컬럼은 '잡티, 미백' 처럼 콤마로 여러 개가 들어있어서
    한 행에 하나의 고민만 오도록 explode(펼치기) 하는 함수.
    product 컬럼도 함께 유지합니다.
    """
    t = df[['product', 'rating', 'skinTrouble']].copy().dropna()
    t['skinTrouble'] = t['skinTrouble'].astype(str).str.split(',')
    t = t.explode('skinTrouble')
    t['skinTrouble'] = t['skinTrouble'].str.strip()
    t = t[t['skinTrouble'] != '']
    return t


# =================================================================
# 6. 제품별 핵심 지표를 한 줄로 요약하는 함수 (스코어카드용)
# =================================================================
def build_summary_table(df):
    """
    제품별로 평점 / 리뷰수 / 재구매율 / 자극 언급률 등을 계산해
    '제품 1줄 = 1행'인 요약 테이블을 만드는 함수.
    개요 비교 차트와 레이더 차트의 재료가 됩니다.
    """
    rows = []
    for product, g in df.groupby('product'):
        n = len(g)
        avg_rating = g['rating'].mean()
        # 재구매율: repurchase 컬럼이 'true'인 비율
        repurchase_rate = (g['repurchase'].astype(str).str.lower() == 'true').mean() * 100
        # 5점 비율
        five_rate = (g['rating'] == 5).mean() * 100
        # 자극 불만률 (긍/부정 분리 분류기 적용 — 진짜 불만만 카운트)
        irritation_n = g['content'].apply(is_negative_irritation).sum()
        irritation_rate = irritation_n / n * 100 if n else 0

        # 가격 만족도 지수 계산
        # 긍정(가성비·합리적·가격대비) × 1.0 + 중립(세일·할인) × 0.5 - 부정(비쌈·가격부담) × 1.0
        pos_price = g['content'].str.contains('가성비|합리적|가격대비|저렴|착하', na=False).sum()
        neu_price = g['content'].str.contains('세일|올영세일|할인|프로모션|기획', na=False).sum()
        neg_price = g['content'].str.contains('비싸|비쌈|가격부담|비싸다|돈아까', na=False).sum()
        price_score = round((pos_price * 1.0 + neu_price * 0.5 - neg_price * 1.0) / n * 100, 1) if n else 0

        rows.append({
            'product': product,
            '리뷰수': n,
            '평균평점': round(avg_rating, 2),
            '5점비율(%)': round(five_rate, 1),
            '재구매율(%)': round(repurchase_rate, 1),
            '자극언급률(%)': round(irritation_rate, 1),
            '가격만족도지수': price_score,
        })
    return pd.DataFrame(rows)


# =================================================================
# 7. 제품 x 키워드 히트맵용 데이터 만드는 함수
# =================================================================
def build_keyword_matrix(df, kw_dict, normalize=True, special=None):
    """
    행=제품, 열=키워드, 값=언급건수(또는 비율) 인 표를 만드는 함수.
    히트맵 차트에 바로 넣을 수 있습니다.
    normalize=True 이면 제품별 리뷰수로 나눠 비율(%)로 바꿔
    리뷰수가 다른 제품끼리도 공정하게 비교할 수 있게 합니다.
    special='pros'/'cons' 이면 분류기 기반 계산형 항목도 함께 추가합니다.
    """
    rows = []
    for product, g in df.groupby('product'):
        counts = get_kw_count(g['content'], kw_dict)
        if special:
            counts = add_special_counts(g['content'], counts, special)
        if normalize:
            n = len(g)
            counts = {k: (v / n * 100 if n else 0) for k, v in counts.items()}
        counts['product'] = product
        rows.append(counts)
    matrix = pd.DataFrame(rows).set_index('product')
    return matrix.fillna(0)


# =================================================================
# 8. 메인 화면
# =================================================================
st.title("🆚 크림 Top 10 리뷰 비교 분석 대시보드")
st.caption("올리브영 크림 카테고리 판매순 Top 10 제품의 리뷰 데이터를 자동으로 비교 분석합니다.")

# --- GitHub 저장소에서 CSV 자동 로드 ---
CSV_FILES = [
    "VT 피디알엔 캡슐 크림 100.csv",
    "라로슈포제시카플라스트 밤 B+.csv",
    "바이오힐보 프로바이오덤 3D 리프팅 크림.csv",
    "아누아 피디알엔 히알루론산 100 수분 크림.csv",
    "에스네이처 아쿠아 스쿠알란 수분크림.csv",
    "에스네이처 아쿠아 오아시스 수분 젤크림.csv",
    "에스트라 아토베리어 365크림.csv",
    "제로이드 수딩 크림.csv",
    "토리든 다이브인 히알루론산 수딩 크림.csv",
    "피지오겔 DMT 페이셜 크림.csv",
]

@st.cache_data
def load_all_csv():
    frames = []
    for filename in CSV_FILES:
        try:
            one_df = pd.read_csv(filename)
            one_df = one_df.drop_duplicates(subset=['content'], keep='first').copy()
            one_df['product'] = filename.replace('.csv', '')
            frames.append(one_df)
        except FileNotFoundError:
            st.warning(f"⚠️ 파일을 찾을 수 없습니다: {filename}")
    if not frames:
        st.error("❌ 데이터 파일을 불러올 수 없습니다. GitHub에 CSV 파일이 있는지 확인해주세요.")
        st.stop()
    merged = pd.concat(frames, ignore_index=True)
    merged['rating'] = pd.to_numeric(merged['rating'], errors='coerce')
    return merged

# --- 데이터 통합 ---
df = load_all_csv()
trouble_df = make_trouble_df(df)
product_list = sorted(df['product'].unique())
summary = build_summary_table(df)  # Step3 이후 탭들도 사용하므로 전역에서 한 번만 계산

# --- 제품 리스트 (리뷰수 표시) ---
total_reviews = len(df)
st.markdown(
    f"크림 카테고리 {len(product_list)}개 제품의 리뷰 "
    f"(총 **{total_reviews:,}건**)를 분석했습니다."
)
# 리뷰수 기준으로 내림차순 표시
summary_sorted = summary.sort_values('리뷰수', ascending=False).reset_index(drop=True)
product_list_text = "  \n".join(
    [f"* {row['product']} ({row['리뷰수']:,}건)"
     for i, row in summary_sorted.iterrows()]
)
with st.expander("📋 분석 제품 리스트 보기"):
    st.markdown(product_list_text)

st.markdown("---")

# --- 탭 분리 ---
tab_compare, tab_detail, tab_ai, tab_recommend = st.tabs(
    ["📊 제품 비교 분석", "🔍 개별 제품 상세", "🤖 AI 비교 전략 리포트", "🎯 맞춤 제품 추천"]
)


# =================================================================
# 탭 1) 제품 비교 뷰
# =================================================================
with tab_compare:

    # ----- 섹션 ① 제품 개요 비교 -----
    st.header("① 제품 개요 비교")

    # 평점 1~3위 색상 강조용 컬러 리스트 계산
    def make_rank_colors(series, highlight_color='#E8404A', base_color='#A8BAC4', top_n=3):
        """상위 top_n개는 강조색, 나머지는 회색으로 색상 리스트 반환"""
        ranked = series.rank(ascending=False)
        return [highlight_color if r <= top_n else base_color for r in ranked]

    s1, s2 = st.columns(2)

    with s1:
        st.subheader("⭐ 평균 평점")
        rating_sorted = summary.sort_values('평균평점', ascending=True)
        colors = make_rank_colors(rating_sorted['평균평점'])
        fig = go.Figure(go.Bar(
            x=rating_sorted['평균평점'],
            y=rating_sorted['product'],
            orientation='h',
            text=rating_sorted['평균평점'],
            textposition='outside',
            marker_color=colors,
        ))
        fig.update_layout(
            xaxis=dict(range=[3.0, 5.0], title='평균 평점'),
            yaxis=dict(title=''),
            height=400,
            margin=dict(l=10, r=60, t=30, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    with s2:
        st.subheader("🔄 재구매자 비율(%)")
        rep_sorted = summary.sort_values('재구매율(%)', ascending=True)
        colors2 = make_rank_colors(rep_sorted['재구매율(%)'])
        fig = go.Figure(go.Bar(
            x=rep_sorted['재구매율(%)'],
            y=rep_sorted['product'],
            orientation='h',
            text=rep_sorted['재구매율(%)'],
            textposition='outside',
            marker_color=colors2,
        ))
        fig.update_layout(
            xaxis=dict(range=[0, 20], title='재구매자 비율 (%)'),
            yaxis=dict(title=''),
            height=400,
            margin=dict(l=10, r=60, t=30, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ----- 섹션 ② 타깃 고객 비교 -----
    st.header("② 타깃 고객 비교")

    # ── 2-1. 제품별 피부타입 분포 (100% 누적 + % 라벨) ──
    st.subheader("🧴 제품별 피부타입 분포 (%)")
    skin_ct = df.groupby(['product', 'skinType']).size().reset_index(name='건수')
    skin_total = skin_ct.groupby('product')['건수'].transform('sum')
    skin_ct['비율'] = (skin_ct['건수'] / skin_total * 100).round(1)
    skin_ct['라벨'] = skin_ct['비율'].astype(str) + '%'
    fig = px.bar(
        skin_ct, x='product', y='비율', color='skinType',
        barmode='stack', text='라벨',
        labels={'비율': '비율 (%)', 'product': '제품'},
    )
    fig.update_traces(textposition='inside', insidetextanchor='middle')
    fig.update_layout(yaxis=dict(range=[0, 100], title='비율 (%)'), height=420)
    st.plotly_chart(fig, use_container_width=True)

    # ── 2-2. 제품별 피부고민 히트맵 (% 통일) ──
    st.subheader("💡 구매자 피부고민 히트맵 (%)")
    trouble_ct = trouble_df.groupby(['product', 'skinTrouble']).size().reset_index(name='건수')
    # 제품별 리뷰수로 나눠 % 변환
    product_n = df.groupby('product').size().reset_index(name='total')
    trouble_ct = trouble_ct.merge(product_n, on='product')
    trouble_ct['비율'] = (trouble_ct['건수'] / trouble_ct['total'] * 100).round(1)
    pivot_trouble = trouble_ct.pivot(index='product', columns='skinTrouble', values='비율').fillna(0)
    fig = px.imshow(
        pivot_trouble, text_auto=True, aspect='auto',
        color_continuous_scale='Blues',
        labels=dict(color='분포율(%)'),
    )
    fig.update_traces(texttemplate='%{z:.1f}%')
    st.plotly_chart(fig, use_container_width=True)

    # ── 2-3. 5점 리뷰어 피부타입 & 피부고민 히트맵 ──
    st.subheader("⭐ 5점 리뷰어 피부타입 & 피부고민 히트맵")
    h1, h2 = st.columns(2)
    with h1:
        df_5 = df[df['rating'] == 5]
        skin5_ct = df_5.groupby(['product', 'skinType']).size().reset_index(name='건수')
        product_5n = df_5.groupby('product').size().reset_index(name='total')
        skin5_ct = skin5_ct.merge(product_5n, on='product')
        skin5_ct['비율'] = (skin5_ct['건수'] / skin5_ct['total'] * 100).round(1)
        pivot_skin5 = skin5_ct.pivot(index='product', columns='skinType', values='비율').fillna(0)
        fig = px.imshow(pivot_skin5, text_auto=True, aspect='auto',
                        color_continuous_scale='Greens',
                        labels=dict(color='비율(%)'))
        fig.update_traces(texttemplate='%{z:.1f}%')
        fig.update_layout(title='피부타입 분포도')
        st.plotly_chart(fig, use_container_width=True)
    with h2:
        trouble_5 = trouble_df[trouble_df['rating'] == 5]
        tr5_ct = trouble_5.groupby(['product', 'skinTrouble']).size().reset_index(name='건수')
        tr5_ct = tr5_ct.merge(product_5n, on='product')
        tr5_ct['비율'] = (tr5_ct['건수'] / tr5_ct['total'] * 100).round(1)
        pivot_tr5 = tr5_ct.pivot(index='product', columns='skinTrouble', values='비율').fillna(0)
        fig = px.imshow(pivot_tr5, text_auto=True, aspect='auto',
                        color_continuous_scale='Greens',
                        labels=dict(color='비율(%)'))
        fig.update_traces(texttemplate='%{z:.1f}%')
        fig.update_layout(title='피부고민 분포도')
        st.plotly_chart(fig, use_container_width=True)

    # ── 2-4. 1~2점 리뷰어 피부타입 & 피부고민 히트맵 ──
    st.subheader("💔 1~2점 리뷰어 피부타입 & 피부고민 히트맵")
    h3, h4 = st.columns(2)
    with h3:
        df_1 = df[df['rating'] <= 2]
        if len(df_1) > 0:
            skin1_ct = df_1.groupby(['product', 'skinType']).size().reset_index(name='건수')
            product_1n = df_1.groupby('product').size().reset_index(name='total')
            skin1_ct = skin1_ct.merge(product_1n, on='product')
            skin1_ct['비율'] = (skin1_ct['건수'] / skin1_ct['total'] * 100).round(1)
            pivot_skin1 = skin1_ct.pivot(index='product', columns='skinType', values='비율').fillna(0)
            fig = px.imshow(pivot_skin1, text_auto=True, aspect='auto',
                            color_continuous_scale='Reds',
                            labels=dict(color='비율(%)'))
            fig.update_traces(texttemplate='%{z:.1f}%')
            fig.update_layout(title='피부타입 분포도')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("1~2점 리뷰 데이터가 없습니다.")
    with h4:
        trouble_1 = trouble_df[trouble_df['rating'] <= 2]
        if len(trouble_1) > 0:
            tr1_ct = trouble_1.groupby(['product', 'skinTrouble']).size().reset_index(name='건수')
            product_1n = df_1.groupby('product').size().reset_index(name='total')
            tr1_ct = tr1_ct.merge(product_1n, on='product')
            tr1_ct['비율'] = (tr1_ct['건수'] / tr1_ct['total'] * 100).round(1)
            pivot_tr1 = tr1_ct.pivot(index='product', columns='skinTrouble', values='비율').fillna(0)
            fig = px.imshow(pivot_tr1, text_auto=True, aspect='auto',
                            color_continuous_scale='Reds',
                            labels=dict(color='비율(%)'))
            fig.update_traces(texttemplate='%{z:.1f}%')
            fig.update_layout(title='피부고민 분포도')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("1~2점 리뷰 데이터가 없습니다.")

    st.markdown("---")

    # ----- 섹션 ③ 제품 경험(UX) 비교 -----
    st.header("③ 제품 경험(UX) 비교")

    # ── 3-1. 장점 히트맵 (카테고리 평균 대비 +n%p 강조) ──
    st.subheader("👍 장점 키워드 히트맵 — 평균 대비 언급률 편차 (%p)")
    pros_matrix = build_keyword_matrix(df, PROS_KW_DICT, normalize=True, special='pros')
    # 각 키워드의 카테고리 평균을 빼서 '평균 대비 차이'로 변환
    pros_diff = pros_matrix.subtract(pros_matrix.mean(axis=0), axis=1).round(1)
    fig = px.imshow(
        pros_diff, text_auto=True, aspect='auto',
        color_continuous_scale='RdYlGn',   # 빨강(낮음) → 노랑 → 초록(높음)
        color_continuous_midpoint=0,
        labels=dict(color='평균 대비 %p'),
    )
    fig.update_traces(texttemplate='%{z:+.1f}%p')
    st.plotly_chart(fig, use_container_width=True)

    # ── 3-2. 단점 히트맵 (카테고리 평균 대비 +n%p 강조) ──
    st.subheader("👎 단점 키워드 히트맵 — 평균 대비 언급률 편차 (%p)")
    cons_matrix = build_keyword_matrix(df, CONS_KW_DICT, normalize=True, special='cons')
    cons_diff = cons_matrix.subtract(cons_matrix.mean(axis=0), axis=1).round(1)
    fig = px.imshow(
        cons_diff, text_auto=True, aspect='auto',
        color_continuous_scale='RdYlGn_r',  # 단점은 반전: 높을수록 빨강
        color_continuous_midpoint=0,
        labels=dict(color='평균 대비 %p'),
    )
    fig.update_traces(texttemplate='%{z:+.1f}%p')
    st.plotly_chart(fig, use_container_width=True)

    # ── 3-3. 자극도 관련 리뷰 비율 막대 그래프 ──
    st.subheader("🧪 자극도 관련 리뷰 비율 비교")
    irr_rows = []
    for product, g in df.groupby('product'):
        n = len(g)
        cls    = g['content'].apply(classify_irritation)
        irr_n  = (cls == 'neg').sum()
        mild_n = (cls == 'pos').sum()
        none_n = n - irr_n - mild_n
        irr_rows.append({'product': product, '구분': '자극 불만(부정)', '비율': round(irr_n / n * 100, 1)})
        irr_rows.append({'product': product, '구분': '무자극·진정(긍정)', '비율': round(mild_n / n * 100, 1)})
        irr_rows.append({'product': product, '구분': '언급 없음', '비율': round(none_n / n * 100, 1)})
    irr_df = pd.DataFrame(irr_rows)
    fig = px.bar(
        irr_df, x='product', y='비율', color='구분',
        barmode='stack', text='비율',
        color_discrete_map={'자극 불만(부정)': '#E8404A', '무자극·진정(긍정)': '#4CAF50', '언급 없음': '#B0BEC5'},
        labels={'비율': '비율 (%)', 'product': '제품'},
    )
    fig.update_traces(texttemplate='%{y:.1f}%', textposition='inside')
    fig.update_layout(yaxis=dict(range=[0, 100]), height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ----- 섹션 ④ 구매 결정 요인 비교 -----
    st.header("④ 구매 결정 요인 비교")

    # ── 4-1. 구매 트리거 히트맵 ──
    st.subheader("🛒 구매 트리거 비교 (%)")
    trigger_matrix = build_keyword_matrix(df, TRIGGER_KW_DICT, normalize=True)
    fig = px.imshow(
        trigger_matrix.round(1), text_auto=True, aspect='auto',
        color_continuous_scale='Purples',
        labels=dict(color='언급률(%)'),
    )
    fig.update_traces(texttemplate='%{z:.1f}%')
    st.plotly_chart(fig, use_container_width=True)

    # ── 4-2. TPO 히트맵 ──
    st.subheader("⏰ TPO(사용 상황) 히트맵 (%)")
    tpo_matrix = build_keyword_matrix(df, TPO_KW_DICT, normalize=True)
    fig = px.imshow(
        tpo_matrix.round(1), text_auto=True, aspect='auto',
        color_continuous_scale='Blues',
        labels=dict(color='언급률(%)'),
    )
    fig.update_traces(texttemplate='%{z:.1f}%')
    st.plotly_chart(fig, use_container_width=True)

    # ── 4-3. 재구매 리뷰 핵심 구매 요인 분석 ──
    st.subheader("💖 재구매 리뷰 핵심 구매 요인 분석 (%)")
    st.caption("재구매자 리뷰에서 장점·로열티 키워드 언급률 — 재구매를 이끈 핵심 요인")
    combined_loyalty = {**LOYALTY_KW_DICT, **PROS_KW_DICT}
    loyalty_rows = []
    for product, g in df.groupby('product'):
        rep = g[g['repurchase'].astype(str).str.lower() == 'true']['content']
        n_rep = len(rep)
        if n_rep == 0:
            continue
        counts = get_kw_count(rep, combined_loyalty)
        for kw, cnt in counts.items():
            loyalty_rows.append({
                'product': product,
                '키워드': kw,
                '언급률': round(cnt / n_rep * 100, 1),
            })
    if loyalty_rows:
        loyalty_long = pd.DataFrame(loyalty_rows)
        loyalty_pivot = loyalty_long.pivot(index='product', columns='키워드', values='언급률').fillna(0)
        fig = px.imshow(
            loyalty_pivot.round(1), text_auto=True, aspect='auto',
            color_continuous_scale='Greens',
            labels=dict(color='언급률(%)'),
        )
        fig.update_traces(texttemplate='%{z:.1f}%')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("재구매 리뷰 데이터가 없습니다.")

    # ── 4-4. 가격 만족도 지수 비교 ──
    st.subheader("💰 가격 만족도 지수 비교")
    st.caption("지수 = (긍정×1.0 + 중립×0.5 - 부정×1.0) / 전체 리뷰수 × 100  |  높을수록 가격 만족도 우수")
    price_sorted = summary.sort_values('가격만족도지수', ascending=True)
    price_colors = make_rank_colors(price_sorted['가격만족도지수'])
    fig = go.Figure(go.Bar(
        x=price_sorted['가격만족도지수'],
        y=price_sorted['product'],
        orientation='h',
        text=price_sorted['가격만족도지수'],
        textposition='outside',
        marker_color=price_colors,
    ))
    fig.update_layout(
        xaxis=dict(title='가격 만족도 지수'),
        yaxis=dict(title=''),
        height=400,
        margin=dict(l=10, r=60, t=30, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ----- 섹션 ⑤ 종합 경쟁력 레이더 + 포지셔닝 맵 -----
    st.header("⑤ 종합 경쟁력 레이더 & 포지셔닝 맵")

    r1, r2 = st.columns(2)

    # ── 5-1. 레이더 차트 (5개 축으로 재정의) ──
    with r1:
        st.subheader("🕸️ 종합 경쟁력 레이더")
        st.caption("축: 5점비율 / 재구매율 / 가격만족도 / 피부안전성 / 핵심효능 (값이 클수록 우수)")

        radar_df = summary.copy()

        # 핵심효능 = 수분감 + 보습력 언급률 합산 (pros_matrix 재활용)
        # pros_matrix 는 위 섹션 ③에서 이미 계산됨
        core_efficacy = pd.Series(index=pros_matrix.index, dtype=float)
        for prod in pros_matrix.index:
            moist = pros_matrix.loc[prod, '수분감'] if '수분감' in pros_matrix.columns else 0
            hydra  = pros_matrix.loc[prod, '보습력'] if '보습력' in pros_matrix.columns else 0
            core_efficacy[prod] = round(moist + hydra, 1)

        radar_df = radar_df.set_index('product')
        radar_df['핵심효능'] = core_efficacy
        radar_df['피부안전성'] = 100 - radar_df['자극언급률(%)']   # 자극 적을수록 높음
        radar_df['가격만족도_정규화'] = radar_df['가격만족도지수'].clip(lower=0)  # 음수 방지

        # 레이더 축마다 스케일이 달라 min-max 정규화 (0~100)
        def minmax(s):
            mn, mx = s.min(), s.max()
            return (s - mn) / (mx - mn) * 100 if mx != mn else s * 0 + 50

        radar_axes = ['5점비율(%)', '재구매율(%)', '가격만족도_정규화', '피부안전성', '핵심효능']
        radar_labels = ['5점 비율', '재구매율', '가격 만족도', '피부 안전성', '핵심 효능']

        fig = go.Figure()
        for prod, row in radar_df.iterrows():
            vals_raw = [row[a] for a in radar_axes]
            # 정규화된 값으로 그래프 (hover에는 원본 수치 표시)
            fig.add_trace(go.Scatterpolar(
                r=[minmax(radar_df[a])[prod] for a in radar_axes],
                theta=radar_labels,
                fill='toself',
                name=prod,
                hovertemplate=(
                    f"<b>{prod}</b><br>" +
                    "<br>".join([f"{lb}: {row[ax]:.1f}" for lb, ax in zip(radar_labels, radar_axes)]) +
                    "<extra></extra>"
                ),
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            height=500,
            legend=dict(orientation='v', x=1.05),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── 5-2. 피부타입 포지셔닝 맵 (복합성 기준 예시) ──
    with r2:
        st.subheader("🗺️ 피부타입 포지셔닝 맵")
        st.caption("X축: 해당 피부타입 리뷰어 비율(%) | Y축: 해당 피부타입 평균 평점 | 버블 크기: 재구매율")

        SKIN_TYPES = ['건성', '지성', '복합성', '민감성', '트러블성', '약건성', '중성']
        selected_skin = st.selectbox(
            "포지셔닝 맵 기준 피부타입 선택",
            SKIN_TYPES, index=2, key='positioning_skin'
        )

        pos_rows = []
        for product, g in df.groupby('product'):
            n = len(g)
            skin_sub = g[g['skinType'] == selected_skin]
            n_skin = len(skin_sub)
            if n_skin == 0:
                continue
            ratio = round(n_skin / n * 100, 1)
            avg_rt = round(skin_sub['rating'].mean(), 2)
            rep_rate = round(
                (g['repurchase'].astype(str).str.lower() == 'true').mean() * 100, 1
            )
            pos_rows.append({
                'product': product,
                '리뷰어비율(%)': ratio,
                '평균평점': avg_rt,
                '재구매율(%)': rep_rate,
            })

        if pos_rows:
            pos_df = pd.DataFrame(pos_rows)
            fig = px.scatter(
                pos_df,
                x='리뷰어비율(%)',
                y='평균평점',
                size='재구매율(%)',
                color='product',
                text='product',
                size_max=50,
                labels={
                    '리뷰어비율(%)': f'{selected_skin} 리뷰어 비율 (%)',
                    '평균평점': '평균 평점',
                },
            )
            # 사분면 배경 구분선
            x_mid = pos_df['리뷰어비율(%)'].mean()
            y_mid = pos_df['평균평점'].mean()
            fig.add_hline(y=y_mid, line_dash='dash', line_color='gray', opacity=0.5)
            fig.add_vline(x=x_mid, line_dash='dash', line_color='gray', opacity=0.5)
            fig.update_traces(textposition='top center')
            fig.update_layout(
                height=500,
                yaxis=dict(range=[
                    max(1, pos_df['평균평점'].min() - 0.3),
                    5.1
                ]),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"'{selected_skin}' 피부타입 리뷰어 데이터가 없습니다.")


# =================================================================
# 탭 2) 개별 제품 상세
# =================================================================
with tab_detail:
    st.header("🔍 개별 제품 상세 분석")
    picked = st.selectbox("분석할 제품을 선택하세요", product_list)

    one_df = df[df['product'] == picked].copy()
    one_trouble = trouble_df[trouble_df['product'] == picked].copy()
    total_n = len(one_df)

    # 공통 전처리 (하위 섹션에서 재사용)
    avg_rating   = one_df['rating'].mean()
    rep_n        = (one_df['repurchase'].astype(str).str.lower() == 'true').sum()
    rep_rate     = round(rep_n / total_n * 100, 1) if total_n else 0
    rep_content  = one_df[one_df['repurchase'].astype(str).str.lower() == 'true']['content']
    cls_one      = one_df['content'].apply(classify_irritation)
    irritation_n = (cls_one == 'neg').sum()
    mild_n       = (cls_one == 'pos').sum()
    no_mention_n = total_n - irritation_n - mild_n

    pros = get_kw_count(one_df['content'], PROS_KW_DICT)
    pros = add_special_counts(one_df['content'], pros, 'pros')   # 트러블케어 추가
    cons = get_kw_count(one_df['content'], CONS_KW_DICT)
    cons = add_special_counts(one_df['content'], cons, 'cons')   # 자극(부정)·트러블유발 추가
    pros_df = pd.DataFrame(list(pros.items()), columns=['항목', '건수'])
    pros_df = pros_df[pros_df['건수'] > 0].sort_values('건수', ascending=False).head(15)
    cons_df = pd.DataFrame(list(cons.items()), columns=['항목', '건수'])
    cons_df = cons_df[cons_df['건수'] > 0].sort_values('건수', ascending=False).head(15)

    # ═══════════════════════════════════════════
    # 구역 1 — 제품 개요
    # ═══════════════════════════════════════════
    st.markdown("---")
    st.subheader("📌 1. 제품 개요")

    ov1, ov2, ov3 = st.columns(3)
    with ov1:
        st.metric("🔬 분석 리뷰 수", f"{total_n:,}건")
    with ov2:
        st.metric("⭐ 평균 평점", f"{avg_rating:.2f} / 5.0")
    with ov3:
        st.metric("🔄 재구매 리뷰 비율", f"{rep_rate}%", f"{rep_n}건")

    # ═══════════════════════════════════════════
    # 구역 2 — 평점 심층 분석
    # ═══════════════════════════════════════════
    st.markdown("---")
    st.subheader("⭐ 2. 평점 심층 분석")

    p1, p2, p3 = st.columns(3)
    with p1:
        st.caption("별점 분포")
        rating_hist = one_df['rating'].value_counts().sort_index().reset_index()
        rating_hist.columns = ['평점', '건수']
        fig = px.bar(rating_hist, x='평점', y='건수', text_auto=True,
                     color='건수', color_continuous_scale='Blues')
        fig.update_layout(xaxis_type='category', showlegend=False,
                          coloraxis_showscale=False, height=280,
                          margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with p2:
        st.caption("피부타입별 평균 평점")
        skin_avg = one_df.groupby('skinType')['rating'].mean().round(2).reset_index()
        skin_avg.columns = ['피부타입', '평균평점']
        skin_avg = skin_avg.sort_values('평균평점', ascending=True)
        fig = px.bar(skin_avg, x='평균평점', y='피부타입', orientation='h',
                     text='평균평점', color='평균평점',
                     color_continuous_scale='RdYlGn', range_color=[3.5, 5.0])
        fig.update_layout(xaxis=dict(range=[3.0, 5.0]),
                          coloraxis_showscale=False, height=280,
                          margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with p3:
        st.caption("피부고민별 평균 평점")
        trouble_avg = one_trouble.groupby('skinTrouble')['rating'].mean().round(2).reset_index()
        trouble_avg.columns = ['피부고민', '평균평점']
        trouble_avg = trouble_avg.sort_values('평균평점', ascending=True)
        fig = px.bar(trouble_avg, x='평균평점', y='피부고민', orientation='h',
                     text='평균평점', color='평균평점',
                     color_continuous_scale='RdYlGn', range_color=[3.5, 5.0])
        fig.update_layout(xaxis=dict(range=[3.0, 5.0]),
                          coloraxis_showscale=False, height=280,
                          margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ═══════════════════════════════════════════
    # 구역 3 — 고객 분석
    # ═══════════════════════════════════════════
    st.markdown("---")
    st.subheader("👥 3. 고객 분석")

    g1, g2 = st.columns(2)
    with g1:
        st.caption("피부 타입 분포")
        vc = one_df['skinType'].value_counts().reset_index()
        vc.columns = ['skinType', 'count']
        fig = px.pie(vc, values='count', names='skinType', hole=0.4)
        fig.update_traces(textinfo='label+percent')
        fig.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        st.caption("피부 고민 분포 (Top 7)")
        vc2 = one_trouble['skinTrouble'].value_counts().head(7).reset_index()
        vc2.columns = ['skinTrouble', 'count']
        fig = px.pie(vc2, values='count', names='skinTrouble', hole=0.4)
        fig.update_traces(textinfo='label+percent')
        fig.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ═══════════════════════════════════════════
    # 구역 4 — 리뷰 키워드 분석
    # ═══════════════════════════════════════════
    st.markdown("---")
    st.subheader("🔑 4. 리뷰 키워드 분석")

    k1, k2 = st.columns(2)
    with k1:
        st.caption("👍 장점 키워드 순위")
        fig = px.bar(pros_df, x='항목', y='건수', text_auto=True,
                     color='건수', color_continuous_scale='Greens')
        fig.update_layout(xaxis={'categoryorder': 'total descending'},
                          coloraxis_showscale=False, height=320,
                          margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with k2:
        st.caption("👎 단점 키워드 순위")
        fig = px.bar(cons_df, x='항목', y='건수', text_auto=True,
                     color='건수', color_continuous_scale='Reds')
        fig.update_layout(xaxis={'categoryorder': 'total descending'},
                          coloraxis_showscale=False, height=320,
                          margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ═══════════════════════════════════════════
    # 구역 5 — 사용 경험 분석
    # ═══════════════════════════════════════════
    st.markdown("---")
    st.subheader("🧪 5. 사용 경험 분석")

    e1, e2 = st.columns(2)
    with e1:
        st.caption("자극 및 트러블 언급 비율")
        irr_data = pd.DataFrame({
            '구분': ['언급 없음', '무자극·진정', '자극 불만'],
            '건수': [no_mention_n, mild_n, irritation_n],
        }).sort_values('건수', ascending=False)
        fig = px.pie(irr_data, values='건수', names='구분', hole=0.4,
                     color='구분',
                     color_discrete_map={
                         '자극 불만': '#E8404A',
                         '무자극·진정': '#4CAF50',
                         '언급 없음': '#B0BEC5'
                     })
        fig.update_traces(textinfo='label+percent')
        fig.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with e2:
        st.caption("사용 상황 분포 (TPO)")
        tpo = get_kw_count(one_df['content'], TPO_KW_DICT)
        tpo_df = pd.DataFrame(list(tpo.items()), columns=['항목', '건수'])
        tpo_df = tpo_df[tpo_df['건수'] > 0].sort_values('건수', ascending=False)
        if len(tpo_df) > 0:
            fig = px.bar(tpo_df, x='항목', y='건수', text_auto=True,
                         color='건수', color_continuous_scale='Purples')
            fig.update_layout(xaxis={'categoryorder': 'total descending'},
                              coloraxis_showscale=False, height=320,
                              margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("TPO 관련 키워드 언급이 없습니다.")

    # ═══════════════════════════════════════════
    # 구역 6 — 구매/이탈 요인 분석
    # ═══════════════════════════════════════════
    st.markdown("---")
    st.subheader("🛒 6. 구매/이탈 요인 분석")

    f1, f2, f3 = st.columns(3)

    with f1:
        st.caption("구매 결정 요인")
        trigger = get_kw_count(one_df['content'], TRIGGER_KW_DICT)
        trigger_df = pd.DataFrame(list(trigger.items()), columns=['항목', '건수'])
        trigger_df = trigger_df[trigger_df['건수'] > 0].sort_values('건수', ascending=False)
        if len(trigger_df) > 0:
            fig = px.pie(trigger_df, values='건수', names='항목', hole=0.4)
            fig.update_traces(textinfo='label+percent')
            fig.update_layout(height=320, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("구매 결정 요인 키워드 언급이 없습니다.")

    with f2:
        st.caption("💖 재구매 이유 Top 5")
        combined_dict = {**LOYALTY_KW_DICT, **PROS_KW_DICT}
        if len(rep_content) > 0:
            loyalty = get_kw_count(rep_content, combined_dict)
            loyalty_df = pd.DataFrame(list(loyalty.items()), columns=['항목', '건수'])
            loyalty_df = loyalty_df[loyalty_df['건수'] > 0].sort_values('건수', ascending=False).head(5)
            fig = px.bar(loyalty_df, x='건수', y='항목', orientation='h',
                         text='건수', color='건수', color_continuous_scale='Greens')
            fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                              coloraxis_showscale=False, height=320,
                              margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("재구매 리뷰 데이터가 없습니다.")

    with f3:
        st.caption("💔 이탈 이유 Top 5 (1~2점 리뷰 기준)")
        bad_df = one_df[one_df['rating'] <= 2]
        if len(bad_df) > 0:
            bad_cons = get_kw_count(bad_df['content'], CONS_KW_DICT)
            bad_df2 = pd.DataFrame(list(bad_cons.items()), columns=['항목', '건수'])
            bad_df2 = bad_df2[bad_df2['건수'] > 0].sort_values('건수', ascending=False).head(5)
            if len(bad_df2) > 0:
                fig = px.bar(bad_df2, x='건수', y='항목', orientation='h',
                             text='건수', color='건수', color_continuous_scale='Reds')
                fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                                  coloraxis_showscale=False, height=320,
                                  margin=dict(t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("1~2점 리뷰에서 단점 키워드가 검출되지 않았습니다.")
        else:
            st.info(f"1~2점 리뷰가 없습니다. (총 {total_n}건 중 0건)")

    # ═══════════════════════════════════════════
    # 구역 7 — AI 인사이트 분석
    # ═══════════════════════════════════════════
    st.markdown("---")
    st.subheader("🤖 7. AI 인사이트 분석")

    if st.button(f"🚀 [{picked}] AI 인사이트 분석 시작", use_container_width=True,
                 key=f"single_ai_{picked}"):
        st.info("🔍 데이터를 분석하고 AI 리포트를 생성 중입니다. 잠시만 기다려주세요... (10~20초 소요)")
        with st.spinner("AI가 리뷰를 읽고 있습니다..."):
            rating_5_df  = one_df[one_df['rating'] == 5]
            trouble_5_df = one_trouble[one_trouble['rating'] == 5]

            single_summary = f"""
            [제품: {picked} (전체 {total_n}건)]
            - 평균 평점: {avg_rating:.2f} / 재구매 비율: {rep_rate}%({rep_n}건)
            - 자극도: 언급없음({no_mention_n}/{total_n}건), 무자극({mild_n}/{total_n}건), 자극({irritation_n}/{total_n}건)
            - 장점 키워드: {pros_df.to_dict('records') if len(pros_df) > 0 else '없음'}
            - 단점 키워드: {cons_df.to_dict('records') if len(cons_df) > 0 else '없음'}
            - 전체 피부타입 순위: {one_df['skinType'].value_counts().to_dict()}
            - 전체 피부고민 순위: {one_trouble['skinTrouble'].value_counts().to_dict()}
            - 평점 분포: {one_df['rating'].value_counts().to_dict()}
            - 5점 부여 고객 피부타입 순위: {rating_5_df['skinType'].value_counts().to_dict()}
            - 5점 부여 고객 피부고민 순위: {trouble_5_df['skinTrouble'].value_counts().to_dict()}
            """

            SINGLE_PROMPT = f"""
너는 10년 차 뷰티 데이터 전략가야. 전달받은 단일 제품 통계를 바탕으로
'리뷰 분석 리포트'를 한국어로 작성해줘.

[작성 규칙]
1. 제목 체계: # / ## / #### 만 사용, 별표(**) 금지.
2. 수치 근거: 모든 수치는 <span style="color:#777777">({{}}/{total_n}건)</span> 형식으로
   반드시 '해당건수/전체건수' 형태로 표기. 예: (144/{total_n}건)
3. 인사말 생략: 바로 # [리뷰 분석 리포트] 제목부터 시작.
4. 단순 요약 금지: 수치 뒤에 반드시 인사이트 1문장 추가.

[리포트 구조]
# [리뷰 분석 리포트]
## 1. 핵심 인사이트 요약
## 2. 타깃 페르소나 분석
## 3. 제품 경험(UX) 분석
## 4. 전략 제안 (2~3가지)
"""
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": SINGLE_PROMPT},
                        {"role": "user", "content":
                            f"{single_summary}\n\n[리뷰 샘플]\n"
                            f"{one_df['content'].dropna().head(15).tolist()}"}
                    ]
                )
                st.markdown(response.choices[0].message.content, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"오류 발생: {e}")


# =================================================================
# 탭 3) AI 비교 전략 리포트
# =================================================================
with tab_ai:
    st.header("🤖 AI 비교 전략 리포트")
    st.caption("리뷰 데이터를 기반으로 AI 경쟁 분석 리포트를 생성합니다.")

    if st.button("🚀 경쟁 제품 비교 전략 분석 시작", use_container_width=True):
        st.info("🔍 10개 제품 데이터를 집계하고 AI 전략 리포트를 생성 중입니다. 잠시만 기다려주세요... (30초~1분 소요)")
        with st.spinner("AI 분석 중입니다..."):

            # ── AI에게 넘길 데이터 최대한 풍부하게 구성 ──
            pros_matrix  = build_keyword_matrix(df, PROS_KW_DICT, normalize=True, special='pros')
            cons_matrix  = build_keyword_matrix(df, CONS_KW_DICT, normalize=True, special='cons')
            tpo_matrix   = build_keyword_matrix(df, TPO_KW_DICT, normalize=True)
            trig_matrix  = build_keyword_matrix(df, TRIGGER_KW_DICT, normalize=True)

            # 카테고리 평균 계산
            cat_avg_rating    = round(summary['평균평점'].mean(), 2)
            cat_avg_rep       = round(summary['재구매율(%)'].mean(), 1)
            cat_avg_irr       = round(summary['자극언급률(%)'].mean(), 1)
            cat_avg_price     = round(summary['가격만족도지수'].mean(), 1)
            cat_avg_pros      = pros_matrix.mean(axis=0).round(1).to_dict()
            cat_avg_cons      = cons_matrix.mean(axis=0).round(1).to_dict()

            # 피부타입별 5점 비율 (제품×피부타입)
            skin_5rate = {}
            for product, g in df.groupby('product'):
                for stype, sg in g.groupby('skinType'):
                    key = f"{product}_{stype}"
                    skin_5rate[key] = round((sg['rating'] == 5).mean() * 100, 1)

            # 피부타입별 1~2점 비율 (제품×피부타입) — 비추 근거용
            skin_low_rate = {}
            for product, g in df.groupby('product'):
                for stype, sg in g.groupby('skinType'):
                    if len(sg) < 3:   # 샘플 너무 적으면 제외
                        continue
                    key = f"{product}_{stype}"
                    skin_low_rate[key] = round((sg['rating'] <= 2).mean() * 100, 1)

            # 피부고민별 5점 비율 (제품×피부고민)
            trouble_5rate = {}
            for product, g in trouble_df.groupby('product'):
                for tr, tg in g.groupby('skinTrouble'):
                    key = f"{product}_{tr}"
                    trouble_5rate[key] = round((tg['rating'] == 5).mean() * 100, 1)

            # 종합 순위 Top 3 (평균평점 × 0.6 + 재구매율 정규화 × 0.4 복합점수 기준)
            # ① 각 지표를 0~100으로 정규화 (min-max)
            def _minmax(s):
                mn, mx = s.min(), s.max()
                return (s - mn) / (mx - mn) * 100 if mx != mn else pd.Series([50.0] * len(s), index=s.index)

            summary_rank = summary.copy()
            summary_rank['_rating_norm'] = _minmax(summary_rank['평균평점'])
            summary_rank['_rep_norm']    = _minmax(summary_rank['재구매율(%)'])
            summary_rank['_price_norm']  = _minmax(summary_rank['가격만족도지수'])
            summary_rank['종합점수']     = (summary_rank['_rating_norm'] * 0.5
                                            + summary_rank['_rep_norm']   * 0.3
                                            + summary_rank['_price_norm'] * 0.2).round(1)
            top3_products = summary_rank.sort_values('종합점수', ascending=False).head(3)[
                ['product', '평균평점', '재구매율(%)', '가격만족도지수', '종합점수']
            ].to_dict('records')

            data_summary = f"""
[제품 목록]
{[p for p in summary['product'].tolist()]}

[종합 순위 Top 3 (평균평점 50% + 재구매율 30% + 가격만족도 20% 복합점수 기준 / 종합점수는 0~100 정규화값)]
{top3_products}

[제품별 핵심 지표]
{summary[['product','리뷰수','평균평점','5점비율(%)','재구매율(%)','자극언급률(%)','가격만족도지수']].to_dict('records')}

[카테고리 전체 평균]
평점: {cat_avg_rating} / 재구매율: {cat_avg_rep}% / 가격만족도: {cat_avg_price}

[제품별 장점 키워드 언급률(%)]
{pros_matrix.round(1).to_dict('index')}

[장점 카테고리 평균(%)]
{cat_avg_pros}

[제품별 단점 키워드 언급률(%)]
{cons_matrix.round(1).to_dict('index')}

[단점 카테고리 평균(%)]
{cat_avg_cons}

[제품별 피부타입 분포(%)]
{df.groupby(['product','skinType']).size().groupby(level=0).transform(lambda x: (x/x.sum()*100).round(1)).reset_index(name='비율').to_dict('records')}

[제품×피부타입 5점 비율(%)]
{skin_5rate}

[제품×피부타입 1~2점 비율(%) — 비추 근거용, 값 클수록 해당 피부타입에 부적합]
{skin_low_rate}

[제품×피부고민 5점 비율(%)]
{trouble_5rate}

[제품별 TPO 언급률(%)]
{tpo_matrix.round(1).to_dict('index')}

[제품별 구매 트리거 언급률(%)]
{trig_matrix.round(1).to_dict('index')}
"""

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": f"[비교 통계 데이터]\n{data_summary}"}
                    ],
                    max_tokens=4000,
                )
                result_text = response.choices[0].message.content
                st.markdown(result_text, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"오류 발생: {e}")


# =================================================================
# 탭 4) 맞춤 제품 추천
# =================================================================
with tab_recommend:
    st.header("🎯 맞춤 제품 추천")
    st.caption("피부타입·고민·TPO를 선택하면 리뷰 데이터 기반 점수로 최적 제품을 추천합니다.")

    # ── Step 1: 피부타입 선택 (필수) ──
    st.subheader("Step 1  피부타입 선택 (필수)")
    SKIN_TYPE_LIST = ["건성", "약건성", "지성", "복합성", "민감성", "트러블성", "중성"]
    selected_type = st.radio(
        "나의 피부타입은?",
        SKIN_TYPE_LIST,
        horizontal=True,
        key="rec_skin_type"
    )

    # ── Step 2: 피부고민 선택 (선택사항) ──
    st.subheader("Step 2  피부고민 선택 (선택사항, 복수 선택 가능)")
    TROUBLE_LIST = ["잡티", "미백", "주름", "각질", "트러블", "블랙헤드",
                    "피지과다", "모공", "탄력", "홍조", "아토피", "다크서클", "건조함"]
    selected_troubles = st.multiselect(
        "해당하는 피부고민을 모두 선택하세요",
        TROUBLE_LIST,
        key="rec_troubles"
    )

    # ── Step 3: TPO 선택 (선택사항) ──
    st.subheader("Step 3  사용 상황(TPO) 선택 (선택사항, 복수 선택 가능)")
    TPO_LIST = ["아침", "밤", "화장 전", "환절기", "여름", "겨울"]
    TPO_KEYWORD_MAP = {
        "아침": "아침",
        "밤": "밤",
        "화장 전": "화장전",
        "환절기": "환절기",
        "여름": "여름",
        "겨울": "겨울",
    }
    selected_tpos = st.multiselect(
        "주로 어떤 상황에서 사용하시나요?",
        TPO_LIST,
        key="rec_tpos"
    )

    # ── Step 4: 기대 효능 선택 (선택사항) ──
    st.subheader("Step 4  기대 효능 선택 (선택사항, 복수 선택 가능)")
    EFFICACY_LIST = sorted(set(PROS_KW_DICT.values()))
    selected_efficacies = st.multiselect(
        "원하는 기대 효능을 모두 선택하세요",
        EFFICACY_LIST,
        key="rec_efficacies"
    )

    st.markdown("---")

    # ── 분석 버튼 ──
    if st.button("✨ 내 피부에 맞는 제품 추천받기", use_container_width=True, key="rec_btn"):

        # ════════════════════════════════
        # 점수 계산 로직
        # ════════════════════════════════
        score_rows = []

        for product, g in df.groupby('product'):
            n = len(g)
            score = 0.0
            detail = {}

            skin_sub = g[g['skinType'] == selected_type]
            n_skin   = len(skin_sub)
            if n_skin > 0:
                five_rate_skin = (skin_sub['rating'] == 5).mean() * 100
                base_score     = five_rate_skin * 0.5
                rep_rate_skin  = (skin_sub['repurchase'].astype(str).str.lower() == 'true').mean() * 100
                base_score    += rep_rate_skin * 0.3
            else:
                five_rate_skin = 0
                rep_rate_skin  = 0
                base_score     = 0

            avg_rt       = g['rating'].mean()
            norm_rating  = (avg_rt - 1) / 4 * 100
            base_score  += norm_rating * 0.2

            score += base_score
            detail['기본점수'] = round(base_score, 2)
            detail[f'{selected_type} 리뷰어 수'] = n_skin
            detail[f'{selected_type} 5점비율'] = f"{five_rate_skin:.1f}%"

            if selected_troubles:
                trouble_scores = []
                for tr in selected_troubles:
                    tr_sub = trouble_df[
                        (trouble_df['product'] == product) &
                        (trouble_df['skinTrouble'] == tr)
                    ]
                    if len(tr_sub) > 0:
                        ts = (tr_sub['rating'] == 5).mean() * 100
                        trouble_scores.append(ts)
                if trouble_scores:
                    trouble_bonus = sum(trouble_scores) / len(trouble_scores)
                    score        += trouble_bonus
                    detail['피부고민 보너스'] = round(trouble_bonus, 2)

            if selected_tpos:
                tpo_bonus = 0
                for tpo_label in selected_tpos:
                    tpo_kw = TPO_KEYWORD_MAP[tpo_label]
                    patterns = [re.escape(normalize_text(k))
                                for k, v in TPO_KW_DICT.items() if v == tpo_kw]
                    if patterns:
                        p = '|'.join(patterns)
                        hits = g['content'].astype(str).apply(normalize_text).str.contains(p, na=False).sum()
                        tpo_bonus += hits / n * 100 if n else 0
                score        += tpo_bonus
                detail['TPO 보너스'] = round(tpo_bonus, 2)

            if selected_efficacies:
                efficacy_bonus = 0
                for eff in selected_efficacies:
                    patterns_n = [re.escape(normalize_text(k))
                                  for k, v in PROS_KW_DICT.items() if v == eff]
                    patterns_c = [re.escape(compact_text(k))
                                  for k, v in PROS_KW_DICT.items() if v == eff]
                    if patterns_n:
                        p_n = '|'.join(patterns_n)
                        p_c = '|'.join(patterns_c)
                        norm_s = g['content'].astype(str).apply(normalize_text)
                        comp_s = g['content'].astype(str).apply(compact_text)
                        hits = (norm_s.str.contains(p_n, na=False) |
                                comp_s.str.contains(p_c, na=False)).sum()
                        efficacy_bonus += hits / n * 100 if n else 0
                score += efficacy_bonus
                detail['기대 효능 보너스'] = round(efficacy_bonus, 2)

            if n_skin > 0:
                irr_penalty = skin_sub['content'].apply(is_negative_irritation).mean() * 100
            else:
                irr_penalty = g['content'].apply(is_negative_irritation).mean() * 100
            score -= irr_penalty * 0.5
            detail['자극 페널티'] = round(irr_penalty * 0.5, 2)
            detail['최종점수']    = round(score, 2)

            score_rows.append({'product': product, '점수': round(score, 2), **detail})

        result_df = pd.DataFrame(score_rows).sort_values('점수', ascending=False).reset_index(drop=True)
        result_df['순위'] = result_df.index + 1

        # ── 결과를 session_state에 저장 (AI 버튼 눌러도 결과 유지되도록) ──
        st.session_state['rec_result_df']       = result_df
        st.session_state['rec_selected_type']   = selected_type
        st.session_state['rec_selected_troubles'] = selected_troubles
        st.session_state['rec_selected_tpos']   = selected_tpos
        st.session_state['rec_selected_efficacies'] = selected_efficacies

    # ── 결과 표시 (session_state에 저장된 경우 항상 렌더링) ──
    if 'rec_result_df' in st.session_state:
        result_df       = st.session_state['rec_result_df']
        s_type          = st.session_state['rec_selected_type']
        s_troubles      = st.session_state['rec_selected_troubles']
        s_tpos          = st.session_state['rec_selected_tpos']
        s_efficacies    = st.session_state['rec_selected_efficacies']

        st.subheader("📊 추천 순위 결과")

        cond_parts = [f"피부타입: {s_type}"]
        if s_troubles:
            cond_parts.append(f"피부고민: {', '.join(s_troubles)}")
        if s_tpos:
            cond_parts.append(f"TPO: {', '.join(s_tpos)}")
        if s_efficacies:
            cond_parts.append(f"기대 효능: {', '.join(s_efficacies)}")
        st.info("  |  ".join(cond_parts))

        # 1~3위 카드
        top3 = result_df.head(3)
        medals = ["🥇", "🥈", "🥉"]
        c1, c2, c3 = st.columns(3)
        for col, (_, row), medal in zip([c1, c2, c3], top3.iterrows(), medals):
            with col:
                st.markdown(
                    f"<div style='background:#f8f9fa;border-radius:12px;padding:16px;text-align:center;'>"
                    f"<div style='font-size:2em;'>{medal}</div>"
                    f"<div style='font-weight:bold;font-size:1.05em;margin:8px 0;'>{row['product']}</div>"
                    f"<div style='color:#E8404A;font-size:1.3em;font-weight:bold;'>{row['점수']:.1f}점</div>"
                    f"<div style='color:gray;font-size:0.85em;margin-top:6px;'>"
                    f"{row.get(f'{s_type} 5점비율', 'N/A')} 만족</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # 전체 순위표
        with st.expander("📋 전체 순위 상세 보기"):
            display_cols = ['순위', 'product', '점수',
                            f'{s_type} 리뷰어 수', f'{s_type} 5점비율',
                            '기본점수']
            if s_troubles:
                display_cols.append('피부고민 보너스')
            if s_tpos:
                display_cols.append('TPO 보너스')
            if s_efficacies:
                display_cols.append('기대 효능 보너스')
            display_cols.append('자극 페널티')
            display_cols = [c for c in display_cols if c in result_df.columns]
            st.dataframe(result_df[display_cols], use_container_width=True)

        # 비추 제품
        worst = result_df.tail(3)['product'].tolist()
        st.warning(f"⚠️ 비추 제품 (하위 3개): {' / '.join(worst)}")

        # ── AI 추천 이유 버튼 (결과 블록 밖에 독립 렌더링 → 리셋 없음) ──
        st.markdown("---")
        if st.button("🤖 AI 추천 이유 상세 설명 받기", use_container_width=True, key="rec_ai_btn"):
            st.info("✍️ AI가 추천 이유를 작성 중입니다. 잠시만 기다려주세요... (10~20초 소요)")
            with st.spinner("AI가 분석 중입니다..."):
                top5_info = result_df.head(5)[
                    ['product', '점수', f'{s_type} 5점비율', '기본점수']
                ].to_dict('records')

                rec_prompt = f"""
너는 뷰티 제품 추천 전문가야. 아래 조건과 점수 데이터를 바탕으로
각 추천 제품의 이유를 한국어로 친절하게 설명해줘.

[사용자 조건]
- 피부타입: {s_type}
- 피부고민: {s_troubles if s_troubles else '선택 없음'}
- TPO: {s_tpos if s_tpos else '선택 없음'}
- 기대 효능: {s_efficacies if s_efficacies else '선택 없음'}

[추천 순위 Top 5 (점수 기준)]
{top5_info}

[비추 제품 (하위 3개)]
{worst}

[작성 규칙]
1. # 제목 없이 바로 시작. 별표(**) 금지.
2. 각 추천 제품마다 #### 제목으로 구분.
3. 추천 이유는 위 조건(피부타입·고민·TPO)과 연결해서 2~3문장으로.
4. 마지막에 비추 제품 이유도 간단히 1문장씩.
5. 전체 500자 이내로 간결하게.
"""
                try:
                    rec_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": rec_prompt}],
                        max_tokens=800,
                    )
                    st.markdown(rec_response.choices[0].message.content, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"오류 발생: {e}")
