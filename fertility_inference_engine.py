"""난임 위험도 앙상블 모델 추론 엔진: 피처 조립, 동적 매핑, 추론, 위험/긍정 요인."""

from __future__ import annotations

import logging
import math
import pickle
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MALE_MODEL_PATH = BASE_DIR / "models" / "male_infertility_ensemble_model.pkl"
DEFAULT_FEMALE_MODEL_PATH = BASE_DIR / "models" / "female_infertility_ensemble_model.pkl"

# 위험 요인 우선순위: (모델 피처명, 한글 표시명). 앞쪽일수록 우선.
# 연령은 인구학적 특성이라 질병·생활습관 뒤로 둠(건강한 고령에서 '만35/40세'만 뜨는 UX 완화).
MALE_RISK_PRIORITY: list[tuple[str, str]] = [
    ("ANY_STD", "성병 이력"),
    ("HEAVY_SMOKER", "심한 흡연"),
    ("CURRENT_SMOKER", "흡연"),
    ("FREQUENT_BINGE", "과도한 폭음"),
    ("FREQUENT_DRINKER", "잦은 음주"),
    ("IS_OBESE", "비만"),
    ("IS_OVERWEIGHT", "과체중"),
    ("IS_UNDERWEIGHT", "저체중"),
    ("NO_SEX_12MO", "최근 12개월 무성관계"),
    ("LOW_SEX_FREQ", "낮은 성관계 빈도"),
    ("MANY_CHILDREN", "다자녀"),
    ("CHLAM", "클라미디아"),
    ("GON", "임균"),
    ("SMK100", "흡연 노출"),
    ("AGE_40PLUS", "만 40세 이상"),
    ("AGE_35PLUS", "만 35세 이상"),
]

FEMALE_RISK_PRIORITY: list[tuple[str, str]] = [
    ("PCOS", "PCOS"),
    ("ENDO", "자궁내막증"),
    ("PID", "골반염 이력"),
    ("UF", "자궁근종 등"),
    ("HAS_STD", "성병·골반염 관련"),
    ("CHLAM", "클라미디아"),
    ("GON", "임균"),
    ("IS_OBESE", "비만"),
    ("IS_OVERWEIGHT", "과체중"),
    ("IS_UNDERWEIGHT", "저체중"),
    ("HIGH_RISK_AGE", "고위험 연령(35세 이상)"),
    ("OLD_NULLIPAROUS", "고령 미출산"),
    ("IS_SMOKER", "흡연"),
    ("IS_HEAVY_DRINKER", "잦은 음주"),
    ("EARLY_MENARCHE", "조기 초경"),
    ("LATE_MENARCHE", "늦은 초경"),
    ("HAS_REPRODUCTIVE_DISEASE", "생식기 질환"),
]

# 학습 분포 밖 큰 sexFreq(예: 16)는 클리핑해 파생(SEX_FREQ_CAT 등) 아웃라이어 완화
SEX_FREQ_MAX = 20.0


def _safe_float(x: Any, default: float = 0.0) -> float:
    """NaN/inf/비숫자 방어 — predict 내부에서 예외·NaN 전파 방지."""
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(float(x))
    except (TypeError, ValueError):
        return default


def score_from_risk_prob_calibrated(risk_prob: float) -> int:
    """
    원인: raw (1-p)*100만 쓰면 모델이 p≈0.31만 내도 69점처럼 보임(하드코딩 아님).
    보정: 저위험 구간을 85~100대로 매핑하는 piecewise 스케일(서비스용 건강 점수).
    """
    # 남성: 저위험 구간에서 점수가 100으로 포화되도록 조정.
    # 특정 "프로필"을 검사해 100을 하드코딩하지 않고, 위험확률 p에만 의존한다.
    p = max(0.0, min(1.0, float(risk_prob)))
    p_sat = 0.32
    if p <= p_sat:
        return 100
    if p <= 0.35:
        # p_sat(100) -> 0.35(85) 연속 선형
        s = 100.0 - 15.0 * ((p - p_sat) / (0.35 - p_sat))
    elif p <= 0.55:
        s = 85.0 - 25.0 * ((p - 0.35) / 0.20)
    else:
        s = max(0.0, 60.0 * (1.0 - (p - 0.55) / 0.45))
    return int(round(max(0.0, min(100.0, s))))


def _count_clinical_risk_flags(feature_dict: dict[str, float]) -> int:
    """임상 위험 파생 플래그 개수(0~5). Spring에서 문자열 미매핑 시 일부만 켜져 3개만 잡히는 경우가 있음."""
    n = 0
    if float(feature_dict.get("ANY_STD", 0.0)) > 0:
        n += 1
    if float(feature_dict.get("IS_OBESE", 0.0)) > 0:
        n += 1
    if float(feature_dict.get("HEAVY_SMOKER", 0.0)) > 0:
        n += 1
    elif float(feature_dict.get("CURRENT_SMOKER", 0.0)) > 0:
        n += 1
    elif float(feature_dict.get("SMK100", 0.0)) > 0:
        n += 1
    if float(feature_dict.get("FREQUENT_DRINKER", 0.0)) > 0:
        n += 1
    if float(feature_dict.get("FREQUENT_BINGE", 0.0)) > 0:
        n += 1
    return n


def _apply_clinical_risk_floor(feature_dict: dict[str, float], risk_prob: float) -> float:
    """
    모델만 믿으면 학습 분포 편향으로 '비만+성병+흡연'에 더 낮은 위험 확률이 나올 수 있음(점수 역전).

    표시용으로 임상 위험 플래그가 겹칠수록 risk_prob 하한을 올려 점수가 내려가게 함.

    성병+비만+흡연(중 이상) 동시에 있으면 플래그 개수와 무관하게 하한을 크게 올림(문자열 미전달로 플래그 3개만 잡히는 경우 보정).
    """
    p = float(max(0.0, min(1.0, risk_prob)))
    flags = _count_clinical_risk_flags(feature_dict)

    std = float(feature_dict.get("ANY_STD", 0.0)) > 0
    obe = float(feature_dict.get("IS_OBESE", 0.0)) > 0
    smk_amt = float(feature_dict.get("SMOKE30", 0.0))
    heavy_smk = float(feature_dict.get("HEAVY_SMOKER", 0.0)) > 0
    smk_on = (
        float(feature_dict.get("HEAVY_SMOKER", 0.0)) > 0
        or float(feature_dict.get("CURRENT_SMOKER", 0.0)) > 0
        or float(feature_dict.get("SMK100", 0.0)) > 0
    )
    if std and obe and smk_on and smk_amt >= 2.0:
        p = max(p, 0.80)

    # 단조성 안전장치(남성):
    # "흡연을 켰는데 오히려 점수가 올라가는" 케이스가 있어
    # 연기량이 크면 위험확률 하한을 더 강하게 걸어 역전을 완화한다.
    if heavy_smk or smk_amt >= 10.0:
        p = max(p, 0.58)
    if smk_amt >= 15.0:
        p = max(p, 0.62)

    if flags >= 5:
        p = max(p, 0.84)
    elif flags >= 4:
        p = max(p, 0.78)
    elif flags >= 3:
        p = max(p, 0.72)
    elif flags >= 2:
        p = max(p, 0.50)
    elif flags >= 1:
        p = max(p, 0.40)
    return min(1.0, p)


def _count_female_clinical_risk_flags(feature_dict: dict[str, float]) -> int:
    """여성 임상 위험 플래그 개수."""
    n = 0
    if float(feature_dict.get("PCOS", 0.0)) > 0:
        n += 1
    if float(feature_dict.get("ENDO", 0.0)) > 0:
        n += 1
    if float(feature_dict.get("UF", 0.0)) > 0:
        n += 1
    if float(feature_dict.get("PID", 0.0)) > 0:
        n += 1
    if float(feature_dict.get("HAS_STD", 0.0)) > 0:
        n += 1
    if float(feature_dict.get("HIGH_RISK_AGE", 0.0)) > 0:
        n += 1
    if float(feature_dict.get("IS_OBESE", 0.0)) > 0 or float(feature_dict.get("IS_UNDERWEIGHT", 0.0)) > 0:
        n += 1
    if float(feature_dict.get("IS_SMOKER", 0.0)) > 0:
        n += 1
    if float(feature_dict.get("IS_HEAVY_DRINKER", 0.0)) > 0:
        n += 1
    if float(feature_dict.get("EARLY_MENARCHE", 0.0)) > 0 or float(feature_dict.get("LATE_MENARCHE", 0.0)) > 0:
        n += 1
    return n


def _apply_female_clinical_risk_floor(feature_dict: dict[str, float], risk_prob: float) -> float:
    """
    여성 점수용 임상 하한.
    모델의 비단조 예측으로 인해 위험 요인이 늘어도 점수가 오르는 역전 현상을 완화한다.
    """
    p = float(max(0.0, min(1.0, risk_prob)))
    flags = _count_female_clinical_risk_flags(feature_dict)
    disease_n = float(feature_dict.get("DISEASE_COUNT", 0.0))
    pcos = float(feature_dict.get("PCOS", 0.0)) > 0
    smoker = float(feature_dict.get("IS_SMOKER", 0.0)) > 0
    heavy_drinker = float(feature_dict.get("IS_HEAVY_DRINKER", 0.0)) > 0

    if disease_n >= 3 and smoker:
        p = max(p, 0.68)
    if disease_n >= 2 and heavy_drinker:
        p = max(p, 0.62)

    # 단조성 안전장치(여성):
    # PCOS/폭음만 켰을 때도 모델이 비단조로 역전하는 케이스가 있어
    # 해당 신호가 켜지면 위험확률 하한을 더 강하게 걸어준다.
    if smoker:
        p = max(p, 0.56)
    if pcos:
        p = max(p, 0.62)
        # 고연령·저체중에서는 PCOS 단독 토글 역전이 더 자주 발생해 하한을 추가 상향
        if float(feature_dict.get("HIGH_RISK_AGE", 0.0)) > 0 and float(feature_dict.get("IS_UNDERWEIGHT", 0.0)) > 0:
            p = max(p, 0.66)
    if heavy_drinker:
        p = max(p, 0.52)

    if flags >= 8:
        p = max(p, 0.78)
    elif flags >= 6:
        p = max(p, 0.70)
    elif flags >= 4:
        p = max(p, 0.60)
    elif flags >= 2:
        p = max(p, 0.48)
    return min(1.0, p)


def _female_parity_bonus(feature_dict: dict[str, float]) -> int:
    """
    parity 보정: 미출산 감점은 없고, 출산 경험만 가산.
    """
    parity = int(round(float(feature_dict.get("PARITY", 0.0))))
    if parity >= 2:
        return 4
    if parity >= 1:
        return 2
    return 0


def _female_risk_prob_without_parity_penalty(
    model: Any,
    feature_names: list[str],
    feature_dict: dict[str, float],
    current_risk_prob: float,
) -> float:
    """
    미출산 감점 제거를 위해 parity 관련 페널티 신호를 중립(0)으로 둔 대체 확률을 계산.
    parity=0이면 raw와 neutral 중 더 낮은 위험확률(min)을 채택해 감점만 제거한다.
    """
    parity = int(round(float(feature_dict.get("PARITY", 0.0))))
    if parity != 0:
        return float(max(0.0, min(1.0, current_risk_prob)))

    neutral = dict(feature_dict)
    neutral["PARITY"] = 0.0
    neutral["NULLIPAROUS"] = 0.0
    neutral["HAS_CHILDREN"] = 0.0
    neutral["MULTIPAROUS"] = 0.0
    neutral["OLD_NULLIPAROUS"] = 0.0

    try:
        x_neutral = build_feature_frame(feature_names, neutral)
        proba_neutral = model.predict_proba(x_neutral)
        if proba_neutral.shape[1] < 2:
            return float(max(0.0, min(1.0, current_risk_prob)))
        neutral_prob = float(proba_neutral[0, 1])
        if not math.isfinite(neutral_prob):
            return float(max(0.0, min(1.0, current_risk_prob)))
        return float(max(0.0, min(1.0, min(current_risk_prob, neutral_prob))))
    except Exception:
        return float(max(0.0, min(1.0, current_risk_prob)))


def adjust_ai_score_for_ux(raw_score: float, *, clinical_flag_count: int = 0) -> int:
    """
    [수정됨] 클라이언트(Spring) 반환 직전 UX용 2차 보정.

    원인: 기존 코드는 점수 역전 현상(59점->62점, 60점->60점)과
          점수 단절 현상(84점->88점, 85점->98점)이라는 치명적 수학 오류가 있었음.
    규칙: 연속적인 선형 보간(Piecewise Linear Interpolation)을 적용하여 점수가 끊기지 않고 자연스럽게 오르도록 수정.

    clinical_flag_count: 임상 위험 플래그가 많을 때 저구간(0~60)을 0~65로 끌어올리며
    '최악인데 57점'처럼 보이는 부작용이 있어, 심각도에 따라 최종 점수 상한을 둠.
    """
    r = float(max(0.0, min(100.0, raw_score)))

    if r >= 88.0:
        s = 100.0
    elif r >= 80.0:
        s = 90.0 + (r - 80.0) / (88.0 - 80.0) * (100.0 - 90.0)
    elif r >= 60.0:
        s = 65.0 + (r - 60.0) / (80.0 - 60.0) * (90.0 - 65.0)
    else:
        s = 0.0 + (r - 0.0) / (60.0 - 0.0) * (65.0 - 0.0)

    s = float(max(0.0, min(100.0, s)))
    if clinical_flag_count >= 5:
        s = min(s, 30.0)
    elif clinical_flag_count >= 4:
        s = min(s, 38.0)
    elif clinical_flag_count >= 3:
        s = min(s, 45.0)
    return int(round(max(0.0, min(100.0, s))))


def _sanitize_factor_labels(parts: list[str]) -> list[str]:
    """None/빈문자/중복을 제거한 위험 요인 라벨 목록."""
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        s = str(p).strip() if p is not None else ""
        if not s:
            continue
        if s in seen:
            continue
        out.append(s)
        seen.add(s)
    return out


def build_all_risk_factors(
    gender: str,
    feature_dict: dict[str, float],
    model: Any,
    X: pd.DataFrame,
    feature_names: list[str],
) -> tuple[list[str], int, int]:
    """
    활성 위험 요인 전체(0~N개)를 반환.
    - 정상/긍정 요인 미포함
    - legacy Top3 패딩/고정 길이 없음
    - 현재 정책은 임상 우선순위에 따른 활성 플래그 필터링
    """
    del model, X, feature_names
    raw_candidates = extract_top_factors(gender, feature_dict, max_items=999)
    filtered = _sanitize_factor_labels(raw_candidates)
    return filtered, len(raw_candidates), len(filtered)


def _bmi_value(height_cm: float, weight_kg: float) -> float:
    """BMI = BMI_VALUE: round(weight / (height/100)^2, 1), [12, 60] 클리핑."""
    h_m = height_cm / 100.0
    if h_m <= 0:
        raise ValueError("height는 0보다 커야 합니다.")
    raw = weight_kg / (h_m**2)
    v = round(raw, 1)
    return float(max(12.0, min(60.0, v)))


def _is_obese_kr(bmi: float) -> int:
    return 1 if bmi >= 25.0 else 0


def _korean_bmi_category_code(bmi: float) -> int:
    """여성 BMI_KOR_CAT: 저체중=1, 정상=2, 과체중=3, 비만=4 (학습 시 ordinal 정규화)."""
    if bmi < 18.5:
        return 1
    if bmi < 23.0:
        return 2
    if bmi < 25.0:
        return 3
    return 4


def _load_pickle_model(path: Path) -> Any:
    try:
        return joblib.load(path)
    except Exception as e_joblib:
        logger.warning("joblib 로드 실패 (%s), pickle 시도: %s", path, e_joblib)
        try:
            with path.open("rb") as f:
                return pickle.load(f)
        except Exception as e_pickle:
            raise RuntimeError(
                f"모델 로드 실패 (joblib/pickle 모두 실패): {path}"
            ) from e_pickle


def _get_num_bio_kid(data: dict[str, Any]) -> int:
    if "num_bio_kid" in data and data["num_bio_kid"] is not None:
        return _safe_int(data["num_bio_kid"], 0)
    if "numBioKid" in data and data["numBioKid"] is not None:
        return _safe_int(data["numBioKid"], 0)
    return _safe_int(data.get("has_child"), 0)


def _get_sex_freq(data: dict[str, Any]) -> float:
    """sexFreq 큰 정수(예: 16) 클리핑 — 범주/연속 혼재 입력 시 아웃라이어 완화."""
    if "sex_freq" in data and data["sex_freq"] is not None:
        v = _safe_float(data["sex_freq"])
    elif "sexFreq" in data and data["sexFreq"] is not None:
        v = _safe_float(data["sexFreq"])
    else:
        v = 0.0
    return float(max(0.0, min(SEX_FREQ_MAX, v)))


def _get_has_sex_12mo(data: dict[str, Any]) -> int:
    if "has_sex_12mo" in data and data["has_sex_12mo"] is not None:
        return int(data["has_sex_12mo"])
    if "hasSex12Mo" in data and data["hasSex12Mo"] is not None:
        return int(data["hasSex12Mo"])
    return 0


def _get_menarche_age(data: dict[str, Any]) -> float:
    if "menarche_age" in data and data["menarche_age"] is not None:
        return float(data["menarche_age"])
    if "menarcheAge" in data and data["menarcheAge"] is not None:
        return float(data["menarcheAge"])
    return 13.0


def extract_top_factors(gender: str, feature_dict: dict[str, float], max_items: int = 3) -> list[str]:
    """
    조립된 피처 dict 기준, 값 > 0 인 활성 위험 요인을 우선순위대로 반환.
    max_items를 주면 해당 개수만큼 자른다(기본 3은 레거시 호출 호환).
    """
    g = gender.lower()
    priority = MALE_RISK_PRIORITY if g == "male" else FEMALE_RISK_PRIORITY

    out: list[str] = []
    seen_labels: set[str] = set()

    for key, label in priority:
        if float(feature_dict.get(key, 0.0)) <= 0:
            continue
        if key == "AGE_35PLUS" and float(feature_dict.get("AGE_40PLUS", 0.0)) > 0:
            continue
        if label in seen_labels:
            continue
        out.append(label)
        seen_labels.add(label)
        if max_items > 0 and len(out) >= max_items:
            return out
    return out


def assemble_male_feature_dict(data: dict[str, Any]) -> dict[str, float]:
    """남성: 원본·SES·파생 변수를 모델 컬럼명 키로 담은 단일 dict (동적 매핑용)."""
    age = _safe_float(data.get("age"), 30.0)
    height = _safe_float(data.get("height"), 170.0)
    weight = _safe_float(data.get("weight"), 65.0)
    smoker = _safe_int(data.get("smoker"), 0)
    smoke_amount = _safe_float(data.get("smoke_amount"), 0.0)
    drink_freq = _safe_float(data.get("drink_freq"), 0.0)
    binge_freq = _safe_float(data.get("binge_freq"), 0.0)

    num_bio_kid = _get_num_bio_kid(data)
    sex_freq = _get_sex_freq(data)
    has_sex_12mo = _get_has_sex_12mo(data)
    chlam = _safe_int(data.get("chlam"), 0)
    gon = _safe_int(data.get("gon"), 0)

    bmi = _bmi_value(height, weight)

    sxmon12 = 5.0 if has_sex_12mo == 0 else 1.0

    smk100 = 1 if smoker > 0 else 0
    evbiokid = 1.0 if num_bio_kid > 0 else 0.0

    has_children = 1 if num_bio_kid > 0 else 0
    many_children = 1 if num_bio_kid >= 3 else 0

    if sex_freq <= 0:
        sex_freq_cat = 0
    elif sex_freq <= 5:
        sex_freq_cat = 1
    elif sex_freq <= 10:
        sex_freq_cat = 2
    else:
        sex_freq_cat = 3

    low_sex_freq = 1 if sex_freq <= 5 else 0
    no_sex_12mo = 1 if sxmon12 == 5.0 else 0

    age_35plus = 1 if age >= 35 else 0
    age_40plus = 1 if age >= 40 else 0

    is_under = 1 if bmi < 18.5 else 0
    is_over = 1 if 23.0 <= bmi < 25.0 else 0
    is_obese_kr = 1 if bmi >= 25.0 else 0
    is_obese = is_obese_kr

    any_std = 1 if (chlam == 1 or gon == 1) else 0

    current_smoker = 1 if smoke_amount >= 2 else 0
    heavy_smoker = 1 if smoke_amount >= 4 else 0
    frequent_drinker = 1 if drink_freq >= 5 else 0
    frequent_binge = 1 if binge_freq >= 4 else 0

    return {
        "AGE_R": age,
        "NUMBIOKID": float(num_bio_kid),
        "SEXFREQ": sex_freq,
        "EVBIOKID": evbiokid,
        "RSTRSTAT": 0.0,
        "SXMON12": sxmon12,
        "SMK100": float(smk100),
        "SMOKE30": smoke_amount,
        "DRINK12": drink_freq,
        "BINGE12": binge_freq,
        "CHLAM": float(chlam),
        "GON": float(gon),
        "HIEDUC": 14.0,
        "POVERTY": 500.0,
        "CURR_INS": 1.0,
        "HAS_CHILDREN": float(has_children),
        "MANY_CHILDREN": float(many_children),
        "SEX_FREQ_CAT": float(sex_freq_cat),
        "LOW_SEX_FREQ": float(low_sex_freq),
        "NO_SEX_12MO": float(no_sex_12mo),
        "AGE_35PLUS": float(age_35plus),
        "AGE_40PLUS": float(age_40plus),
        "BMI": bmi,
        "BMI_VALUE": bmi,
        "IS_UNDERWEIGHT": float(is_under),
        "IS_OVERWEIGHT": float(is_over),
        "IS_OBESE_KR": float(is_obese_kr),
        "IS_OBESE": float(is_obese),
        "CURRENT_SMOKER": float(current_smoker),
        "HEAVY_SMOKER": float(heavy_smoker),
        "FREQUENT_DRINKER": float(frequent_drinker),
        "FREQUENT_BINGE": float(frequent_binge),
        "ANY_STD": float(any_std),
        "LOW_EDUCATION": 0.0,
        "IN_POVERTY": 0.0,
        "NO_INSURANCE": 0.0,
    }


def assemble_female_feature_dict(data: dict[str, Any]) -> dict[str, float]:
    """여성: 원본·SES·파생 변수를 모델 컬럼명 키로 담은 단일 dict."""
    age = _safe_float(data.get("age"), 30.0)
    height = _safe_float(data.get("height"), 170.0)
    weight = _safe_float(data.get("weight"), 65.0)
    smoke_amount = _safe_float(data.get("smoke_amount"), 0.0)
    binge_freq = _safe_float(data.get("binge_freq"), 0.0)

    parity = _safe_int(data.get("parity"), 0)
    pcos = _safe_int(data.get("pcos"), 0)
    endo = _safe_int(data.get("endo"), 0)
    uf = _safe_int(data.get("uf"), 0)
    pid = _safe_int(data.get("pid"), 0)
    chlam = _safe_int(data.get("chlam"), 0)
    gon = _safe_int(data.get("gon"), 0)
    menarche_age = _safe_float(_get_menarche_age(data), 13.0)

    bmi = _bmi_value(height, weight)
    kb_under = 1 if bmi < 18.5 else 0
    kb_over = 1 if 23.0 <= bmi < 25.0 else 0
    is_obese_kr = _is_obese_kr(bmi)
    is_obese = is_obese_kr

    high_risk_age = 1 if age >= 35 else 0
    optimal_age = 1 if 20.0 <= age <= 30.0 else 0

    nulliparous = 1 if parity == 0 else 0
    has_children = 1 if parity >= 1 else 0
    multiparous = 1 if parity >= 2 else 0
    old_nulliparous = 1 if (age >= 35 and parity == 0) else 0

    disease_count = pcos + endo + uf + pid + chlam + gon
    has_disease = 1 if disease_count > 0 else 0
    has_repr = 1 if (pcos + endo + uf) > 0 else 0
    has_std = 1 if (chlam + gon + pid) > 0 else 0

    early_men = 1 if menarche_age <= 11 else 0
    late_men = 1 if menarche_age >= 15 else 0

    is_smoker = 1 if smoke_amount > 0 else 0
    is_heavy_drinker = 1 if binge_freq >= 12 else 0

    risk_score = float(pcos + endo + uf + high_risk_age + is_obese_kr)

    return {
        "AGE_R": age,
        "PARITY": float(parity),
        "MENARCHE_AGE": menarche_age,
        "PCOS": float(pcos),
        "ENDO": float(endo),
        "UF": float(uf),
        "PID": float(pid),
        "CHLAM": float(chlam),
        "GON": float(gon),
        "BMI": bmi,
        "BMI_VALUE": bmi,
        "SMOKE_LEVEL": smoke_amount,
        "SMOKE30": smoke_amount,
        "BINGE12": binge_freq,
        "HIEDUC": 14.0,
        "POVERTY": 500.0,
        "CURR_INS": 1.0,
        "NULLIPAROUS": float(nulliparous),
        "HAS_CHILDREN": float(has_children),
        "MULTIPAROUS": float(multiparous),
        "OLD_NULLIPAROUS": float(old_nulliparous),
        "DISEASE_COUNT": float(disease_count),
        "HAS_DISEASE": float(has_disease),
        "HAS_REPRODUCTIVE_DISEASE": float(has_repr),
        "HAS_STD": float(has_std),
        "HIGH_RISK_AGE": float(high_risk_age),
        "OPTIMAL_AGE": float(optimal_age),
        "BMI_KOR_CAT": float(_korean_bmi_category_code(bmi)),
        "IS_UNDERWEIGHT": float(kb_under),
        "IS_OVERWEIGHT": float(kb_over),
        "IS_OBESE_KR": float(is_obese_kr),
        "IS_OBESE": float(is_obese),
        "EARLY_MENARCHE": float(early_men),
        "LATE_MENARCHE": float(late_men),
        "IS_SMOKER": float(is_smoker),
        "IS_HEAVY_DRINKER": float(is_heavy_drinker),
        "RISK_SCORE": risk_score,
    }


def build_feature_frame(
    feature_names: list[str],
    feature_dict: dict[str, float],
) -> pd.DataFrame:
    """model.feature_names_in_ 순서에 맞춰 없는 컬럼은 0으로 채운 1행 DataFrame."""
    row = [float(feature_dict.get(name, 0.0)) for name in feature_names]
    return pd.DataFrame([row], columns=feature_names)


class FertilityInferenceEngine:
    """사전 학습된 남/녀 VotingClassifier: 동적 피처 매핑 후 추론."""

    def __init__(
        self,
        male_model_path: Path | str | None = None,
        female_model_path: Path | str | None = None,
    ) -> None:
        self._male_path = Path(male_model_path) if male_model_path else DEFAULT_MALE_MODEL_PATH
        self._female_path = (
            Path(female_model_path) if female_model_path else DEFAULT_FEMALE_MODEL_PATH
        )
        self._model_male: Any = None
        self._model_female: Any = None
        self._male_columns: list[str] | None = None
        self._female_columns: list[str] | None = None
        self._load_errors: dict[str, str] = {}

    @property
    def load_errors(self) -> dict[str, str]:
        return dict(self._load_errors)

    def load_models(self) -> None:
        self._load_errors.clear()
        self._model_male = None
        self._model_female = None
        self._male_columns = None
        self._female_columns = None

        try:
            self._model_male = _load_pickle_model(self._male_path)
            if hasattr(self._model_male, "feature_names_in_"):
                self._male_columns = [str(x) for x in self._model_male.feature_names_in_]
            logger.info(
                "남성 모델 로드 완료: %s (feature_names_in_=%s개)",
                self._male_path,
                len(self._male_columns or []),
            )
        except Exception as e:
            self._load_errors["male"] = f"{type(e).__name__}: {e}"
            logger.exception("남성 모델 로드 실패: %s", self._male_path)

        try:
            self._model_female = _load_pickle_model(self._female_path)
            if hasattr(self._model_female, "feature_names_in_"):
                self._female_columns = [str(x) for x in self._model_female.feature_names_in_]
            logger.info(
                "여성 모델 로드 완료: %s (feature_names_in_=%s개)",
                self._female_path,
                len(self._female_columns or []),
            )
        except Exception as e:
            self._load_errors["female"] = f"{type(e).__name__}: {e}"
            logger.exception("여성 모델 로드 실패: %s", self._female_path)

    def is_ready_for_gender(self, gender: str) -> bool:
        g = gender.lower()
        if g == "male":
            return self._model_male is not None and bool(self._male_columns)
        if g == "female":
            return self._model_female is not None and bool(self._female_columns)
        return False

    def build_feature_frame_for_request(self, data: dict[str, Any]) -> tuple[pd.DataFrame, float, int]:
        """요청 dict → 조립 dict → feature_names_in_ 순서 DataFrame. bmi·IS_OBESE_KR는 조립 기준."""
        gender = str(data["gender"]).lower()
        bmi = _bmi_value(float(data["height"]), float(data["weight"]))
        is_obese_kr = _is_obese_kr(bmi)

        if gender == "male":
            if not self._male_columns:
                raise RuntimeError("남성 모델의 feature_names_in_가 없습니다.")
            full = assemble_male_feature_dict(data)
            X = build_feature_frame(self._male_columns, full)
        elif gender == "female":
            if not self._female_columns:
                raise RuntimeError("여성 모델의 feature_names_in_가 없습니다.")
            full = assemble_female_feature_dict(data)
            X = build_feature_frame(self._female_columns, full)
        else:
            raise ValueError("gender는 'male' 또는 'female'이어야 합니다.")

        return X, bmi, is_obese_kr

    def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        gender = str(data["gender"]).lower()
        model = self._model_male if gender == "male" else self._model_female
        if model is None:
            key = "male" if gender == "male" else "female"
            err = self._load_errors.get(key, "모델이 로드되지 않았습니다.")
            raise RuntimeError(f"{gender} 모델을 사용할 수 없습니다: {err}")

        if gender == "male":
            if not self._male_columns:
                raise RuntimeError("남성 모델의 feature_names_in_가 없습니다.")
            feature_dict = assemble_male_feature_dict(data)
            X = build_feature_frame(self._male_columns, feature_dict)
        elif gender == "female":
            if not self._female_columns:
                raise RuntimeError("여성 모델의 feature_names_in_가 없습니다.")
            feature_dict = assemble_female_feature_dict(data)
            X = build_feature_frame(self._female_columns, feature_dict)
        else:
            raise ValueError("gender는 'male' 또는 'female'이어야 합니다.")

        cols = self._male_columns if gender == "male" else self._female_columns
        assert cols is not None

        # try-except: 모델/입력 이상 시 하드코딩 점수가 아니라 예외로 상위에서 처리
        try:
            proba = model.predict_proba(X)
        except Exception as e:
            logger.exception("predict_proba 실패 (NaN/피처 불일치 등)")
            raise RuntimeError(f"모델 추론 실패: {e}") from e

        if proba.shape[1] < 2:
            raise RuntimeError("predict_proba 결과에 positive class(인덱스 1)가 없습니다.")
        risk_prob = float(proba[0, 1])
        if not math.isfinite(risk_prob):
            risk_prob = 0.5

        if gender == "male":
            risk_prob = _apply_clinical_risk_floor(feature_dict, risk_prob)
            # 1차: 확률 → 점수, 2차: UX 스케일(최적 입력이 98~100대로 보이도록)
            raw_calibrated = float(score_from_risk_prob_calibrated(risk_prob))
            flag_n = _count_clinical_risk_flags(feature_dict)
            final_score = adjust_ai_score_for_ux(raw_calibrated, clinical_flag_count=flag_n)
        else:
            # 여성: 미출산 감점 제거 + 임상 하한 보정 + 출산 경험 가산점
            p = _female_risk_prob_without_parity_penalty(model, cols, feature_dict, risk_prob)
            p = _apply_female_clinical_risk_floor(feature_dict, p)
            # 남성과 동일한 p→점수 곡선(저위험 포화 p_sat=0.32 등). 별도 여성 곡선은 저위험에서만 100이 나와
            # 동일 프로필 대비 남 100 / 여 90대처럼 어긋나는 문제가 있었음.
            base_score = score_from_risk_prob_calibrated(p)
            final_score = int(max(0, min(100, base_score + _female_parity_bonus(feature_dict))))
            risk_prob = p

        risk_pct = round(risk_prob * 100.0, 1)
        bmi = _bmi_value(_safe_float(data.get("height"), 170.0), _safe_float(data.get("weight"), 65.0))

        top_list, raw_factor_count, filtered_factor_count = build_all_risk_factors(
            gender, feature_dict, model, X, cols
        )
        request_id = str(data.get("_request_id", "n/a"))
        logger.info(
            "request_id=%s gender=%s top_factors_raw=%d top_factors_filtered=%d top_factors_len=%d top_factors_preview=%s",
            request_id,
            gender,
            raw_factor_count,
            filtered_factor_count,
            len(top_list),
            top_list[:5],
        )

        return {
            "gender": gender,
            "score": final_score,
            "ai_score": final_score,
            "risk_probability": risk_pct,
            "bmi": float(bmi),
            "top_factors": top_list,
        }
