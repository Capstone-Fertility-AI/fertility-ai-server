"""FastAPI 진입점: Spring 클라이언트와 느슨한 JSON 호환을 위한 PredictionRequest 및 라우팅."""

from __future__ import annotations

import logging
import traceback
from contextlib import asynccontextmanager
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from fertility_inference_engine import FertilityInferenceEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

engine = FertilityInferenceEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.load_models()
    if engine.load_errors:
        logger.warning("일부 모델 로드 실패: %s", engine.load_errors)
    yield


app = FastAPI(
    title="Fertility Risk AI Server",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Spring/클라이언트 키 → 서버 내부 필드명 (before 단계에서 통일) ---
_CLIENT_KEY_ALIASES: dict[str, str] = {
    "SMOKE30": "smoke_amount",
    "DRINK12": "drink_freq",
    "BINGE12": "binge_freq",
    "NUMBIOKID": "num_bio_kid",
    "SEXFREQ": "sex_freq",
    "MENARCHE_AGE": "menarche_age",
}


def _drop_null_fields(data: dict[str, Any]) -> dict[str, Any]:
    """JSON null은 Pydantic float/int와 호환되지 않으므로 키를 제거해 Field 기본값이 쓰이게 함."""
    return {k: v for k, v in data.items() if v is not None}


def _normalize_client_keys(data: dict[str, Any]) -> dict[str, Any]:
    """클라이언트 전용 대문자 키 등을 내부 snake_case로 복사(원본 키는 유지해 alias와 병행 가능)."""
    out = dict(data)
    for src, dst in _CLIENT_KEY_ALIASES.items():
        if src in out and out[src] is not None and dst not in out:
            out[dst] = out[src]
    return out


def _parse_smoke_status_ko(text: str) -> tuple[int, float]:
    """한글 흡연 상태 문자열 → (smoker, smoke_amount)."""
    s = text.strip().lower()
    if any(x in s for x in ("안 피", "안피", "비흡연", "금연", "전혀 안", "안 함")):
        return 0, 0.0
    if "매일" in s or "하루" in s:
        return 1, 18.0
    if "주" in s or "가끔" in s or "월" in s:
        return 1, 6.0
    return 1, 4.0


def _parse_weekly_habit_ko(text: str) -> float:
    """한글 음주/폭음 빈도 문자열 → DRINK12/BINGE12 근사치."""
    s = text.strip().lower()
    if any(x in s for x in ("안 함", "전혀", "비음주", "금주", "마시지")):
        return 0.0
    if "거의 매일" in s or "매일" in s:
        return 10.0
    if "주 1회" in s or "주1" in s or "이상" in s:
        return 5.0
    if "주 2" in s or "주2" in s:
        return 7.0
    if "월" in s or "가끔" in s:
        return 3.0
    return 4.0


def _map_lifestyle_strings_to_numerics(d: dict[str, Any]) -> dict[str, Any]:
    """
    Spring이 smokeStatus·drinkStatus 등 문자열만 보내고 숫자 필드가 비면
    모델은 비흡연·비음주로 처리되어 '고위험 프로필이 더 높은 점수' 같은 역전이 발생할 수 있음.

    문자열이 있으면 그 값으로 수치 필드를 채움(0으로 잘못 온 경우까지 덮어씀).
    이미 SMOKE30 등으로 양의 수치만 온 경우는 유지.
    """
    out = dict(d)

    if isinstance(out.get("hasSex12Mo"), bool):
        out["has_sex_12mo"] = 1 if out["hasSex12Mo"] else 0
    if isinstance(out.get("has_sex_12mo"), bool):
        out["has_sex_12mo"] = 1 if out["has_sex_12mo"] else 0

    raw_smoke = out.get("smokeStatus") or out.get("smoke_status")
    if isinstance(raw_smoke, str) and raw_smoke.strip():
        sm, amt = _parse_smoke_status_ko(raw_smoke)
        out["smoker"] = sm
        out["smoke_amount"] = amt

    raw_drink = out.get("drinkStatus") or out.get("drink_status")
    if isinstance(raw_drink, str) and raw_drink.strip():
        out["drink_freq"] = _parse_weekly_habit_ko(raw_drink)

    raw_binge = out.get("bingeStatus") or out.get("binge_status")
    if isinstance(raw_binge, str) and raw_binge.strip():
        out["binge_freq"] = _parse_weekly_habit_ko(raw_binge)

    return out


class PredictionRequest(BaseModel):
    """
    Spring 등에서 오는 느슨한 JSON을 수용하는 요청 모델.

    - model_validator(before): 키 별칭 병합, null→기본값, 누락 필드 보정.
    - model_validator(after): SMOKE30 등으로 채워진 smoke_amount 기준으로 smoker 보강.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    age: float = Field(default=30.0, description="나이 (null/누락 시 30)")
    height: float = Field(default=170.0, description="키(cm)")
    weight: float = Field(default=65.0, description="체중(kg)")

    smoker: int = Field(default=0, ge=0, description="흡연 (after 단계에서 smoke_amount와 정합)")
    smoke_amount: float = Field(
        default=0.0,
        validation_alias=AliasChoices("smoke_amount", "smokeAmount", "SMOKE30"),
        description="흡연량",
    )
    drink_freq: float = Field(
        default=0.0,
        validation_alias=AliasChoices("drink_freq", "drinkFreq", "DRINK12"),
        description="음주 빈도",
    )
    binge_freq: float = Field(
        default=0.0,
        validation_alias=AliasChoices("binge_freq", "bingeFreq", "BINGE12"),
        description="폭음 빈도",
    )

    num_bio_kid: int = Field(
        default=0,
        validation_alias=AliasChoices("num_bio_kid", "numBioKid", "NUMBIOKID"),
    )
    sex_freq: float = Field(
        default=0.0,
        validation_alias=AliasChoices("sex_freq", "sexFreq", "SEXFREQ"),
    )
    has_sex_12mo: int = Field(
        default=0,
        validation_alias=AliasChoices("has_sex_12mo", "hasSex12Mo"),
    )

    chlam: int = Field(default=0)
    gon: int = Field(default=0)

    has_child: int = Field(
        default=0,
        validation_alias=AliasChoices("has_child", "hasChild"),
    )
    std_history: int = Field(
        default=0,
        validation_alias=AliasChoices("std_history", "stdHistory"),
    )

    parity: int = Field(default=0)
    pcos: int = Field(default=0)
    endo: int = Field(default=0)
    uf: int = Field(default=0)
    pid: int = Field(default=0)
    mens_irregular: int = Field(
        default=0,
        validation_alias=AliasChoices("mens_irregular", "mensIrregular"),
    )
    misc_num: int = Field(
        default=0,
        validation_alias=AliasChoices("misc_num", "miscNum"),
    )
    menarche_age: float = Field(
        default=13.0,
        validation_alias=AliasChoices("menarche_age", "menarcheAge", "MENARCHE_AGE"),
    )

    @model_validator(mode="before")
    @classmethod
    def _loosen_input(cls, data: Any) -> Any:
        # 리스트 등 비정상 타입은 그대로 두어 이후 단계에서 에러 처리
        if not isinstance(data, dict):
            return data
        d = _normalize_client_keys(dict(data))
        d = _map_lifestyle_strings_to_numerics(d)
        d = _drop_null_fields(d)
        return d

    @model_validator(mode="after")
    def _infer_smoker_from_smoke_amount(self) -> PredictionRequest:
        """SMOKE30만 오고 smoker가 0이면, 흡연량이 양수일 때 smoker=1로 보정."""
        sm = float(self.smoke_amount)
        if sm > 0 and int(self.smoker) == 0:
            return self.model_copy(update={"smoker": 1})
        return self


class PredictionRequestWithGender(PredictionRequest):
    """레거시 POST /predict: gender 생략·null 시 male로 수용."""

    gender: Literal["male", "female"] | None = Field(
        default=None,
        validation_alias=AliasChoices("gender", "GENDER"),
    )

    @model_validator(mode="before")
    @classmethod
    def _gender_fallback(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = dict(data)
        if d.get("gender") is None or d.get("gender") == "":
            d["gender"] = "male"
        elif isinstance(d.get("gender"), str):
            g = d["gender"].strip().lower()
            if g in ("m", "man", "male"):
                d["gender"] = "male"
            elif g in ("f", "woman", "female"):
                d["gender"] = "female"
            else:
                d["gender"] = "male"
        return d


# 하위 호환 별칭
UserHealthBodyBase = PredictionRequest
UserHealthData = PredictionRequestWithGender


class PredictResult(BaseModel):
    """Spring 연동용 응답 모델."""

    model_config = ConfigDict(populate_by_name=True)

    gender: str
    score: int
    ai_score: int = Field(serialization_alias="aiScore", description="보정된 건강/난임 점수(동일 값)")
    risk_probability: float
    bmi: float
    top_factors: list[str] = Field(description="활성 위험 요인 전체 목록(정상/긍정 요인 제외)")


class PredictResponse(BaseModel):
    status: Literal["success"] = "success"
    result: PredictResult


def _predict_core(data: dict[str, Any]) -> PredictResponse:
    if not engine.is_ready_for_gender(data["gender"]):
        raise RuntimeError(
            f"gender={data['gender']} 에 해당하는 모델이 없습니다. "
            f"load_errors={engine.load_errors}"
        )
    request_id = str(data.get("request_id") or data.get("session_id") or uuid4())
    enriched = dict(data)
    enriched["_request_id"] = request_id
    result = engine.predict(enriched)
    return PredictResponse(
        status="success",
        result=PredictResult(**result),
    )


@app.post("/api/predict/male", response_model=PredictResponse, response_model_by_alias=True)
async def predict_male(payload: PredictionRequest) -> PredictResponse:
    """스프링 `path.male` 기본값과 동일: POST /api/predict/male"""
    try:
        data = payload.model_dump(by_alias=False)
        data["gender"] = "male"
        return _predict_core(data)
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("추론 실패 (male)")
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "type": type(e).__name__,
                "traceback": tb,
            },
        ) from e


@app.post("/api/predict/female", response_model=PredictResponse, response_model_by_alias=True)
async def predict_female(payload: PredictionRequest) -> PredictResponse:
    """스프링 `path.female` 기본값과 동일: POST /api/predict/female"""
    try:
        data = payload.model_dump(by_alias=False)
        data["gender"] = "female"
        return _predict_core(data)
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("추론 실패 (female)")
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "type": type(e).__name__,
                "traceback": tb,
            },
        ) from e


@app.post("/predict", response_model=PredictResponse, response_model_by_alias=True)
async def predict(payload: PredictionRequestWithGender) -> PredictResponse:
    """하위 호환: body의 gender로 분기 (없으면 male)."""
    try:
        data = payload.model_dump(by_alias=False)
        assert data["gender"] is not None
        return _predict_core(data)
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("추론 실패")
        detail: dict[str, Any] = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": tb,
        }
        raise HTTPException(status_code=500, detail=detail) from e
