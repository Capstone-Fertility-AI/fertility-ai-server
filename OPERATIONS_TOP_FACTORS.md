# top_factors 운영 가이드

## 변경 요약

- 이전: `result.top_factors`가 사실상 3개 고정(Top3 중심 로직)
- 현재: `result.top_factors`는 **활성 위험요인 전체(0~N개)** 를 반환
- 더 이상 Top3 패딩/강제 길이 없음

## 샘플 응답 길이 변화

### 샘플 1: 위험요인 없음 (0개)

```json
{
  "status": "success",
  "result": {
    "gender": "female",
    "score": 95,
    "aiScore": 95,
    "risk_probability": 18.5,
    "bmi": 21.7,
    "top_factors": []
  }
}
```

### 샘플 2: 일부 위험요인 (2개)

```json
{
  "status": "success",
  "result": {
    "gender": "male",
    "score": 72,
    "aiScore": 72,
    "risk_probability": 37.1,
    "bmi": 26.2,
    "top_factors": ["비만", "흡연"]
  }
}
```

### 샘플 3: 다수 위험요인 (7개)

```json
{
  "status": "success",
  "result": {
    "gender": "female",
    "score": 40,
    "aiScore": 40,
    "risk_probability": 70.0,
    "bmi": 26.4,
    "top_factors": ["PCOS", "골반염 이력", "비만", "흡연", "잦은 음주", "생식기 질환", "성병·골반염 관련"]
  }
}
```

## 스펙

- 표준 키: `top_factors` (snake_case)
- 길이: 0~N
- 값: 활성 위험요인 문자열 배열(정상/긍정 요인 제외)

