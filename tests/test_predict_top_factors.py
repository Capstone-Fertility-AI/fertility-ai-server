import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import main


def _payload():
    return {
        "age": 30,
        "height": 170,
        "weight": 65,
        "smoker": 0,
        "smoke_amount": 0,
        "drink_freq": 0,
        "binge_freq": 0,
        "num_bio_kid": 0,
        "sex_freq": 4,
        "has_sex_12mo": 1,
        "chlam": 0,
        "gon": 0,
        "parity": 0,
        "pcos": 0,
        "endo": 0,
        "uf": 0,
        "pid": 0,
        "menarche_age": 13,
    }


def _stub_result(top_factors):
    return {
        "gender": "male",
        "score": 88,
        "ai_score": 88,
        "risk_probability": 22.0,
        "bmi": 22.5,
        "top_factors": top_factors,
    }


class TopFactorsApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    def _post_with_factors(self, factors):
        with patch.object(main.engine, "is_ready_for_gender", return_value=True), patch.object(
            main.engine, "predict", return_value=_stub_result(factors)
        ):
            return self.client.post("/api/predict/male", json=_payload())

    def test_top_factors_zero_items(self):
        r = self._post_with_factors([])
        self.assertEqual(r.status_code, 200)
        result = r.json()["result"]
        self.assertIn("top_factors", result)
        self.assertEqual(result["top_factors"], [])
        self.assertNotIn("top1Factor", result)
        self.assertNotIn("mission_candidates", result)

    def test_top_factors_one_item(self):
        r = self._post_with_factors(["흡연"])
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()["result"]["top_factors"]), 1)

    def test_top_factors_two_items(self):
        r = self._post_with_factors(["흡연", "비만"])
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()["result"]["top_factors"]), 2)

    def test_top_factors_three_items(self):
        r = self._post_with_factors(["흡연", "비만", "잦은 음주"])
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()["result"]["top_factors"]), 3)

    def test_top_factors_seven_items(self):
        factors = ["흡연", "비만", "잦은 음주", "과도한 폭음", "성병 이력", "고위험 연령(35세 이상)", "PCOS"]
        r = self._post_with_factors(factors)
        self.assertEqual(r.status_code, 200)
        result = r.json()["result"]
        self.assertEqual(len(result["top_factors"]), 7)
        self.assertEqual(result["top_factors"], factors)

if __name__ == "__main__":
    unittest.main()

