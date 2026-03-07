import unittest

import pandas as pd

from stock_picker.a_share_picker import FEATURE_COLUMNS, build_feature_frame


class FeatureBuilderTest(unittest.TestCase):
    def test_build_feature_frame_adds_expected_columns(self):
        rows = []
        for idx in range(90):
            base = 10 + idx * 0.1
            rows.append(
                {
                    "日期": pd.Timestamp("2025-01-01") + pd.Timedelta(days=idx),
                    "股票代码": "600000",
                    "开盘": base,
                    "收盘": base + 0.05,
                    "最高": base + 0.2,
                    "最低": base - 0.1,
                    "成交量": 1_000_000 + idx * 1_000,
                    "成交额": 120_000_000 + idx * 500_000,
                    "振幅": 2.0 + idx * 0.01,
                    "涨跌幅": 0.3,
                    "涨跌额": 0.03,
                    "换手率": 1.2 + idx * 0.01,
                }
            )

        history = pd.DataFrame(rows)
        features = build_feature_frame(history, symbol="600000", name="测试股票")

        for column in FEATURE_COLUMNS + ["target_next_return", "avg_amount_20", "symbol", "name"]:
            self.assertIn(column, features.columns)

        latest = features.iloc[-1]
        self.assertTrue(pd.isna(latest["target_next_return"]))
        self.assertEqual(latest["symbol"], "600000")
        self.assertEqual(latest["name"], "测试股票")


if __name__ == "__main__":
    unittest.main()
