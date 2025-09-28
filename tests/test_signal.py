import pytest

pd = pytest.importorskip("pandas")

from momentum.signal import logreturns
from config_loader import load_config_yaml


def test_logreturns_clip_limits_extreme_values(tmp_path):
    price = pd.DataFrame(
        {
            "asset": [1.0, 1.0, 1000.0, 0.001],
        },
        index=pd.date_range("2020-01-01", periods=4, freq="D"),
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
factor:
  window: 1
  skip: 0
  clip: 0.5
        """.strip()
    )

    config = load_config_yaml(config_path)
    signal = logreturns({"price": price}, config["factor"])

    # First row lacks sufficient history and should remain NaN
    assert pd.isna(signal.iloc[0, 0])

    # All finite signal entries are clipped to the configured bounds
    finite_signal = signal.iloc[1:, 0]
    assert (finite_signal <= 0.5).all()
    assert (finite_signal >= -0.5).all()