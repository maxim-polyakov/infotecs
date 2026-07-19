from ueba_prototype.demo import generate_normal_rows
from ueba_prototype.features import rows_to_matrix
from ueba_prototype.model import train_model


def test_autoencoder_flags_synthetic_anomaly() -> None:
    normal = rows_to_matrix(generate_normal_rows(count=120, seed=11))
    bundle = train_model(normal, epochs=120, learning_rate=0.015, threshold_quantile=0.98, seed=1)

    anomaly = normal[:1].copy()
    anomaly[0, 9] = 250.0
    anomaly[0, 12] = 90.0
    errors, _, _ = bundle.evaluate(anomaly)

    assert errors[0] > bundle.threshold
