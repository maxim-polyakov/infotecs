from ueba_prototype.demo import generate_normal_rows
from ueba_prototype.features import rows_to_matrix
from ueba_prototype.model import ModelBundle, train_model


def test_autoencoder_flags_synthetic_anomaly() -> None:
    normal = rows_to_matrix(generate_normal_rows(count=120, seed=11))
    bundle = train_model(normal, epochs=120, learning_rate=0.015, threshold_quantile=0.98, seed=1)

    anomaly = normal[:1].copy()
    anomaly[0, 9] = 250.0
    anomaly[0, 12] = 90.0
    errors, _, _ = bundle.evaluate(anomaly)

    assert errors[0] > bundle.threshold


def test_threat_classifier_is_saved_and_loaded(tmp_path) -> None:
    normal = rows_to_matrix(generate_normal_rows(count=120, seed=12))
    bundle = train_model(normal, epochs=80, learning_rate=0.015, threshold_quantile=0.98, seed=2)
    model_dir = tmp_path / "model"
    bundle.save(model_dir)

    loaded = ModelBundle.load(model_dir)
    anomaly = normal[:1].copy()
    anomaly[0, 15] = anomaly[0, 15] + 250000.0
    _, normalized, _ = loaded.evaluate(anomaly)
    prediction = loaded.classify_threat(normalized)

    assert prediction["threat_class"] in {
        "data_exfiltration",
        "c2_or_remote_access",
        "port_tunneling_or_policy_bypass",
        "process_execution_burst",
        "resource_abuse",
    }
    assert float(prediction["threat_confidence"]) > 0
