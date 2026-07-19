import numpy as np

from ueba_prototype.features import FeatureNormalizer, rows_to_matrix


def test_rows_to_matrix_and_normalizer_roundtrip() -> None:
    rows = [
        {"cpu_percent": 10, "memory_percent": 40, "connection_count": 5},
        {"cpu_percent": 20, "memory_percent": 50, "connection_count": 7},
        {"cpu_percent": 30, "memory_percent": 60, "connection_count": 9},
    ]
    columns = ["cpu_percent", "memory_percent", "connection_count"]

    matrix = rows_to_matrix(rows, columns)
    normalizer = FeatureNormalizer.fit(matrix)
    restored = normalizer.inverse_transform(normalizer.transform(matrix))

    assert matrix.shape == (3, 3)
    assert np.allclose(restored, matrix)
