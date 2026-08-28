from cfb_picks.calibration import edge_bucket, load_calibration, save_calibration


def test_edge_bucket_boundaries():
    assert edge_bucket(0) == "0-2"
    assert edge_bucket(1.9) == "0-2"
    assert edge_bucket(2) == "2-5"
    assert edge_bucket(4.9) == "2-5"
    assert edge_bucket(5) == "5-8"
    assert edge_bucket(7.9) == "5-8"
    assert edge_bucket(8) == "8+"
    assert edge_bucket(100) == "8+"


def test_save_and_load_calibration_roundtrip(tmp_path):
    calibration = {"0-2": 0.51, "2-5": 0.55, "5-8": 0.61, "8+": 0.58}
    path = tmp_path / "calibration.json"

    save_calibration(calibration, path)

    assert load_calibration(path) == calibration
