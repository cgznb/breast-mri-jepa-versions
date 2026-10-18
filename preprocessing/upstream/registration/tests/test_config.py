from pathlib import Path

import pytest

import ispy2_registration.cli as registration_cli
import ispy2_registration.config as registration_config
from ispy2_registration.config import PathConfig, RegistrationConfig


def test_registration_defaults_target_t0() -> None:
    config = RegistrationConfig()

    assert config.fixed_visit == "T0"
    assert config.moving_visits == ("T1", "T2", "T3")
    assert config.workers == 4
    assert config.elastix_threads == 8
    assert config.seed == 20260719
    assert config.bspline_resolutions == 4
    assert config.bspline_iterations == (300, 300, 300, 1200)
    assert config.bspline_spatial_samples == 5000
    assert config.rigid_learning_rate == 2.0
    assert config.rigid_min_step == 1e-4
    assert config.rigid_relaxation_factor == 0.5
    assert config.rigid_shrink_factors == (4, 2, 1)
    assert config.rigid_smoothing_sigmas == (2.0, 1.0, 0.0)
    assert config.protected_margin_mm == 10.0
    assert config.max_outside_fold_fraction == 0.02


def test_registration_config_rejects_invalid_concurrency() -> None:
    with pytest.raises(ValueError, match="workers"):
        RegistrationConfig(workers=0)

    with pytest.raises(ValueError, match="elastix_threads"):
        RegistrationConfig(elastix_threads=0)


def test_path_defaults_match_current_dataset() -> None:
    paths = PathConfig()

    assert paths.input_root == Path(
        "/root/autodl-tmp/Datasets/manifest-1781750940386/ISPY2/"
        "processed_ispy2_world_phase3_scan"
    )
    assert paths.output_root == Path(
        "/root/autodl-tmp/Datasets/manifest-1781750940386/ISPY2/"
        "processed_ispy2_world_phase3_scan_phase_aligned_multires_registered_to_T0"
    )
    assert paths.manifest_path == paths.input_root / "manifest/ispy2_breastdcedl_full_manifest.csv"


def test_toml_populates_paths_registration_and_qc(tmp_path: Path) -> None:
    path = tmp_path / "registration.toml"
    path.write_text(
        """
[paths]
input_root = "/input"
manifest = "/input/manifest/data.csv"
output_root = "/output"

[registration]
workers = 2
elastix_threads = 3
seed = 17
tumor_weight = 0.75
rigid_samples = 1234
rigid_iterations = 321
rigid_learning_rate = 1.5
rigid_min_step = 0.0002
rigid_relaxation_factor = 0.6
rigid_shrink_factors = [4, 2, 1]
rigid_smoothing_sigmas = [2.0, 1.0, 0.0]

[qc]
max_ftv_volume_change = 0.2
protected_margin_mm = 8.0
max_outside_fold_fraction = 0.01
""",
        encoding="utf-8",
    )

    paths, registration = registration_config.load_toml_config(path)

    assert paths.manifest_path == Path("/input/manifest/data.csv")
    assert paths.output_root == Path("/output")
    assert registration.workers == 2
    assert registration.rigid_learning_rate == 1.5
    assert registration.max_ftv_volume_change == 0.2


def test_explicit_cli_values_override_toml(tmp_path: Path) -> None:
    path = tmp_path / "registration.toml"
    path.write_text(
        """
[paths]
input_root = "/input"
output_root = "/output"

[registration]
workers = 2
elastix_threads = 3
seed = 17
""",
        encoding="utf-8",
    )
    args = registration_cli.build_parser().parse_args(
        ["register-cohort", "--config", str(path), "--workers", "5"]
    )

    paths, registration = registration_cli._runtime_config(args)

    assert paths.output_root == Path("/output")
    assert registration.workers == 5
    assert registration.elastix_threads == 3
    assert registration.seed == 17
