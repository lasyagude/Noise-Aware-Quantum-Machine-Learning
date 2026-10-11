"""Basic tests for the central configuration module."""

from qml_noise import config


def test_paths_exist():
    assert config.PROJECT_ROOT.is_dir()
    assert config.RESULTS_DIR.is_dir()
    assert config.PLOTS_DIR.is_dir()
    assert config.REPORT_DIR.is_dir()
    assert config.CONFIGS_DIR.is_dir()


def test_seeds():
    assert len(config.RANDOM_SEEDS) == 5
    assert config.DEFAULT_SEED == config.RANDOM_SEEDS[0]
    assert len(set(config.RANDOM_SEEDS)) == 5  # all unique


def test_shots_is_mandatory_value():
    assert config.SHOTS == 1024


def test_supported_datasets():
    assert "iris" in config.SUPPORTED_DATASETS
    assert "wine" in config.SUPPORTED_DATASETS
    assert "breast_cancer" in config.SUPPORTED_DATASETS
    assert config.PRIMARY_DATASET == "iris"


def test_classical_config_defaults():
    cfg = config.CLASSICAL_CONFIG
    assert cfg.logistic_regression_max_iter > 0
    assert cfg.svm_C > 0


def test_get_logger_returns_same_handlers_on_repeat_calls():
    logger1 = config.get_logger("qml_noise.test_logger")
    logger2 = config.get_logger("qml_noise.test_logger")
    assert logger1 is logger2
    assert len(logger1.handlers) == 1
