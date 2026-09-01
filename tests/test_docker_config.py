from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]


def test_docker_image_and_compose_define_gpu_training_runtime():
    dockerfile = (ROOT / "Dockerfile").read_text()
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    service = compose["services"]["trainer"]

    assert "cuda" in dockerfile.lower()
    assert "requirements.txt" in dockerfile
    assert "docker/entrypoint.sh" in dockerfile
    assert service["gpus"] == "all"
    assert "./data:/app/data" in service["volumes"]
    assert "./weights:/app/weights" in service["volumes"]
    assert "./outputs:/app/outputs" in service["volumes"]
    assert service["env_file"] == [".env"]


def test_secret_and_large_runtime_data_are_excluded_from_image_and_git():
    dockerignore = (ROOT / ".dockerignore").read_text().splitlines()
    gitignore = (ROOT / ".gitignore").read_text().splitlines()
    env = (ROOT / ".env").read_text()

    for name in (".env", "data", "weights", "outputs"):
        assert name in dockerignore
    assert ".env" in gitignore
    assert "KAGGLE_API_TOKEN=" in env
    assert env.split("KAGGLE_API_TOKEN=", 1)[1].splitlines()[0] == ""


def test_production_config_uses_canonical_download_paths():
    config = yaml.safe_load((ROOT / "config" / "training.yaml").read_text())

    assert config["paths"]["train_csv"] == "data/train.csv"
    assert config["paths"]["test_csv"] == "data/test.csv"
    assert config["paths"]["train_series_csv"] == "data/train_series.csv"
    assert config["paths"]["test_series_csv"] == "data/test_series.csv"
    assert config["paths"]["sample_submission_csv"] == "data/sample_submission.csv"
