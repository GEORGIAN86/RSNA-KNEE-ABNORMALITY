from pathlib import Path

import pandas as pd

from Training.resource_planner import (
    Candidate,
    generate_balanced_candidates,
    estimate_candidate_memory,
    profile_data,
    profile_hardware,
    select_candidate,
    select_workers,
    write_planner_outputs,
)


def test_hardware_profile_tolerates_missing_available_pages(monkeypatch):
    def sysconf(name):
        if name == "SC_AVPHYS_PAGES":
            raise ValueError("unsupported")
        return {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 1000}[name]

    monkeypatch.setattr("Training.resource_planner.os.sysconf", sysconf)
    monkeypatch.setattr("Training.resource_planner.torch.cuda.is_available", lambda: False)

    profile = profile_hardware()

    assert profile["ram_total"] == 4_096_000
    assert profile["ram_available"] == profile["ram_total"]


def test_balanced_candidates_degrade_in_safe_order():
    candidates = generate_balanced_candidates()

    assert candidates[0].sam_slices == 2
    assert candidates[0].dino_blocks == 4
    assert candidates[0].sam_blocks == 2
    assert candidates[0].batch_size == 1
    assert candidates[0].gradient_accumulation == 8
    assert any(candidate.checkpointing for candidate in candidates[1:])
    assert candidates[-1].dino_blocks == 0


def test_selector_retries_oom_and_keeps_vram_margin():
    attempts = []

    def probe(candidate: Candidate):
        attempts.append(candidate)
        if len(attempts) == 1:
            return {"success": False, "oom": True, "peak_reserved": 0}
        return {"success": True, "oom": False, "peak_reserved": 7_000}

    selected, records = select_candidate(generate_balanced_candidates()[:3], probe, total_vram=10_000)

    assert selected == attempts[1]
    assert len(records) == 2


def test_worker_selection_is_bounded_by_cpu_and_ram():
    assert select_workers(cpu_count=16, available_ram=64 * 1024**3) == 8
    assert select_workers(cpu_count=4, available_ram=2 * 1024**3) == 1


def test_memory_estimate_decreases_with_smaller_sam_and_fewer_trainable_blocks():
    candidates = generate_balanced_candidates()

    largest = estimate_candidate_memory(candidates[0])
    smallest = estimate_candidate_memory(candidates[-1])

    assert largest["estimated_peak_bytes"] > smallest["estimated_peak_bytes"]
    assert largest["estimated_parameter_bytes"] == smallest["estimated_parameter_bytes"]


def test_profiles_csv_coverage_and_writes_outputs(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame({"StudyInstanceUID": ["a", "b"]}).to_csv(data / "train.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": ["t"]}).to_csv(data / "test.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": ["a", "a", "b"], "SeriesInstanceUID": ["1", "2", "3"]}).to_csv(data / "train_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": ["t"], "SeriesInstanceUID": ["4"]}).to_csv(data / "test_series.csv", index=False)
    config = {"paths": {"data_dir": data, "cache_dir": data / "cache", "train_csv": data / "train.csv", "test_csv": data / "test.csv", "train_series_csv": data / "train_series.csv", "test_series_csv": data / "test_series.csv", "outputs_dir": tmp_path / "outputs"}}

    profile = profile_data(config)
    report_path, config_path = write_planner_outputs(config, {"hardware": {}, "data": profile}, config)

    assert profile["train"]["studies"] == 2
    assert profile["train"]["series"] == 3
    assert report_path.is_file()
    assert config_path.is_file()
