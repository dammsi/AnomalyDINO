import sys

import pytest

import run_anomalydino
import run_anomalydino_batched


def parse(module, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    return module.parse_args()


def test_defaults(monkeypatch):
    args = parse(run_anomalydino, monkeypatch)
    assert args.shots == [1]
    assert args.preprocess == "agnostic"
    assert args.mask_ref_images is False
    assert args.faiss_on_cpu is False


@pytest.mark.parametrize("argv, expected", [([], False), (["--mask_ref_images"], True), (["--no-mask_ref_images"], False)])
def test_mask_ref_images_flag(monkeypatch, argv, expected):
    assert parse(run_anomalydino, monkeypatch, *argv).mask_ref_images is expected


def test_multiple_shots(monkeypatch):
    assert parse(run_anomalydino, monkeypatch, "--shots", "1", "2", "4").shots == [1, 2, 4]


@pytest.mark.parametrize("module, argv", [
    (run_anomalydino, ["--preprocess", "agnostc"]),
    (run_anomalydino, ["--knn_metric", "cosine"]),
    (run_anomalydino, ["--mask_ref_images", "False"]),  # boolean flags do not take values
    (run_anomalydino_batched, ["--model_size", "xl"]),
])
def test_invalid_arguments_are_rejected(monkeypatch, module, argv):
    with pytest.raises(SystemExit):
        parse(module, monkeypatch, *argv)


def test_batched_args(monkeypatch):
    args = parse(run_anomalydino_batched, monkeypatch, "--model_size", "S", "--no-save_examples")
    assert args.model_size == "s"
    assert args.save_examples is False


def test_layer_argument(monkeypatch):
    assert parse(run_anomalydino, monkeypatch).layer is None
    assert parse(run_anomalydino, monkeypatch, "--layer", "9", "10").layer == [9, 10]
    assert parse(run_anomalydino, monkeypatch, "--layer", "-2").layer == [-2]
    assert parse(run_anomalydino_batched, monkeypatch, "--layer", "9").layer == [9]


@pytest.mark.parametrize("module", [run_anomalydino, run_anomalydino_batched])
def test_num_threads_argument(monkeypatch, module):
    assert parse(module, monkeypatch).num_threads == 4
    assert parse(module, monkeypatch, "--num_threads", "0").num_threads == 0
