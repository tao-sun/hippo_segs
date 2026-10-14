"""User-facing CLI contracts over installed nnU-Net 2.8.1 APIs."""

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from snn_nnunet import prepare_plans
from snn_nnunet import cli
from snn_nnunet import run_model


def _workspace(tmp_path, monkeypatch):
    for name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        folder = tmp_path / name
        folder.mkdir()
        monkeypatch.setenv(name, str(folder))
    dataset = tmp_path / "nnUNet_preprocessed" / "Dataset011_BraTS"
    dataset.mkdir()
    plans = prepare_plans.derive_snn_plans(
        {"dataset_name": "Dataset011_BraTS", "plans_name": "Stock", "configurations": {
            "3d_fullres": {"data_identifier": "Stock_3d_fullres", "patch_size": [96, 96, 96], "batch_size": 2}
        }}, prepare_plans.DEFAULT_SNN_CONFIG,
    )
    (dataset / "SNNPlans.json").write_text(json.dumps(plans))
    return dataset / "SNNPlans.json"


def test_parser_exposes_five_commands_and_approved_training_defaults():
    parser = cli.build_parser()
    assert {"prepare", "train", "validate", "predict", "full-cv"} <= {
        action for action in parser._subparsers._group_actions[0].choices
    }
    train = parser.parse_args(["train", "--dataset-id", "11", "--fold", "0"])
    assert (train.model, train.temporal_axis, train.k, train.use_fptt, train.gpus) == (
        "orig", 0, 16, True, 1,
    )
    changed = parser.parse_args(["train", "--dataset-id", "11", "--fold", "4", "--model", "deep",
                                 "--temporal-axis", "2", "--k", "8", "--no-fptt", "--gpus", "2", "--continue"])
    assert (changed.model, changed.temporal_axis, changed.k, changed.use_fptt,
            changed.gpus, changed.continue_training) == ("deep", 2, 8, False, 2, True)
    predict = parser.parse_args(["predict", "--dataset-id", "11", "--input", "images", "--output", "pred"])
    assert predict.folds == (0, 1, 2, 3, 4)
    assert predict.checkpoint == "checkpoint_final.pth"
    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--dataset-id", "11", "--fold", "5"])


def test_configure_runtime_sets_external_discovery_before_native_import(monkeypatch):
    monkeypatch.delenv("nnUNet_extTrainer", raising=False)
    monkeypatch.delenv("nnUNet_compile", raising=False)
    cli.configure_runtime()
    assert Path(os.environ["nnUNet_extTrainer"]).resolve() == Path(cli.__file__).resolve().parent
    assert os.environ["nnUNet_compile"] == "false"
    monkeypatch.setenv("nnUNet_compile", "true")
    cli.configure_runtime()
    assert os.environ["nnUNet_compile"] == "true"


def test_prepare_registers_dataset_and_runs_verified_native_preparation(tmp_path, monkeypatch):
    _workspace(tmp_path, monkeypatch)
    calls = []
    conversion = importlib.import_module("snn_nnunet.dataset_conversion")
    monkeypatch.setattr(conversion, "convert_or_register_dataset", lambda *a: calls.append(("convert", a)))
    monkeypatch.setattr(prepare_plans, "run_native_prepare", lambda *a: calls.append(("prepare", a)))
    assert cli.main(["prepare", "--dataset-root", str(tmp_path / "extracted"), "--dataset-id", "11",
                     "--dataset-name", "BraTS", "--fingerprint-processes", "3", "--preprocess-processes", "2"]) == 0
    assert calls == [
        ("convert", (tmp_path / "extracted", 11, "BraTS", tmp_path / "nnUNet_raw")),
        ("prepare", (11, 3, 2)),
    ]


def test_train_persists_complete_config_and_passes_native_flags(tmp_path, monkeypatch):
    plans_path = _workspace(tmp_path, monkeypatch)
    run_dir = tmp_path / "nnUNet_results" / "resume_run"
    run_dir.mkdir()
    saved_plans = json.loads(plans_path.read_text())
    saved_plans["snn_config"].update(model_name="medium", temporal_axis=1, k=8, use_fptt=False)
    (run_dir / "run_config.json").write_text(json.dumps({"plans": saved_plans}))
    training = importlib.import_module("nnunetv2.run.run_training")
    calls = []
    monkeypatch.setattr(training, "run_training", lambda *a, **kw: calls.append((a, kw)))
    assert cli.main(["train", "--dataset-id", "11", "--fold", "2", "--model", "medium",
                     "--temporal-axis", "1", "--k", "8", "--no-fptt", "--gpus", "2", "--continue",
                     "--run-dir", str(run_dir)]) == 0
    assert calls == [((), {"dataset_name_or_id": "11", "configuration": "3d_fullres", "fold": 2,
                          "trainer_class_name": "nnUNetTrainerSNNFPTT", "plans_identifier": "SNNPlans",
                          "num_gpus": 2, "continue_training": True, "only_run_validation": False,
                          "val_with_best": False, "export_validation_probabilities": False})]
    config = json.loads(plans_path.read_text())["snn_config"]
    assert config == {**prepare_plans.DEFAULT_SNN_CONFIG, "model_name": "medium", "temporal_axis": 1,
                      "k": 8, "use_fptt": False}


def test_incompatible_results_rejected_before_native_launch(tmp_path, monkeypatch):
    plans_path = _workspace(tmp_path, monkeypatch)
    run_dir = tmp_path / "nnUNet_results" / "existing_run"
    result = run_dir / "Dataset011_BraTS" / "nnUNetTrainerSNNFPTT__SNNPlans__3d_fullres"
    result.mkdir(parents=True)
    (result / "plans.json").write_text(plans_path.read_text())
    training = importlib.import_module("nnunetv2.run.run_training")
    calls = []
    monkeypatch.setattr(training, "run_training", lambda *a, **kw: calls.append((a, kw)))
    with pytest.raises(ValueError, match="incompatible"):
        cli.main(["train", "--dataset-id", "11", "--fold", "0", "--k", "8",
              "--run-dir", str(run_dir)])
    assert calls == []
    assert json.loads(plans_path.read_text())["snn_config"]["k"] == 16


def test_validate_uses_saved_config_and_native_best_flag(tmp_path, monkeypatch):
    _workspace(tmp_path, monkeypatch)
    run_dir = tmp_path / "nnUNet_results" / "validation_run"
    run_dir.mkdir()
    plans_path = tmp_path / "nnUNet_preprocessed" / "Dataset011_BraTS" / "SNNPlans.json"
    (run_dir / "run_config.json").write_text(json.dumps({"plans": json.loads(plans_path.read_text())}))
    training = importlib.import_module("nnunetv2.run.run_training")
    calls = []
    monkeypatch.setattr(training, "run_training", lambda *a, **kw: calls.append(kw))
    assert cli.main(["validate", "--dataset-id", "11", "--fold", "3", "--best", "--save-probabilities",
                     "--run-dir", str(run_dir)]) == 0
    assert calls == [{"dataset_name_or_id": "11", "configuration": "3d_fullres", "fold": 3,
                      "trainer_class_name": "nnUNetTrainerSNNFPTT", "plans_identifier": "SNNPlans",
                      "num_gpus": 1, "continue_training": False, "only_run_validation": True,
                      "val_with_best": True, "export_validation_probabilities": True}]


def test_full_cv_runs_folds_sequentially_with_one_config(tmp_path, monkeypatch):
    plans_path = _workspace(tmp_path, monkeypatch)
    training = importlib.import_module("nnunetv2.run.run_training")
    seen = []
    def run(*args, **kwargs):
        seen.append((kwargs["fold"], json.loads(plans_path.read_text())["snn_config"]["k"]))
    monkeypatch.setattr(training, "run_training", run)
    assert cli.main(["full-cv", "--dataset-id", "11", "--k", "4"]) == 0
    assert seen == [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]


def test_training_dataset_id_reaches_native_trainer_preconstruction(tmp_path, monkeypatch):
    plans_path = _workspace(tmp_path, monkeypatch)
    (plans_path.parent / "dataset.json").write_text("{}")
    native = importlib.import_module("nnunetv2.run.run_training")
    constructed = []

    class Trainer:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

    monkeypatch.setattr(native, "recursive_find_trainer_class_by_name", lambda name: Trainer)

    def enter_native(**kwargs):
        native.get_trainer_from_args(
            kwargs["dataset_name_or_id"], kwargs["configuration"], kwargs["fold"],
            kwargs["trainer_class_name"], kwargs["plans_identifier"], kwargs["continue_training"],
        )

    monkeypatch.setattr(native, "run_training", enter_native)
    assert cli.main(["train", "--dataset-id", "11", "--fold", "0"]) == 0
    assert len(constructed) == 1
    assert constructed[0]["plans"]["snn_config"]["k"] == 16
    assert constructed[0]["configuration"] == "3d_fullres"
    assert constructed[0]["fold"] == 0


def test_training_preserves_valid_edited_plans_fields_outside_cli_overrides(tmp_path, monkeypatch):
    plans_path = _workspace(tmp_path, monkeypatch)
    plans = json.loads(plans_path.read_text())
    plans["snn_config"]["model_kwargs"]["linear_projection"] = False
    plans["snn_config"]["model_kwargs"]["patch_embedding_spiking"] = False
    plans["snn_config"]["fptt_alpha"] = 0.75
    plans["snn_config"]["fptt_lambda"] = 3.0
    plans_path.write_text(json.dumps(plans))
    native = importlib.import_module("nnunetv2.run.run_training")
    monkeypatch.setattr(native, "run_training", lambda **kwargs: None)

    assert cli.main(["train", "--dataset-id", "11", "--fold", "0", "--model", "deep",
                     "--temporal-axis", "2", "--k", "8", "--no-fptt"]) == 0
    saved = json.loads(plans_path.read_text())["snn_config"]
    assert saved["model_kwargs"] == plans["snn_config"]["model_kwargs"]
    assert saved["fptt_alpha"] == 0.75
    assert saved["fptt_lambda"] == 3.0
    assert (saved["model_name"], saved["temporal_axis"], saved["k"], saved["use_fptt"]) == (
        "deep", 2, 8, False,
    )


def test_training_rejects_invalid_existing_plans_config_before_native_launch(tmp_path, monkeypatch):
    plans_path = _workspace(tmp_path, monkeypatch)
    plans = json.loads(plans_path.read_text())
    plans["snn_config"]["model_kwargs"]["patch_size"] = 0
    plans_path.write_text(json.dumps(plans))
    native = importlib.import_module("nnunetv2.run.run_training")
    calls = []
    monkeypatch.setattr(native, "run_training", lambda **kwargs: calls.append(kwargs))
    with pytest.raises(ValueError, match="patch_size"):
        cli.main(["train", "--dataset-id", "11", "--fold", "0"])
    assert calls == []
    assert json.loads(plans_path.read_text()) == plans


def test_predict_uses_exact_native_predictor_and_normalizes_checkpoint(tmp_path, monkeypatch):
    _workspace(tmp_path, monkeypatch)
    run_dir = tmp_path / "nnUNet_results" / "predict_run"
    run_dir.mkdir()
    native = importlib.import_module("nnunetv2.inference.predict_from_raw_data")
    seen = []
    class Predictor:
        def __init__(self, **kwargs):
            seen.append(("init", kwargs))
        def initialize_from_trained_model_folder(self, *args, **kwargs):
            seen.append(("model", args, kwargs))
        def predict_from_files(self, *args, **kwargs):
            seen.append(("files", args, kwargs))
    monkeypatch.setattr(native, "nnUNetPredictor", Predictor)
    assert cli.main(["predict", "--dataset-id", "11", "--input", "images", "--output", "pred",
                     "--folds", "0", "2", "4", "--checkpoint", "best",
                     "--run-dir", str(run_dir)]) == 0
    assert seen == [
        ("init", {"tile_step_size": 0.5, "use_gaussian": True, "use_mirroring": True}),
        ("model", (str(run_dir / "Dataset011_BraTS" /
                        "nnUNetTrainerSNNFPTT__SNNPlans__3d_fullres"),),
         {"use_folds": (0, 2, 4), "checkpoint_name": "checkpoint_best.pth"}),
        ("files", ("images", "pred"), {}),
    ]


def test_external_trainer_discovery_from_unrelated_directory(tmp_path):
    script = """import os
from pathlib import Path
from snn_nnunet.cli import configure_runtime
configure_runtime()
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
trainer = recursive_find_trainer_class_by_name('nnUNetTrainerSNNFPTT')
print(trainer.__name__)
print(Path(os.environ['nnUNet_extTrainer']).resolve())
"""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("nnUNet_extTrainer", None)
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env,
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "nnUNetTrainerSNNFPTT" in result.stdout
    assert str(Path(cli.__file__).resolve().parent) in result.stdout


def test_full_cv_can_run_native_best_configuration_selection(tmp_path, monkeypatch):
    _workspace(tmp_path, monkeypatch)
    training = importlib.import_module("nnunetv2.run.run_training")
    selection = importlib.import_module("nnunetv2.evaluation.find_best_configuration")
    events = []
    monkeypatch.setattr(training, "run_training", lambda **kw: events.append(kw["fold"]))
    monkeypatch.setattr(selection, "find_best_configuration", lambda *a, **kw: events.append(("best", a, kw)))
    assert cli.main(["full-cv", "--dataset-id", "11", "--select-best"]) == 0
    assert events == [0, 1, 2, 3, 4,
                      ("best", (11,), {"allowed_trained_models": ({"plans": "SNNPlans",
                            "configuration": "3d_fullres", "trainer": "nnUNetTrainerSNNFPTT"},),
                            "allow_ensembling": False, "folds": (0, 1, 2, 3, 4), "strict": True})]


def test_predict_postprocessing_uses_native_selection_and_application(tmp_path, monkeypatch):
    _workspace(tmp_path, monkeypatch)
    run_dir = tmp_path / "nnUNet_results" / "postprocess_run"
    run_dir.mkdir()
    native = importlib.import_module("nnunetv2.inference.predict_from_raw_data")
    selection = importlib.import_module("nnunetv2.evaluation.find_best_configuration")
    processing = importlib.import_module("nnunetv2.postprocessing.remove_connected_components")
    files = importlib.import_module("batchgenerators.utilities.file_and_folder_operations")
    events = []
    class Predictor:
        def __init__(self, **kwargs):
            pass
        def initialize_from_trained_model_folder(self, *args, **kwargs):
            pass
        def predict_from_files(self, *args, **kwargs):
            events.append("predict")
    monkeypatch.setattr(native, "nnUNetPredictor", Predictor)
    monkeypatch.setattr(selection, "find_best_configuration", lambda *a, **kw: {
        "best_model_or_ensemble": {"postprocessing_file": "native.pkl", "selected_model_or_models": [
            {"configuration": "3d_fullres", "trainer": "nnUNetTrainerSNNFPTT", "plans_identifier": "SNNPlans"}
        ]}})
    monkeypatch.setattr(files, "load_pickle", lambda path: (['native_fn'], [{"native": True}]))
    monkeypatch.setattr(processing, "apply_postprocessing_to_folder", lambda *a, **kw: events.append((a, kw)))
    assert cli.main(["predict", "--dataset-id", "11", "--input", "images", "--output", "pred",
                     "--postprocess", "--run-dir", str(run_dir)]) == 0
    assert events == ["predict", (("pred", "pred_postprocessed", ['native_fn'], [{"native": True}]), {})]


def _write_model_config(path, run_dir, family="unet2d"):
    path.write_text(yaml.safe_dump({
        "family": family,
        "dataset_id": 1,
        "fold": 0,
        "gpus": 1,
        "run_dir": str(run_dir),
    }))
    return path


def _write_native_plans(dataset_folder):
    dataset_folder.mkdir(parents=True)
    (dataset_folder / "nnUNetPlans.json").write_text(json.dumps({
        "configurations": {
            "2d": {"data_identifier": "nnUNetPlans_2d"},
            "3d_fullres": {"data_identifier": "nnUNetPlans_3d_fullres"},
        }
    }))


def test_model_config_accepts_unet2d_and_rejects_unknown_family(tmp_path):
    config_path = _write_model_config(tmp_path / "experiment.yaml", tmp_path / "results")
    assert run_model._config(config_path)["family"] == "unet2d"

    _write_model_config(config_path, tmp_path / "results", family="resnet")
    with pytest.raises(ValueError, match="unet2d, unet3d, or snn"):
        run_model._config(config_path)


@pytest.mark.parametrize(
    ("family", "configuration", "data_identifier"),
    [
        ("unet2d", "2d", "nnUNetPlans_2d"),
        ("unet3d", "3d_fullres", "nnUNetPlans_3d_fullres"),
    ],
)
def test_model_launcher_uses_selected_native_configuration(
    tmp_path, monkeypatch, family, configuration, data_identifier
):
    preprocessed = tmp_path / "preprocessed"
    dataset_folder = preprocessed / "Dataset001_BraTS24GLI"
    _write_native_plans(dataset_folder)
    (dataset_folder / data_identifier).mkdir()
    run_dir = tmp_path / "results"
    config_path = _write_model_config(
        tmp_path / "experiment.yaml", run_dir, family=family
    )
    calls = []

    monkeypatch.setenv("nnUNet_raw", str(tmp_path / "raw"))
    monkeypatch.setenv("nnUNet_preprocessed", str(preprocessed))
    monkeypatch.setattr(
        "nnunetv2.run.run_training.run_training",
        lambda **kwargs: calls.append(kwargs),
    )

    run_model.run(config_path)

    assert calls == [{
        "dataset_name_or_id": "1",
        "configuration": configuration,
        "fold": 0,
        "trainer_class_name": "nnUNetTrainer",
        "plans_identifier": "nnUNetPlans",
        "num_gpus": 1,
    }]
    assert yaml.safe_load((run_dir / "experiment.yaml").read_text())["family"] == family


def test_model_launcher_requires_preprocessed_2d_data(tmp_path, monkeypatch):
    preprocessed = tmp_path / "preprocessed"
    _write_native_plans(preprocessed / "Dataset001_BraTS24GLI")
    run_dir = tmp_path / "results"
    config_path = _write_model_config(tmp_path / "experiment.yaml", run_dir)
    monkeypatch.setenv("nnUNet_raw", str(tmp_path / "raw"))
    monkeypatch.setenv("nnUNet_preprocessed", str(preprocessed))

    with pytest.raises(ValueError, match="Missing preprocessed data for configuration '2d'"):
        run_model.run(config_path)

    assert not run_dir.exists()


def test_cli_accepts_temporal_mamba_options_and_persists_them(tmp_path, monkeypatch):
    plans_path = _workspace(tmp_path, monkeypatch)
    native = importlib.import_module("nnunetv2.run.run_training")
    monkeypatch.setattr(native, "run_training", lambda **kwargs: None)

    assert cli.main([
        "train", "--dataset-id", "11", "--fold", "0", "--temporal-mamba",
        "--temporal-mamba-kernel-size", "5", "--temporal-mamba-d-state", "8",
        "--temporal-mamba-dt-rank", "4", "--temporal-mamba-debug-shapes",
    ]) == 0

    model_kwargs = json.loads(plans_path.read_text())["snn_config"]["model_kwargs"]
    assert model_kwargs["temporal_mamba"] == {
        "enabled": True, "kernel_size": 5, "d_state": 8,
        "dt_rank": 4, "debug_shapes": True,
        "norm_spiking": True, "dwconv1D_spiking": True,
    }


def test_cli_persists_spatial_and_temporal_plif_switches(tmp_path, monkeypatch):
    plans_path = _workspace(tmp_path, monkeypatch)
    native = importlib.import_module("nnunetv2.run.run_training")
    monkeypatch.setattr(native, "run_training", lambda **kwargs: None)

    assert cli.main([
        "train", "--dataset-id", "11", "--fold", "0",
        "--no-spatial-patch-embedding-spiking", "--no-spatial-dwconv2d-spiking",
        "--no-spatial-output-spiking", "--no-vss-output-spiking",
        "--temporal-mamba", "--no-temporal-mamba-plif",
    ]) == 0

    model_kwargs = json.loads(plans_path.read_text())["snn_config"]["model_kwargs"]
    assert model_kwargs["patch_embedding_spiking"] is False
    assert model_kwargs["dwconv2d_spiking"] is False
    assert model_kwargs["spatial_output_spiking"] is False
    assert model_kwargs["vss_output_spiking"] is False
    assert "output_spiking" not in model_kwargs
    assert model_kwargs["temporal_mamba"]["norm_spiking"] is False
    assert model_kwargs["temporal_mamba"]["dwconv1D_spiking"] is False


def test_cli_persists_independent_temporal_plif_switches(tmp_path, monkeypatch):
    plans_path = _workspace(tmp_path, monkeypatch)
    native = importlib.import_module("nnunetv2.run.run_training")
    monkeypatch.setattr(native, "run_training", lambda **kwargs: None)

    assert cli.main([
        "train", "--dataset-id", "11", "--fold", "0", "--temporal-mamba",
        "--no-temporal-mamba-norm-spiking", "--temporal-mamba-dwconv1d-spiking",
    ]) == 0

    temporal = json.loads(plans_path.read_text())["snn_config"]["model_kwargs"]["temporal_mamba"]
    assert temporal["norm_spiking"] is False
    assert temporal["dwconv1D_spiking"] is True


def test_model_config_reads_temporal_mamba_from_brats_config():
    config = run_model._config(Path(__file__).resolve().parents[1] / "brats24_fold0_model.yaml")
    assert config["snn"]["temporal_mamba"] == {
        "enabled": True, "kernel_size": 3, "d_state": 16,
        "dt_rank": "auto", "debug_shapes": False,
        "norm_spiking": False, "dwconv1D_spiking": False,
    }
    assert config["snn"]["vss_output_spiking"] is False
    assert config["snn"]["spatial_mamba"] == {
        "patch_embedding_spiking": False,
        "dwconv2d_spiking": False,
        "output_spiking": False,
    }


def test_model_launcher_forwards_plif_switches_from_yaml(tmp_path, monkeypatch):
    preprocessed = tmp_path / "preprocessed"
    dataset_folder = preprocessed / "Dataset001_BraTS24GLI"
    dataset_folder.mkdir(parents=True)
    (dataset_folder / "SNNPlans.json").write_text(json.dumps({
        "configurations": {"3d_fullres": {"data_identifier": "snn_data"}},
    }))
    (dataset_folder / "snn_data").mkdir()
    run_dir = tmp_path / "results"
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump({
        "family": "snn", "dataset_id": 1, "fold": 0, "gpus": 1,
        "run_dir": str(run_dir),
        "snn": {
            "variant": "orig", "temporal_axis": 0, "k": 128, "use_fptt": False,
            "vss_output_spiking": False,
            "spatial_mamba": {
                "patch_embedding_spiking": False,
                "dwconv2d_spiking": True,
                "output_spiking": False,
            },
            "temporal_mamba": {
                "enabled": True, "kernel_size": 3, "d_state": 16,
                "dt_rank": "auto", "debug_shapes": False, "plif_enabled": False,
            },
        },
    }))
    calls = []
    monkeypatch.setenv("nnUNet_raw", str(tmp_path / "raw"))
    monkeypatch.setenv("nnUNet_preprocessed", str(preprocessed))
    monkeypatch.setattr(cli, "main", lambda args: calls.append(args))

    run_model.run(config_path)

    assert calls == [[
        "train", "--dataset-id", "1", "--fold", "0", "--run-dir", str(run_dir),
        "--gpus", "1", "--model", "orig", "--temporal-axis", "0", "--k", "128",
        "--no-fptt", "--no-vss-output-spiking", "--temporal-mamba",
        "--temporal-mamba-kernel-size", "3",
        "--temporal-mamba-d-state", "16", "--temporal-mamba-dt-rank", "auto",
        "--no-temporal-mamba-norm-spiking", "--no-temporal-mamba-dwconv1d-spiking",
        "--no-temporal-mamba-debug-shapes",
        "--no-spatial-patch-embedding-spiking",
        "--spatial-dwconv2d-spiking",
        "--no-spatial-output-spiking",
    ]]
