import importlib.metadata
import subprocess
import sys


def test_python_310_environment_has_installed_project_and_nnunet(monkeypatch, tmp_path):
    assert sys.version_info[:2] == (3, 10)
    assert importlib.metadata.version("nnunetv2") == "2.8.1"

    import snn_nnunet

    assert snn_nnunet.__file__ is not None
    monkeypatch.chdir(tmp_path)
    subprocess.run(
        [sys.executable, "-I", "-c", "import snn_nnunet"],
        check=True,
        capture_output=True,
        text=True,
    )
