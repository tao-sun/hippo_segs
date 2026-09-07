import torch
import torch.nn as nn

from model import print_model_info


class SelectiveScanLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 2, bias=False)
        self.A_log = nn.Parameter(torch.ones(3))


def test_model_info_counts_selective_scan_parameters(capsys):
    model = nn.Sequential(SelectiveScanLayer())

    print_model_info(model)

    output = capsys.readouterr().out
    fields = [line.split() for line in output.splitlines()]
    assert ["0", "3"] in fields
    assert ["0.projection", "4"] in fields
    assert ["Total", "trainable", "params:", "7"] in fields


def test_model_info_counts_shared_parameters_once(capsys):
    model = nn.Module()
    model.first = nn.Module()
    model.second = nn.Module()
    shared = nn.Parameter(torch.ones(3))
    model.first.register_parameter("weight", shared)
    model.second.register_parameter("weight", shared)

    print_model_info(model)

    output = capsys.readouterr().out
    fields = [line.split() for line in output.splitlines()]
    assert ["Total", "trainable", "params:", "3"] in fields
