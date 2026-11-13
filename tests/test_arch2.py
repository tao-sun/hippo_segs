"""Architecture, state and gradient contracts for the agreed Arch2."""
from copy import deepcopy
import json

import pytest
import torch
from torch import nn


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def options(**overrides):
    result = dict(encoder={'channels': [4, 6, 8]}, decoder={'channels': [8, 6, 4]},
                  ssm={'d_state': 2, 'expand': 1}, dropout=0.)
    result.update(overrides)
    return result


def reference_scan():
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
    return selective_scan_ref


def core(**overrides):
    from snn_nnunet.arch2 import Arch2
    return Arch2(selective_scan=reference_scan(), **options(**overrides))


def test_config_defaults_are_complete_json_and_do_not_mutate_input():
    from snn_nnunet.arch2_config import arch2_options
    supplied = options()
    before = deepcopy(supplied)
    resolved = arch2_options(supplied)
    assert supplied == before
    assert resolved['patch_size'] == 2
    assert resolved['encoder']['patch_embedding_spiking'] is True
    assert resolved['encoder']['vss_output_spiking'] is True
    assert resolved['decoder']['spiking'] is True
    assert arch2_options(json.loads(json.dumps(resolved))) == resolved


@pytest.mark.parametrize('invalid', [
    {'hierarchical_encoder': True}, {'stem': {}}, {'temporal_mamba': {}},
    {'encoder': {'output_spiking': False}}, {'encoder': {'depths': [1, 1, 2]}},
    {'decoder': {'full_resolution_skip': True}}, {'patch_size': True},
    {'encoder': {'channels': [4, 8]}}, {'decoder': {'spiking': 'false'}},
    {'ssm': {'dwconv2d_kernel_size': 2}}, {'ssm': {'dt_rank': False}},
    {'ssm': {'d_state': 0}}, {'neurons': {'init_tau': 1.}},
    {'neurons': {'surrogate_alpha': float('nan')}}, {'dropout': 1.},
])
def test_invalid_or_unused_options_are_rejected(invalid):
    from snn_nnunet.arch2_config import arch2_options
    with pytest.raises(ValueError):
        arch2_options(invalid)


def test_three_embeddings_and_vss_no_hierarchical_extensions():
    from snn_nnunet.spike_neurons import PLIFNode
    model = core()
    assert len(model.encoder) == 3
    for block in model.encoder:
        assert block.patch_embed.proj.kernel_size == (2, 2)
        assert block.patch_embed.proj.stride == (2, 2)
        assert isinstance(block.post_norm, nn.GroupNorm)
        assert isinstance(block.post_plif, PLIFNode)
    assert not any('stem' in name or 'extra_vss' in name or 'concat0' in name
                   for name, _ in model.named_modules())


def test_optional_plifs_use_gelu_and_decoder_is_independent():
    from snn_nnunet.spike_neurons import PLIFNode
    model = core(encoder={'channels': [4,6,8], 'patch_embedding_spiking': False,
                          'dwconv2d_spiking': False},
                 decoder={'channels': [8,6,4], 'spiking': False})
    for block in model.encoder:
        mixer = block.mixer.mamba_layer
        assert isinstance(mixer.patch_sl, nn.GELU)
        assert isinstance(mixer.dwconv2d_sl, nn.GELU)
        assert isinstance(mixer.patch_norm, nn.GroupNorm)
        assert isinstance(mixer.dwconv2d_norm, nn.GroupNorm)
        assert isinstance(block.post_plif, PLIFNode)
    assert isinstance(model.deconv_block1.spike_neurons, nn.GELU)
    assert isinstance(model.deconv3_conv.spike_neurons, nn.GELU)
    assert isinstance(model.class_conv.spike_neurons, nn.Identity)


@pytest.mark.parametrize('decoder_spiking', [False, True])
def test_vss_output_spiking_can_be_disabled_independently(decoder_spiking):
    from snn_nnunet.spike_neurons import PLIFNode
    model = core(encoder={'channels': [4,6,8], 'vss_output_spiking': False},
                 decoder={'channels': [8,6,4], 'spiking': decoder_spiking}).train()
    for block in model.encoder:
        assert isinstance(block.post_norm, nn.GroupNorm)
        assert isinstance(block.post_plif, nn.GELU)
        assert isinstance(block.mixer.mamba_layer.patch_sl, PLIFNode)
        assert isinstance(block.mixer.mamba_layer.dwconv2d_sl, PLIFNode)
    expected = PLIFNode if decoder_spiking else nn.GELU
    assert isinstance(model.deconv_block1.spike_neurons, expected)
    inputs = torch.randn(2,3,4,17,19, requires_grad=True)
    whole = model(inputs)
    chunked = torch.cat([model(inputs[:,:1]), model(inputs[:,1:], t0=1)], dim=2)
    torch.testing.assert_close(chunked, whole, rtol=0, atol=0)
    whole.square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize('invalid', [None, 0, 1, 'false'])
def test_vss_output_spiking_requires_a_boolean(invalid):
    from snn_nnunet.arch2_config import arch2_options
    with pytest.raises(ValueError, match='vss_output_spiking'):
        arch2_options({'encoder': {'vss_output_spiking': invalid}})


def test_four_scans_use_row_column_order_and_reversals():
    mixer = core().encoder[0].mixer.mamba_layer
    tokens = torch.arange(6.).reshape(1,2,3,1).expand(1,2,3,mixer.d_inner)
    routes = mixer._cross_scan_routes(tokens, 2, 3)
    expected = ([0,1,2,3,4,5], [5,4,3,2,1,0], [0,3,1,4,2,5], [5,2,4,1,3,0])
    for route, order in zip(routes, expected):
        assert route[0,:,0].tolist() == order


@pytest.mark.parametrize('patch_size', [1, 2, 3])
def test_shapes_skips_and_finite_backward_on_odd_rectangular_inputs(patch_size):
    model = core(patch_size=patch_size).train()
    inputs = torch.randn(2,2,4,17,19, requires_grad=True)
    output = model(inputs)
    assert output.shape == (2,3,2,17,19)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert inputs.grad.abs().sum() > 0
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_membrane_state_chunks_reset_and_different_volume_shapes():
    model = core().eval()
    inputs = torch.randn(1,4,4,17,19)
    whole = model(inputs)
    chunked = torch.cat([model(inputs[:,:2], t0=0), model(inputs[:,2:], t0=2)], dim=2)
    torch.testing.assert_close(chunked, whole, rtol=0, atol=0)
    torch.testing.assert_close(model(inputs), whole, rtol=0, atol=0)
    model.detach_states()
    for module in model.modules():
        if hasattr(module, 'v') and isinstance(module.v, torch.Tensor):
            assert module.v.grad_fn is None
    assert model(torch.randn(2,1,4,23,25)).shape == (2,3,1,23,25)


def test_raw_classifier_fp32_and_three_region_contract():
    from snn_nnunet.arch2 import Arch2
    model = core().eval()
    assert model.class_conv.conv.out_channels == 3
    assert not model.class_conv.normalization
    with pytest.raises(TypeError, match='FP32'):
        model(torch.randn(1,1,4,16,16).half())
    with pytest.raises(ValueError, match='three|3'):
        Arch2(out_channels=4)


def test_neuron_config_reaches_all_encoder_and_decoder_plifs():
    from snn_nnunet.spike_neurons import PLIFNode
    model = core(neurons={'init_tau': 3., 'surrogate_alpha': 4., 'v_threshold': .8,
                          'v_reset': None, 'detach_reset': False})
    for module in model.modules():
        if isinstance(module, PLIFNode):
            assert module.tau() == pytest.approx(3.)
            assert module.surrogate_function.alpha == 4.
            assert module.v_threshold == .8
            assert module.v_reset is None
            assert module.detach_reset is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_scan_matches_reference_forward_and_backward():
    from snn_nnunet.arch2 import Arch2
    native = Arch2(**options()).cuda().eval()
    reference = Arch2(selective_scan=reference_scan(), **options()).cuda().eval()
    reference.load_state_dict(native.state_dict())
    data = torch.randn(2,2,4,16,24, device='cuda')
    actual, expected = native(data), reference(data)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)
    actual.square().mean().backward()
    expected.square().mean().backward()
    for (name, parameter), (_, other) in zip(native.named_parameters(), reference.named_parameters()):
        torch.testing.assert_close(parameter.grad, other.grad, rtol=2e-2, atol=1e-3, msg=name)
