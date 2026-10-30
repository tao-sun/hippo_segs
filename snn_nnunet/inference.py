"""FP32 sliding-window inference for the SNN nnU-Net integration."""

from contextlib import contextmanager
import importlib
from threading import RLock

import torch
from tqdm import tqdm

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.helpers import empty_cache
from snn_nnunet.precision import configure_full_precision, require_fp32, require_fp32_module


class FP32nnUNetPredictor(nnUNetPredictor):
    """Keep network execution, Gaussian weights, and tile accumulation in FP32."""

    @torch.inference_mode()
    def _internal_predict_sliding_window_return_logits(
        self, data: torch.Tensor, slicers, do_on_device: bool = True
    ):
        results_device = self.device if do_on_device else torch.device("cpu")
        predicted_logits = n_predictions = prediction = gaussian = tile = None
        try:
            configure_full_precision()
            require_fp32(data, "Inference input")
            require_fp32_module(self.network)
            # The public nnU-Net predictor enables AMP outside this method.
            # Disable it here, inside that context, for the whole tile path.
            with torch.autocast(self.device.type, enabled=False):
                data = data.to(device=results_device, dtype=torch.float32)
                predicted_logits = torch.zeros(
                    (self.label_manager.num_segmentation_heads, *data.shape[1:]),
                    dtype=torch.float32, device=results_device,
                )
                n_predictions = torch.zeros(
                    data.shape[1:], dtype=torch.float32, device=results_device,
                )
                gaussian = (
                    compute_gaussian(
                        tuple(self.configuration_manager.patch_size),
                        sigma_scale=1. / 8, value_scaling_factor=10,
                        dtype=torch.float32, device=results_device,
                    )
                    if self.use_gaussian else 1.0
                )
                for sl in tqdm(slicers, disable=not self.allow_tqdm):
                    tile = data[sl].unsqueeze(0).contiguous().to(
                        device=self.device, dtype=torch.float32,
                    )
                    prediction = self._internal_maybe_mirror_and_predict(tile)[0].to(
                        device=results_device, dtype=torch.float32,
                    )
                    if not torch.isfinite(prediction).all():
                        raise FloatingPointError("Non-finite SNN logits during FP32 inference")
                    predicted_logits[sl] += prediction * gaussian
                    n_predictions[sl[1:]] += gaussian
                predicted_logits.div_(n_predictions)
                if not torch.isfinite(predicted_logits).all():
                    raise FloatingPointError("Non-finite logits after FP32 tile accumulation")
                return predicted_logits
        except Exception:
            # nnU-Net retries CPU accumulation while handling this exception.
            # Its traceback must not retain the failed attempt's GPU buffers.
            data = predicted_logits = n_predictions = prediction = gaussian = tile = None
            compute_gaussian.cache_clear()
            empty_cache(self.device)
            empty_cache(results_device)
            raise


_native_validation_lock = RLock()


@contextmanager
def native_fp32_predictor():
    """Use FP32 in the pinned native trainer's validation, then restore its factory.

    nnU-Net 2.8.1 constructs its predictor directly from a module global and
    provides no predictor factory argument to perform_actual_validation.
    """
    native_trainer = importlib.import_module(
        "nnunetv2.training.nnUNetTrainer.nnUNetTrainer"
    )
    with _native_validation_lock:
        original = native_trainer.nnUNetPredictor
        native_trainer.nnUNetPredictor = FP32nnUNetPredictor
        try:
            yield
        finally:
            native_trainer.nnUNetPredictor = original
