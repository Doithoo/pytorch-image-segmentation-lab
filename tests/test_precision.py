import warnings

import pytest
import torch

from image_segmenter.training.precision import PrecisionController


def test_disabled_precision_uses_full_precision_backward() -> None:
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    controller = PrecisionController(torch.device("cpu"), requested=False)

    with controller.autocast():
        loss = parameter.square()
    controller.backward_step(loss, optimizer, [parameter], grad_clip=0.0)

    assert parameter.item() == pytest.approx(1.6)
    assert controller.effective is False
    assert controller.state_dict() == {"requested": False, "effective": False, "scaler": None}


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_non_cuda_amp_request_emits_notice_and_stays_disabled(device: str) -> None:
    with pytest.warns(UserWarning, match=f"AMP requested on {device}.*full precision"):
        controller = PrecisionController(torch.device(device), requested=True)

    assert controller.effective is False


def test_precision_rejects_changed_effective_semantics() -> None:
    controller = PrecisionController(torch.device("cpu"), requested=False)

    with pytest.raises(ValueError, match="effective AMP semantics"):
        controller.load_state_dict({"requested": True, "effective": True, "scaler": {}})


def test_legacy_missing_precision_state_is_allowed_for_full_precision() -> None:
    controller = PrecisionController(torch.device("cpu"), requested=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        controller.load_state_dict(None)
