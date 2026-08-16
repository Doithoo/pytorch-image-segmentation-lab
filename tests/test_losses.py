import pytest
import torch

from image_segmenter.training.losses import SegmentationLoss, foreground_soft_dice_loss, multiclass_soft_dice_loss


def test_ignored_logit_changes_do_not_change_dice() -> None:
    target = torch.tensor([[[1, 255]]])
    first = torch.tensor([[[[0.0, -5.0]], [[5.0, 5.0]]]])
    second = first.clone()
    second[:, :, 0, 1] = torch.tensor([100.0, -100.0])
    assert torch.allclose(
        foreground_soft_dice_loss(first, target, 255),
        foreground_soft_dice_loss(second, target, 255),
    )


def test_empty_foreground_uses_cross_entropy_only() -> None:
    criterion = SegmentationLoss(1.0, 1.0, ignore_index=255)
    logits = torch.zeros(1, 2, 2, 2, requires_grad=True)
    target = torch.zeros(1, 2, 2, dtype=torch.int64)
    total, components = criterion(logits, target)
    assert components["dice_active"].item() == 0.0
    assert torch.allclose(total, components["cross_entropy"])


def test_loss_rejects_batch_without_valid_pixels() -> None:
    criterion = SegmentationLoss(1.0, 0.0, ignore_index=255)
    with pytest.raises(ValueError, match="no valid pixels"):
        criterion(torch.zeros(1, 2, 2, 2), torch.full((1, 2, 2), 255))


def test_multiclass_dice_averages_selected_present_classes() -> None:
    logits = torch.tensor(
        [
            [
                [[10.0, -10.0, -10.0]],
                [[-10.0, 10.0, -10.0]],
                [[-10.0, -10.0, 10.0]],
            ]
        ]
    )
    target = torch.tensor([[[0, 1, 2]]])

    loss = multiclass_soft_dice_loss(logits, target, 255, class_indices=(0, 1, 2))

    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_segmentation_loss_uses_nonbackground_classes_for_multiclass_dice() -> None:
    criterion = SegmentationLoss(0.0, 1.0, ignore_index=255)
    logits = torch.zeros(1, 3, 1, 2, requires_grad=True)
    target = torch.tensor([[[1, 2]]])

    total, components = criterion(logits, target)

    assert components["dice_active"].item() == 1.0
    assert torch.isfinite(total)
