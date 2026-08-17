# How One Sample Moves Through the System

[简体中文](how-it-works.zh-CN.md) | [Documentation](../README.md)

```text
archive/paired directory -> validated manifests -> Dataset + paired transform
-> image [B,3,H,W] + target [B,H,W] -> logits [B,2,H,W]
-> loss/update or argmax -> checkpoints, metric tables, masks, overlays
```

Manifests fix membership and portable paths. Dataset centralizes `1/2/3` to
`1/0/255` mapping and paired geometry. Models know only tensors and return the
same direct logits shape, whether handwritten U-Net or adapted DeepLabV3.

Cross-entropy and soft Dice provide gradients; validation pet IoU selects the
best checkpoint. Global metrics pool pixel counts while per-image scores support
diagnosis. Pixel accuracy alone can hide foreground failure.

Checkpoints preserve model, preprocessing, labels, configuration, version, and
training state. Predictor therefore does not ask users to reconstruct training
preprocessing. Numeric JSON/CSV, raw class-index PNG, and display overlays stay
separate because they serve different consumers.

The [learning path](../tutorial/learning-path.md) provides actions; this page explains why
the components are arranged this way.
