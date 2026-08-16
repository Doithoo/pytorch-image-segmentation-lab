# Image Segmentation Tutorial

[简体中文](README.zh-CN.md) | [Back to README](../../README.md)

Do not read all six chapters before running anything. Complete the short README
workflow first, then open the chapter that answers the question in front of you.

| Chapter | Question it answers | Read it when |
| --- | --- | --- |
| [00 Segmentation basics](00-basics.md) | What are logits, masks, and cross-entropy? | Tensor shapes are unfamiliar |
| [01 Environment](01-environment.md) | How do I install, select a device, and judge speed? | Commands do not run |
| [02 Data and masks](02-data-and-masks.md) | How does a trimap become a target? | Preparing or replacing data |
| [03 U-Net](03-unet.md) | How do encoder, skips, and decoder connect? | Reading model code |
| [04 Training](04-training.md) | How do loss, optimizer, checkpoints, and resume interact? | Before a real experiment |
| [05 Evaluation and inference](05-evaluation-and-inference.md) | How do I read IoU, Dice, and error overlays? | After training |

A practical order is `00 -> 02 -> 03 -> 04 -> 05`. Keep chapter 01 nearby as a
setup reference. Each chapter ends with a question you can answer by running or
reading the project.

For a concrete failure, go directly to [troubleshooting](../troubleshooting.md).
