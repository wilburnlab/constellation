"""Neural-network primitives — ResNet blocks, MLPs, transformer scaffolds.

PascalCase names throughout (no legacy `resnet_block` snake_case).

Modules (TODO; scaffolded only):
    layers           - ResNetUnit, ResNetBlock, MonotonicMLP,
                       PositionalEncoding
    encoders         - Transformer encoder/decoder scaffolds (shared
                       between Chronologer/peptide encoder + CoLLAGE)
"""
