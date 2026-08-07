import torch

from batdetect2.models.blocks import (
    SelfAttention,
    SelfAttentionConfig,
    VerticalMeanConfig,
)
from batdetect2.models.bottleneck import (
    Bottleneck,
    BottleneckConfig,
    build_bottleneck,
)


def test_bottleneck_layers_use_frequency_aggregation_channels() -> None:
    """Layers after frequency aggregation are built for aggregated channels."""
    config = BottleneckConfig(
        channels=128,
        frequency_aggregation=VerticalMeanConfig(channels=128),
        layers=[SelfAttentionConfig(attention_channels=32)],
    )

    bottleneck = build_bottleneck(
        input_height=8,
        in_channels=64,
        config=config,
    )

    assert isinstance(bottleneck, Bottleneck)
    attention = bottleneck.layers[0]
    assert isinstance(attention, SelfAttention)
    assert attention.in_channels == 128

    output = bottleneck(torch.randn(2, 64, 8, 10))

    assert output.shape == (2, 128, 8, 10)


def test_bottleneck_default_frequency_aggregation_matches_channels() -> None:
    """Minimal configs keep advertised and actual output channels in sync."""
    config = BottleneckConfig(channels=128, layers=[])

    bottleneck = build_bottleneck(
        input_height=8,
        in_channels=64,
        config=config,
    )

    assert isinstance(bottleneck, Bottleneck)
    assert bottleneck.out_channels == 128
    assert bottleneck.conv_vert.out_channels == 128

    output = bottleneck(torch.randn(2, 64, 8, 10))

    assert output.shape == (2, bottleneck.out_channels, 8, 10)
