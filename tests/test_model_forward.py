import torch

from ottoman_htr.config import ModelConfig
from ottoman_htr.geometry.features import GEOMETRY_DIM
from ottoman_htr.model import OttomanManuscriptHTR, binarize_images, build_model


def test_forward_output_shape():
    config = ModelConfig()
    config.geometry.feature_dim = GEOMETRY_DIM
    vocab_size = 40
    model = OttomanManuscriptHTR(config, vocab_size=vocab_size)

    images = torch.rand(2, 1, 64, 256)
    geometry = torch.rand(2, GEOMETRY_DIM)

    log_probs = model(images, geometry)

    assert log_probs.shape[0] == 2
    assert log_probs.shape[2] == vocab_size
    assert torch.isfinite(log_probs).all()


def test_binarize_images_is_hard_zero_one():
    images = torch.rand(3, 1, 32, 64)
    binary = binarize_images(images)
    assert set(binary.unique().tolist()) <= {0.0, 1.0}
    assert binary.shape == images.shape


def test_binarize_images_flags_dark_pixels_as_ink():
    # one bright (background) image, one with a dark patch (ink) -- the patch should binarize to 1
    images = torch.ones(1, 1, 10, 10)
    images[0, 0, 4:6, 4:6] = 0.0
    binary = binarize_images(images)
    assert binary[0, 0, 4, 4] == 1.0
    assert binary[0, 0, 0, 0] == 0.0


def test_build_model_all_three_kinds_forward_correctly():
    config = ModelConfig()
    config.geometry.feature_dim = GEOMETRY_DIM
    vocab_size = 12
    images = torch.rand(2, 1, 64, 128)
    geometry = torch.rand(2, GEOMETRY_DIM)

    for kind in ["binary", "neural", "fuzzy"]:
        model = build_model(kind, config, vocab_size)
        log_probs = model(images, geometry)
        assert log_probs.shape[0] == 2
        assert log_probs.shape[2] == vocab_size
        assert torch.isfinite(log_probs).all()
        # log-probabilities: each timestep's class distribution must sum to ~1 in prob space
        assert torch.allclose(log_probs.exp().sum(dim=-1), torch.ones(2, log_probs.shape[1]), atol=1e-4)


def test_build_model_rejects_unknown_kind():
    config = ModelConfig()
    try:
        build_model("nonsense", config, vocab_size=5)
        assert False, "expected ValueError"
    except ValueError:
        pass
