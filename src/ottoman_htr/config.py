from dataclasses import dataclass, field


@dataclass
class VisionConfig:
    in_channels: int = 1
    embed_dim: int = 256


@dataclass
class GeometryConfig:
    feature_dim: int = 21  # see geometry.features.GEOMETRY_DIM
    embed_dim: int = 256


@dataclass
class LanguageModelConfig:
    order: int = 3


@dataclass
class ModelConfig:
    vision: VisionConfig = field(default_factory=VisionConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    language_model: LanguageModelConfig = field(default_factory=LanguageModelConfig)
