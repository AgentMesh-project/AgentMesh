"""
AgentMesh Configuration Management

Provides configuration classes and utilities for setting up
AgentMesh components (SRR, TSD, ESF).
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import yaml
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class SRRConfig:
    """
    Configuration for SRR (Semantic Residual Retrieval).

    Key defaults:
      confidence_threshold (τ): 0.7  — gating threshold
      hotspot_k: 3                   — top-k hotspot tokens
      slm_model: Qwen2.5-0.5B-Instruct — SLM for query reformulation
    """
    enabled: bool = True
    confidence_threshold: float = 0.7       # τ — confidence gating threshold
    retrieval_depth: int = 10               # Number of similar entries to search
    hotspot_k: int = 3                      # Top-k hotspot tokens for reformulation
    cache_size: int = 1000                  # Maximum cache entries
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    slm_model: str = "Qwen/Qwen2.5-0.5B-Instruct"  # SLM for reformulation


@dataclass
class TSDConfig:
    """
    Configuration for TSD (Topology-aware Semantic Decoupling).

    Manages supervisor-worker TDA (Topologically Decoupled Attention)
    parameters for parallel KV prefill.
    """
    enabled: bool = True
    max_parallel_workers: int = 3           # Max concurrent worker prefills
    num_layers: int = 32                    # Transformer layers
    num_heads: int = 32                     # Attention heads
    head_dim: int = 128                     # Dimension per head
    rope_base: float = 10000.0              # RoPE base frequency


@dataclass
class ESFConfig:
    """
    Configuration for ESF (Elastic Semantic Flow).

    Controls adaptive microbatch sizing via OSE.
    """
    enabled: bool = True
    initial_microbatch_size: int = 512      # Initial θ (tokens)
    theta_min: int = 64                     # θ_min
    theta_max: int = 4096                   # θ_max
    damping_factor: float = 1.0             # Smoothing factor
    window_size: int = 5                    # OSE sliding window


@dataclass
class LLMConfig:
    """Configuration for LLM backend."""
    endpoint: str = "http://localhost:8000/v1"
    model: str = "Qwen/Qwen3-32B"
    api_key: Optional[str] = None
    timeout: int = 120
    max_tokens: int = 4096
    temperature: float = 0.7


@dataclass
class AgentMeshConfig:
    """
    Main configuration for AgentMesh runtime.

    Combines configurations for all three mechanisms (SRR, TSD, ESF)
    plus general runtime settings.
    """
    srr: SRRConfig = field(default_factory=SRRConfig)
    tsd: TSDConfig = field(default_factory=TSDConfig)
    esf: ESFConfig = field(default_factory=ESFConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    # General settings
    log_level: str = "INFO"
    device: str = "cuda"
    num_workers: int = 3

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMeshConfig":
        """Create config from dictionary."""
        srr_data = data.get("srr", {})
        tsd_data = data.get("tsd", {})
        esf_data = data.get("esf", {})
        llm_data = data.get("llm", {})

        return cls(
            srr=SRRConfig(**srr_data),
            tsd=TSDConfig(**tsd_data),
            esf=ESFConfig(**esf_data),
            llm=LLMConfig(**llm_data),
            log_level=data.get("log_level", "INFO"),
            device=data.get("device", "cuda"),
            num_workers=data.get("num_workers", 3),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "AgentMeshConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_env(cls) -> "AgentMeshConfig":
        """Create config from environment variables."""
        config = cls()

        # LLM settings
        if endpoint := os.environ.get("AGENTMESH_LLM_ENDPOINT"):
            config.llm.endpoint = endpoint
        if model := os.environ.get("AGENTMESH_LLM_MODEL"):
            config.llm.model = model
        if api_key := os.environ.get("AGENTMESH_API_KEY"):
            config.llm.api_key = api_key

        # Embedding model
        if emb_model := os.environ.get("AGENTMESH_EMBEDDING_MODEL"):
            config.srr.embedding_model = emb_model

        # SLM model
        if slm_model := os.environ.get("AGENTMESH_SLM_MODEL"):
            config.srr.slm_model = slm_model

        # SRR confidence threshold
        if threshold := os.environ.get("AGENTMESH_CONFIDENCE_THRESHOLD"):
            config.srr.confidence_threshold = float(threshold)

        # Log level
        if log_level := os.environ.get("AGENTMESH_LOG_LEVEL"):
            config.log_level = log_level

        # Device
        if device := os.environ.get("AGENTMESH_DEVICE"):
            config.device = device

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "srr": {
                "enabled": self.srr.enabled,
                "confidence_threshold": self.srr.confidence_threshold,
                "retrieval_depth": self.srr.retrieval_depth,
                "hotspot_k": self.srr.hotspot_k,
                "cache_size": self.srr.cache_size,
                "embedding_model": self.srr.embedding_model,
                "slm_model": self.srr.slm_model,
            },
            "tsd": {
                "enabled": self.tsd.enabled,
                "max_parallel_workers": self.tsd.max_parallel_workers,
                "num_layers": self.tsd.num_layers,
                "num_heads": self.tsd.num_heads,
                "head_dim": self.tsd.head_dim,
                "rope_base": self.tsd.rope_base,
            },
            "esf": {
                "enabled": self.esf.enabled,
                "initial_microbatch_size": self.esf.initial_microbatch_size,
                "theta_min": self.esf.theta_min,
                "theta_max": self.esf.theta_max,
                "damping_factor": self.esf.damping_factor,
                "window_size": self.esf.window_size,
            },
            "llm": {
                "endpoint": self.llm.endpoint,
                "model": self.llm.model,
                "timeout": self.llm.timeout,
                "max_tokens": self.llm.max_tokens,
                "temperature": self.llm.temperature,
            },
            "log_level": self.log_level,
            "device": self.device,
            "num_workers": self.num_workers,
        }

    def to_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def load_config(
    config_path: Optional[str] = None,
    use_env: bool = True,
) -> AgentMeshConfig:
    """
    Load configuration with priority: file > env > defaults.

    Args:
        config_path: Optional path to config file.
        use_env: Whether to apply environment variable overrides.

    Returns:
        Merged configuration.
    """
    if config_path and os.path.exists(config_path):
        config = AgentMeshConfig.from_yaml(config_path)
        logger.info(f"Loaded config from {config_path}")
    else:
        config = AgentMeshConfig()
        logger.info("Using default configuration")

    if use_env:
        env_config = AgentMeshConfig.from_env()
        if os.environ.get("AGENTMESH_LLM_ENDPOINT"):
            config.llm.endpoint = env_config.llm.endpoint
        if os.environ.get("AGENTMESH_LLM_MODEL"):
            config.llm.model = env_config.llm.model
        if os.environ.get("AGENTMESH_EMBEDDING_MODEL"):
            config.srr.embedding_model = env_config.srr.embedding_model
        if os.environ.get("AGENTMESH_SLM_MODEL"):
            config.srr.slm_model = env_config.srr.slm_model
        if os.environ.get("AGENTMESH_CONFIDENCE_THRESHOLD"):
            config.srr.confidence_threshold = env_config.srr.confidence_threshold
        if os.environ.get("AGENTMESH_LOG_LEVEL"):
            config.log_level = env_config.log_level

    return config
