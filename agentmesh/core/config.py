"""
AgentMesh Configuration Management

Provides configuration classes and utilities for setting up the three
AgentMesh mechanisms (DTR, BPP, DES) plus the LLM backend and general
runtime settings.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import yaml
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class DTRConfig:
    """
    Configuration for DTR (Delta Tool Retrieval) — the "produce" step.

    DTR serves the cached *base* of a tool result and fetches only the
    semantic *delta*, then async-refreshes the exact entry. This removes
    REDUNDANCY in the produce stage of a tool→LLM dataflow.

    Key defaults:
      confidence_threshold (τ): 0.8  — gating threshold (paper deployment)
      hotspot_k: 3                   — top-k hotspot tokens
      slm_model: Qwen2.5-0.5B-Instruct — SLM for query reformulation
    """
    enabled: bool = True
    confidence_threshold: float = 0.8       # τ — confidence gating threshold
    retrieval_depth: int = 10               # Number of similar entries to search
    hotspot_k: int = 3                      # Top-k hotspot tokens for reformulation
    cache_size: int = 1000                  # Maximum cache entries
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    slm_model: str = "Qwen/Qwen2.5-0.5B-Instruct"  # SLM for reformulation


@dataclass
class BPPConfig:
    """
    Configuration for BPP (Branch-Parallel Prefill) — the "consume" step.

    Controls Branch-Parallel Attention (BPA) over the supervisor-worker
    topology: independent worker branches are prefilled in parallel against
    the shared supervisor prefix, removing the BARRIER in the consume stage
    of an LLM→LLM dataflow.

    Note: ``max_parallel_branches`` was historically named
    ``max_parallel_workers``; the field is renamed but the semantics are
    identical (max concurrent branch prefills).
    """
    enabled: bool = True
    max_parallel_branches: int = 3          # Max concurrent branch prefills
    num_layers: int = 32                    # Transformer layers
    num_heads: int = 32                     # Attention heads
    head_dim: int = 128                     # Dimension per head
    rope_base: float = 10000.0              # RoPE base frequency


@dataclass
class DESConfig:
    """
    Configuration for DES (Dynamic-Equilibrium Streaming).

    Drives the pipeline to a dynamic equilibrium (slack δ→0) by adapting the
    streaming chunk size θ via Online Sensitivity Estimation (OSE). This
    overlaps the produce/transfer/consume stages of a dataflow.
    """
    enabled: bool = True
    initial_theta: int = 512                # Initial θ (tokens)
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

    Combines configurations for all three mechanisms (DTR, BPP, DES)
    plus general runtime settings.
    """
    dtr: DTRConfig = field(default_factory=DTRConfig)
    bpp: BPPConfig = field(default_factory=BPPConfig)
    des: DESConfig = field(default_factory=DESConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    # General settings
    log_level: str = "INFO"
    device: str = "cuda"
    num_workers: int = 3

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMeshConfig":
        """Create config from dictionary."""
        dtr_data = data.get("dtr", {})
        bpp_data = data.get("bpp", {})
        des_data = data.get("des", {})
        llm_data = data.get("llm", {})

        return cls(
            dtr=DTRConfig(**dtr_data),
            bpp=BPPConfig(**bpp_data),
            des=DESConfig(**des_data),
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
            config.dtr.embedding_model = emb_model

        # SLM model
        if slm_model := os.environ.get("AGENTMESH_SLM_MODEL"):
            config.dtr.slm_model = slm_model

        # DTR confidence threshold
        if threshold := os.environ.get("AGENTMESH_CONFIDENCE_THRESHOLD"):
            config.dtr.confidence_threshold = float(threshold)

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
            "dtr": {
                "enabled": self.dtr.enabled,
                "confidence_threshold": self.dtr.confidence_threshold,
                "retrieval_depth": self.dtr.retrieval_depth,
                "hotspot_k": self.dtr.hotspot_k,
                "cache_size": self.dtr.cache_size,
                "embedding_model": self.dtr.embedding_model,
                "slm_model": self.dtr.slm_model,
            },
            "bpp": {
                "enabled": self.bpp.enabled,
                "max_parallel_branches": self.bpp.max_parallel_branches,
                "num_layers": self.bpp.num_layers,
                "num_heads": self.bpp.num_heads,
                "head_dim": self.bpp.head_dim,
                "rope_base": self.bpp.rope_base,
            },
            "des": {
                "enabled": self.des.enabled,
                "initial_theta": self.des.initial_theta,
                "theta_min": self.des.theta_min,
                "theta_max": self.des.theta_max,
                "damping_factor": self.des.damping_factor,
                "window_size": self.des.window_size,
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
            config.dtr.embedding_model = env_config.dtr.embedding_model
        if os.environ.get("AGENTMESH_SLM_MODEL"):
            config.dtr.slm_model = env_config.dtr.slm_model
        if os.environ.get("AGENTMESH_CONFIDENCE_THRESHOLD"):
            config.dtr.confidence_threshold = env_config.dtr.confidence_threshold
        if os.environ.get("AGENTMESH_LOG_LEVEL"):
            config.log_level = env_config.log_level

    return config
