"""
AgentMesh Dataflow Abstraction

Formalizes the produce–transfer–consume abstraction that AgentMesh applies to
every multi-agent (MAS) interaction.

A *dataflow* is one directed edge in the MAS interaction graph. It has three
stages:

  produce  — a *producer* creates an artifact. The producer is either a tool
             execution or an upstream agent's LLM decoding.
  transfer — the artifact crosses the transport layer (in-process queue,
             gRPC stream, network hop, ...).
  consume  — a *consumer* ingests the artifact. The consumer is ALWAYS an LLM
             prefill (the downstream agent reads the artifact into its context).

A structure-blind runtime treats each dataflow as one opaque, coarse-grained
block, which exposes two pathologies:

  REDUNDANCY — the producer re-produces content the consumer (or cache) already
               holds. Addressed at the *produce* stage by DTR.
  BARRIER    — the consumer idles until the producer fully finishes. Addressed
               at the *consume* stage by BPP, and smoothed across all three
               stages by DES.

Mechanism → stage mapping:

  DTR (Delta Tool Retrieval)            → produce   (removes REDUNDANCY)
  BPP (Branch-Parallel Prefill)         → consume   (removes BARRIER)
  DES (Dynamic-Equilibrium Streaming)   → overlap   (drives slack δ→0)

The two dominant edge types in practice are ``TOOL_TO_LLM`` (tool result fed to
an agent's prefill) and ``LLM_TO_LLM`` (an upstream agent's decode fed to a
downstream agent's prefill).

This module is pure-Python and typed; it carries no heavy dependencies and is
safe to import anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional


class DataflowEdge(Enum):
    """
    The dominant dataflow edge types in a multi-agent system.

    Attributes:
        TOOL_TO_LLM: A tool's result is produced and consumed by an LLM prefill.
                     This edge is the primary target of DTR (Delta Tool
                     Retrieval), which serves the cached base and fetches only
                     the semantic delta.
        LLM_TO_LLM:  An upstream agent's LLM decoding is produced and consumed
                     by a downstream agent's LLM prefill. This edge is the
                     primary target of BPP (Branch-Parallel Prefill), which
                     prefills independent branches in parallel.
    """

    TOOL_TO_LLM = "tool_to_llm"
    LLM_TO_LLM = "llm_to_llm"


class Stage(Enum):
    """The three stages of a single dataflow."""

    PRODUCE = "produce"
    TRANSFER = "transfer"
    CONSUME = "consume"


@dataclass
class Producer:
    """
    The source side of a dataflow.

    Attributes:
        node_id: Identifier of the producing node (tool id or agent id).
        is_tool: True if the producer is a tool execution; False if it is an
                 upstream agent's LLM decoding.
        metadata: Free-form producer metadata.
    """

    node_id: str
    is_tool: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Consumer:
    """
    The sink side of a dataflow.

    The consumer is always an LLM prefill: a downstream agent reads the
    transferred artifact into its context window.

    Attributes:
        node_id: Identifier of the consuming agent.
        metadata: Free-form consumer metadata.
    """

    node_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Dataflow:
    """
    One produce–transfer–consume dataflow in the MAS interaction graph.

    The three cost fields decompose the wall-clock latency of the edge so the
    runtime can reason about REDUNDANCY (paid in ``produce_cost``) and BARRIER
    (paid as idle time inside ``consume_cost``). DES uses the relationship
    between produce and consume costs to drive the streaming slack to zero.

    Attributes:
        producer: The source side.
        consumer: The sink side.
        edge: The dataflow edge type (TOOL_TO_LLM or LLM_TO_LLM).
        produce_cost: Producing time (tool execution / upstream decode), seconds.
        transfer_cost: Transport time (network / queue), seconds.
        consume_cost: Consuming time (downstream prefill), seconds.
        payload_tokens: Size of the transferred artifact, in tokens.
        metadata: Free-form dataflow metadata.
    """

    producer: Producer
    consumer: Consumer
    edge: DataflowEdge
    produce_cost: float = 0.0
    transfer_cost: float = 0.0
    consume_cost: float = 0.0
    payload_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_cost(self) -> float:
        """Sum of produce, transfer, and consume costs (no overlap)."""
        return self.produce_cost + self.transfer_cost + self.consume_cost

    @property
    def slack(self) -> float:
        """
        The barrier slack δ of this dataflow.

        Positive slack means the consumer waits on the producer/transfer
        (a BARRIER); DES adjusts the streaming chunk size θ to drive δ→0.
        """
        return (self.produce_cost + self.transfer_cost) - self.consume_cost

    def primary_mechanism(self) -> str:
        """
        The AgentMesh mechanism that primarily optimizes this edge.

        TOOL_TO_LLM edges are dominated by REDUNDANCY in the produce stage →
        DTR. LLM_TO_LLM edges are dominated by the BARRIER in the consume
        stage → BPP. DES overlaps the stages of either edge.
        """
        if self.edge == DataflowEdge.TOOL_TO_LLM:
            return "DTR"
        return "BPP"


__all__ = [
    "DataflowEdge",
    "Stage",
    "Producer",
    "Consumer",
    "Dataflow",
]
