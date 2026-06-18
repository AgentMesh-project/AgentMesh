"""
Tests for the produce–transfer–consume dataflow abstraction.

Every multi-agent interaction is one directed dataflow edge with three stages:
produce, transfer, consume. These tests check the cost accounting
(total_cost), the barrier slack δ sign, the primary-mechanism mapping
(TOOL_TO_LLM→DTR, LLM_TO_LLM→BPP), and the enum surfaces.
"""

import pytest

from agentmesh.core.dataflow import (
    Dataflow,
    Producer,
    Consumer,
    DataflowEdge,
    Stage,
)


def _flow(edge, produce=0.0, transfer=0.0, consume=0.0, is_tool=False, payload=0):
    return Dataflow(
        producer=Producer(node_id="p", is_tool=is_tool),
        consumer=Consumer(node_id="c"),
        edge=edge,
        produce_cost=produce,
        transfer_cost=transfer,
        consume_cost=consume,
        payload_tokens=payload,
    )


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
def test_edge_enum_values():
    assert DataflowEdge.TOOL_TO_LLM.value == "tool_to_llm"
    assert DataflowEdge.LLM_TO_LLM.value == "llm_to_llm"


def test_stage_enum_members():
    assert {s.value for s in Stage} == {"produce", "transfer", "consume"}


# ---------------------------------------------------------------------------
# Cost accounting
# ---------------------------------------------------------------------------
def test_total_cost_is_sum_of_stages():
    flow = _flow(DataflowEdge.TOOL_TO_LLM, produce=1.0, transfer=0.5, consume=2.0)
    assert flow.total_cost == pytest.approx(3.5)


def test_total_cost_zero_default():
    flow = _flow(DataflowEdge.LLM_TO_LLM)
    assert flow.total_cost == 0.0


# ---------------------------------------------------------------------------
# Slack δ sign (BARRIER detection)
# ---------------------------------------------------------------------------
def test_positive_slack_consumer_waits():
    """produce+transfer > consume → positive slack (a BARRIER)."""
    flow = _flow(DataflowEdge.TOOL_TO_LLM, produce=2.0, transfer=1.0, consume=0.5)
    assert flow.slack == pytest.approx(2.5)
    assert flow.slack > 0


def test_negative_slack_producer_outpaced():
    """produce+transfer < consume → negative slack."""
    flow = _flow(DataflowEdge.LLM_TO_LLM, produce=0.2, transfer=0.1, consume=2.0)
    assert flow.slack == pytest.approx(-1.7)
    assert flow.slack < 0


def test_zero_slack_equilibrium():
    flow = _flow(DataflowEdge.TOOL_TO_LLM, produce=0.5, transfer=0.5, consume=1.0)
    assert flow.slack == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Primary-mechanism mapping
# ---------------------------------------------------------------------------
def test_tool_to_llm_maps_to_dtr():
    flow = _flow(DataflowEdge.TOOL_TO_LLM, is_tool=True)
    assert flow.primary_mechanism() == "DTR"


def test_llm_to_llm_maps_to_bpp():
    flow = _flow(DataflowEdge.LLM_TO_LLM)
    assert flow.primary_mechanism() == "BPP"


# ---------------------------------------------------------------------------
# Producer / Consumer surfaces
# ---------------------------------------------------------------------------
def test_producer_is_tool_flag():
    tool_producer = Producer(node_id="search_tool", is_tool=True)
    agent_producer = Producer(node_id="upstream_agent", is_tool=False)
    assert tool_producer.is_tool is True
    assert agent_producer.is_tool is False


def test_metadata_defaults_independent():
    p1 = Producer(node_id="a")
    p2 = Producer(node_id="b")
    p1.metadata["x"] = 1
    # Default factories must not share state across instances.
    assert p2.metadata == {}


def test_payload_tokens_carried():
    flow = _flow(DataflowEdge.LLM_TO_LLM, payload=128)
    assert flow.payload_tokens == 128
