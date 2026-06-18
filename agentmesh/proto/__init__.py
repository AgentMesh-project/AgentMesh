"""
AgentMesh Protocol Buffers

gRPC service definitions for the produce–transfer–consume dataflow between
agents, LLMs, and tools.

Generated files (*_pb2.py, *_pb2_grpc.py) are excluded from version control.
To regenerate:
    python -m grpc_tools.protoc \
        -I./agentmesh/proto \
        --python_out=./agentmesh/proto \
        --grpc_python_out=./agentmesh/proto \
        ./agentmesh/proto/agentmesh.proto
"""
