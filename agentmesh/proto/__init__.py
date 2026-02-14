"""
AgentMesh Protocol Buffers

gRPC service definitions for agent-LLM-tool communication.

Generated files (*_pb2.py, *_pb2_grpc.py) are excluded from version control.
To regenerate:
    python -m grpc_tools.protoc \
        -I./agentmesh/proto \
        --python_out=./agentmesh/proto \
        --grpc_python_out=./agentmesh/proto \
        ./agentmesh/proto/agentmesh.proto
"""
