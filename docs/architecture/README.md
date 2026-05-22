# Architecture Map

This directory holds the versioned architecture docs for the Python CLI knowledge-base prototype.

## Layers

- `high-level.md` explains product boundaries, runtime flow, data domains, and non-goals.
- `mid-level.md` maps subsystems, command-to-service boundaries, and data movement.
- `low-level.md` maps key files and current implementation responsibilities.

The agent merge point at commit `336a5ff` is treated as architecture, not just
CLI sugar: it adds an optional OpenAI Agents SDK boundary while preserving the
existing command/service split and local approval gates.

## Update Rules

Update the relevant layer when any of the following changes:

- command or service boundaries
- agent tool boundaries, approval policy, or session/persistence behavior
- raw/wiki/vault/graph data layout
- provider or tool orchestration boundaries
- CI or packaging behavior that changes the engineering workflow
- evaluation-oriented comparison behavior or export surfaces

## External References

- [OpenClaude and Browzy review](../../../Resources/Research/OpenClaude_Browzy_Architectural_Review.md)
- [OpenClaude local source tree](../../../Resources/openclaude/README.md)
- [Browzy local source tree](../../../Resources/browzy.ai/)
