# Third-party (vendored)

Vendored upstream dependencies, shipped in-tree under their own licenses.

- **`verl-agent/`**: a modified vendored copy of
  [`langfengQ/verl-agent`](https://github.com/langfengQ/verl-agent) (itself built on
  [veRL](https://github.com/volcengine/verl)), Apache-2.0. It keeps its upstream
  `LICENSE` / `Notice.txt`; FedAgent's first-party hooks are woven in, and every
  modification is documented in [`verl-agent/CHANGES.md`](verl-agent/CHANGES.md).

Whole-project third-party attribution is in the root [`NOTICE`](../NOTICE).
