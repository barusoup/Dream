# Dream Engine

A Rust game engine that doubles as an app framework. It includes BedFrame, a UI framework for apps and HUDs, and aims to be a working alternative to React-based app frameworks and to engines like Godot and Unity.

**Pre-alpha.** Work in progress. Many features are missing, and the code changes rapidly.

## Requirements

- Rust (edition 2024)
- GPU with Vulkan 1.3

## Run

```
cargo run --release --bin dream_app              # editor app
cargo run --release --bin dream_app -- --selfcheck
cargo run --release --bin dream_study            # headless integration check
cargo test --all-targets
```

Contributor conventions: [`AGENTS.md`](AGENTS.md). BedFrame guide: [`docs/bedframe-cookbook.md`](docs/bedframe-cookbook.md).

## License

AGPL-3.0 with additional attribution and commercial-license terms. See [`LICENSE`](LICENSE). Third-party assets: [`NOTICE`](NOTICE).
