"""Additive efficiency patches for prime-rl, one module per paper mechanism.

Everything here is off by default (env-var / TOML-filter gated) so a checkout
with this patch applied still runs vanilla baseline GRPO unchanged unless a
condition explicitly turns a mechanism on. See ../../../README.md (repo root)
for which env vars map to which paper.
"""
