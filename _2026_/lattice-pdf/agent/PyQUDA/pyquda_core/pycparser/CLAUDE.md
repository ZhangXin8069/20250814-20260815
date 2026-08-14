# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory.

## Overview

**Uninitialized git submodule placeholder.** `pyquda_core` vendors `pycparser` (a dependency for parsing C headers in generated wrappers) as a submodule; this directory is empty until the submodule is initialized. The vendored, populated copy used at runtime lives in `agent/PyQUDA/pyquda_plugins/pycparser/`.

## Initialize

```bash
cd agent/PyQUDA
git submodule update --init --recursive   # populates pyquda_core/pycparser
```

## Conventions

- Do not add source files here directly — populate via the submodule.
- See `agent/PyQUDA/CLAUDE.md` and `agent/PyQUDA/pyquda_plugins/pycparser/CLAUDE.md`.
