# aptree architecture

This document outlines the dependencis between internal modules within the `aptree` package and layer rules.

Core principles:
- Dependencies must flow **downwards only**.
- Lower layers must never import higher layers.

## Package Layout

Current layout:
```
aptree/
├── __init__.py
├── types
├── utils
├── structures

```


## Layer definitions
Layers exist to explicitly restrict and clarify module dependencies.

### Layer 0: types
- Purpose: Define core data types and interfaces used throughout the package.
- Allowed dependencies: None (base layer).

### Layer 1: utils
- Purpose: Provide utility functions and helpers that operate on core types.
- Allowed dependencies: Layer 0 (types).

### Layer 2: structures
- Purpose: Implement data structures and algorithms that utilize core types and utilities.
- Allowed dependencies: Layer 0 (types), Layer 1 (utils).

### Root Layer: aptree
- Purpose: Defines the public API surface of the package.
- Allowed dependencies: All layers.