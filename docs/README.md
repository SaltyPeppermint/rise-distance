# Documentation

The documentation is organized by purpose:

```text
docs/
├── guide_experiment.md   runnable experiment workflow
├── candidates/           guide-candidate construction
├── counting/             counting algorithms and size discovery
└── generation/           direct grammar term generation
```

## Start here

- To run the guide experiment, follow the
  [guide experiment pipeline](guide_experiment.md).
- For seed generation, read
  [random term generation](generation/random_terms.md).
- To understand how exact novel candidates are defined, counted, and drawn,
  start with
  [exact novel-candidate construction](candidates/exact_novel_candidates.md).
- To understand the shared frontier constraint and the available exact
  selection policies, read
  [exact frontier drawing policies](candidates/exact_frontier_drawing.md).
- For count-free, low-memory construction, read
  [rejection-candidate construction](candidates/rejection_candidates.md).

## Counting internals

The counting documents build on each other:

1. [Size-layered, root-restricted term counting](counting/layered_counting.md)
   explains the shared budget primitive and generic exact-size dynamic program,
   including cyclic e-graphs.
2. [Root-restricted novel-size search and exact package
   construction](counting/novel_size_search.md)
   explains budget-aware matching, cap-to-final pruning, rooted joint counting,
   package retention, and telemetry.
