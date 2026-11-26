# Documentation

The documentation is organized by pipeline stage:

```text
docs/
├── generation/           direct grammar term generation (seeds)
├── candidates/           novel-candidate overview and frontier drawing
└── counting/             size-layered counting and novel-size search
```

## Reading order

1. **Seed generation:** [Random term generation](generation/random_terms.md) —
   uniform exact-size sampling directly from a language grammar before any
   e-graph exists.
2. **Overview:** [Exact novel-candidate construction](candidates/novel_candidates.md) —
   core definitions (`novel`, `joint`), how the counting and drawing stages fit
   together, end-to-end correctness, a worked example, and the public API.
3. **Drawing & policies:** [Exact frontier drawing](candidates/frontier_drawing.md) —
   the two-state frontier automaton (`Novel` / `SharedWith`), agreement
   profiles, selection policies (`Count`, `Uniform`, `Smallest`), and
   `SampledTrie` without-replacement sampling.
4. **Counting kernel:** [Size-layered, root-restricted term counting](counting/layered_counting.md) —
   the generic `LayerTable` dynamic program (including cyclic e-graphs) and
   shared `MinSizes` / `RootBudgets`.
5. **Size search & package build:** [Root-restricted novel-size search and exact package construction](counting/novel_size_search.md) —
   geometric cap doubling, memory behavior of `search_step` / `search_steps`,
   budget-aware match enumeration, simultaneous whole/joint scanning, and
   final-size pruning.
