# Documentation

The documentation is organized by purpose:

```text
docs/
├── guide_experiment.md   runnable experiment workflow
├── counting/             counting algorithms and size discovery
└── sampling/             frontier semantics and sampling policies
```

## Start here

- To run the guide experiment, follow the
  [guide experiment pipeline](guide_experiment.md).
- To understand how novel terms are defined, counted, and sampled, start with
  [novel-term counting and sampling](sampling/novel_sampling.md).
- To understand the shared frontier constraint and the available sampling
  policies, read [frontier sampling policies](sampling/frontier_sampling.md).

## Counting internals

The counting documents build on each other:

1. [Size-layered term counting](counting/layered_counting.md) explains the
   generic dynamic program for exact-size histograms, including cyclic
   e-graphs and root-restricted budgets.
2. [Finding the smallest novel sizes](counting/novel_size_search.md) explains
   how `backoff_precompute` advances plain and joint counts together and stops
   at the requested novel size.
