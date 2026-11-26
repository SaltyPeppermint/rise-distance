# Root-restricted novel-size search and exact package construction

When building a `FrontierPackage`, we want to find the **smallest** root AST
size `s` that provides at least `min_extractable` novel terms (terms
extractable from `curr` that cannot be extracted from any e-class in `prev`),
and retain just the tables needed to draw from it.

The challenge is that we do not know that winning size in advance. If we
immediately build match tables and size-layered DP histograms up to the maximum
allowed size (`cap`), we pay peak memory for `cap` even when the answer is
small.

`FrontierPackage::build` solves this by **growing the size cap geometrically**
from `min(root) + search_step` up to `cap = start_size + search_steps`,
advancing whole and joint counts layer by layer until the threshold is met, and
then pruning the retained tables down to the exact winning size.

> **Prerequisites & related docs:**
> - Read [layered_counting.md](layered_counting.md) first for `LayerTable`,
>   `MinSizes`, and `RootBudgets`.
> - For the high-level pipeline, worked example, and correctness proof, see
>   [novel_candidates.md](../candidates/novel_candidates.md).
> - For how the finished package is consumed during drawing, see
>   [frontier_drawing.md](../candidates/frontier_drawing.md).

## Where the code lives

- [src/sampling/count/budgets.rs](../../src/sampling/count/budgets.rs) —
  `MinSizes`, `RootBudgets`, and `doubling_caps`.
- [src/sampling/count/mod.rs](../../src/sampling/count/mod.rs) —
  generic `LayerTable` size-layered DP kernel.
- [src/sampling/count/novel.rs](../../src/sampling/count/novel.rs) —
  `NovelTermCount::build`, `enumerate_matches`, `step_joint`, `prune_matches`,
  and `NovelTermCount::assemble`.
- [src/sampling/draw/frontier.rs](../../src/sampling/draw/frontier.rs) —
  `FrontierPackage::build`, package retention, and diagnostics.

---

## 1. End-to-end build flow

```text
cap       = start_size + search_steps
min_sizes = MinSizes::new(curr)              // one AstSize pass, reused throughout
root_min  = min_sizes.get(root)

for try_cap in doubling_caps(root_min, cap, search_step):
    try_budgets = min_sizes.budgets_for(curr, root, try_cap)

    // 1. Enumerate matches against `prev`, then drop `prev` before DP allocates.
    prev_lookup = prev_boundary.prev_index(curr)
    matches     = enumerate_matches(curr, prev_lookup, try_budgets)
    drop(prev_lookup)

    // 2. Step whole and joint LayerTables together, one size at a time.
    term_count = 0
    for s in 1..=try_cap:
        whole::step(curr, try_budgets, whole, s)
        root_non_novel = step_joint(curr, matches, try_budgets, joint, s)
        novel = whole(root, s) - root_non_novel
        term_count += novel

        if term_count >= min_extractable:
            // 3. Tighten to the exact winning size (no second DP pass needed).
            final_budgets = min_sizes.budgets_for(curr, root, s)
            prune_matches(curr, matches, final_budgets)
            final_budgets.prune_histograms(whole)
            final_budgets.prune_histograms(joint)
            return Ok((s, NovelTermCount::assemble(whole, joint, matches)))

return Err(cap)
```

Key properties of this loop:

- **Reused `MinSizes`:** The bottom-up `AstSize` analysis runs once before the
  loop and is reused for every `try_cap` and for `final_budgets`.
- **Non-overlapping peak memory:** `prev_lookup` is built for match enumeration
  and dropped *before* `whole` and `joint` allocate their DP tables.
- **No second DP pass:** Once `term_count >= min_extractable` at `s`, the
  already-computed `whole` and `joint` tables are pruned in place to
  `final_budgets` and assembled directly.

`WholePackage::build` (`whole::count`) follows the exact same doubling-cap
loop, without `prev_lookup`, match enumeration, or joint counting.

---

## 2. How `search_step` and `search_steps` drive memory

Two parameters control the search window:

- **`search_step` (`initial_step`):** initial headroom above `min(root)`.
  `doubling_caps` tries `root_min + step`, `root_min + 2*step`,
  `root_min + 4*step`, ..., clamping the final attempt to `cap`.
- **`search_steps`:** sets the hard upper bound `cap = start_size + search_steps`.

Every attempt sizes its data structures against `try_cap`. Raising `try_cap`
increases memory in three places:

1. **More classes enter `RootBudgets`:** A larger `try_cap` raises every
   reachable class's budget and admits deeper classes whose cheapest term did
   not fit a smaller budget.
2. **Every layer touches more classes:** `step` visits every budgeted class
   with `s <= budget(c)`, so higher budgets keep more classes active at *every*
   layer—making histograms (and arity ≥ 3 suffix caches) denser per layer, not
   just taller.
3. **Match enumeration runs at `try_cap`:** `enumerate_matches` visits every
   e-node fitting `try_budgets` and builds the full match table before the scan
   knows the final size; `prune_matches` only narrows it after the scan
   succeeds.

### Early success vs. the give-up path

- **When the search succeeds quickly:** Geometric doubling keeps `try_cap`
  small (at most roughly double the needed headroom), and the inner loop stops
  as soon as `term_count >= min_extractable`. You never allocate full-`cap`
  match tables or DP layers.
- **When the search fails:** The loop eventually reaches `try_cap == cap` and
  runs all `cap` layers over the full cap-admitted graph before returning
  `Err(cap)`.

> **Practical trap:** Raising `search_steps` bounds the cost of *failing*, and
> that cost grows superlinearly (more layers × more classes × denser
> histograms). Under an external RSS limit, widening `search_steps` to rescue
> borderline start terms can turn terms that used to fail cleanly with "too few
> terms" into OOM kills—because the give-up path is the most expensive path.

---

## 3. Budget-aware match enumeration

We first find which budgeted e-nodes in `curr` correspond to e-nodes in `prev`.
`enumerate_matches` tracks two mutually dependent structures:

- **`cover[c]`** — canonical `prev` classes known to share at least one
  extractable term with `curr` class `c`.
- **`matches[(c, idx)]`** — list of `NodeMatch { prev_class: pc, prev_children: [pc_1, ..., pc_k] }`
  witnessing that e-node `idx` in `curr` class `c` also exists in `prev`.

Because discovering a parent match adds `pc` to `cover[c]` (which can unlock
matches higher up), `enumerate_matches` runs a bottom-up fixpoint over `curr`'s
budgeted classes while querying the **complete** `prev_lookup` (previous
classes are never filtered by a previous root):

```text
function enumerate_matches(curr, prev_lookup, budgets):
    cover   : Map<Id, Set<PrevId>> = empty
    matches : Map<(Id, usize), List<NodeMatch>> = empty

    repeat:
        changed = false
        for each current class c in budgets:
            for each e-node n at index idx in c:
                if not budgets.node_fits(curr, c, n):
                    continue
                [c_1, ..., c_k] = canonical curr children of n
                for each combo (pc_1, ..., pc_k) in cover[c_1] x ... x cover[c_k]:
                    // For leaves (k = 0), the product has one empty tuple [].
                    translated = n with children replaced by [pc_1, ..., pc_k]
                    if prev_lookup.lookup(translated) = Some(pc):
                        nm = NodeMatch { prev_class: prev_lookup.find(pc), prev_children: combo }
                        if nm not in matches[(c, idx)]:
                            matches[(c, idx)].push(nm)
                            cover[c].insert(nm.prev_class)
                            changed = true
    until not changed

    return matches
```

### Why rooted enumeration is complete

Take any non-novel term `t` of size at most `budget(c)` extractable from a
budgeted current class `c`:

- **Base case (leaf):** `t` has arity 0, so the empty Cartesian product queries
  the leaf directly in `prev_lookup` on the first pass.
- **Inductive step:** Every child subtree `t_i` is strictly smaller than `t`,
  is shared between `c_i` and some previous class `pc_i`, and fits `budget(c_i)`
  by the root-budget propagation rule. By induction on term size, `pc_i` is
  discovered in `cover[c_i]`. The Cartesian product then visits
  `(pc_1, ..., pc_k)` and discovers the parent match in `prev_lookup`.

Cycles in the e-graph do not break this induction because any finite extracted
tree strictly decreases in size from parent to child.

---

## 4. Exact incremental scan (`whole` and `joint`)

Within each `try_cap` attempt, `whole` and `joint` `LayerTable`s advance
together one size layer `s = 1, 2, ...` at a time:

```text
joint(c, pc, s) =
    sum over (n at idx in c, m in matches[(c, idx)] with m.prev_class = pc):
        sum over s_1 + ... + s_k = s - 1:
            product_i joint(c_i, m.prev_children[i], s_i)
```

For leaves (`k = 0`), a matching leaf contributes `1` at `s = 1`. Each pair
`(c, pc)` inherits `budget(c)` from its current class; missing child pairs
contribute zero terms. Because every child size is strictly less than `s`,
`step_joint` reuses `LayerTable` with no fixpoint—even on cyclic e-graphs.

After layer `s` finishes, the exact novel count at `root` is:

```text
novel(root, s) = whole(root, s) - sum_pc joint(root, pc, s)
```

All counts use arbitrary-precision `BigUint`, so nonzero detection and
cumulative sums are exact.

---

## 5. Final-size pruning and package assembly

Because the winning `final_max_size` can be smaller than `try_cap`, the
in-flight tables may contain classes, matches, and histogram entries that are
unreachable at `final_max_size`.

Before assembling the package, `NovelTermCount::build` computes
`final_budgets = min_sizes.budgets_for(curr, root, final_max_size)` and prunes
in place:

1. `prune_matches` drops entries whose current class is absent from
   `final_budgets` or whose e-node no longer satisfies `node_fits`.
2. `final_budgets.prune_histograms` removes out-of-budget classes and size
   entries `> budget(c)` from both `whole` and `joint`.
3. `NovelTermCount::assemble` derives:
   - the drawer-facing `cover[c]` from the keys of nonempty `joint` histograms;
   - `novel[c]` by pointwise subtraction (`whole(c, s) - sum_pc joint(c, pc, s)`),
     retaining only positive entries.

### Correctness invariant

For the selected `root` and final limit `M`, every lookup reachable while
drawing a novel term of size `<= M` is preserved:

- the exact `novel[root]` histogram through `M`;
- all recursive `novel[c]` and `joint[(c, pc)]` child-histogram lookups;
- all `cover[c]` entries and `NodeMatch` witnesses needed by feasible frontier
  states; and
- all child-size histogram entries needed to split child budgets on demand.

Only states unreachable from `(root, M)` are dropped.

---

## 6. Diagnostics and verification

### Diagnostics

`FrontierPackage::build` prints a diagnostic to `stderr` only on failure (when
`cap` is exhausted with fewer than `min_extractable` novel terms, or when
`novel[root]` is empty). Successful builds stay silent; call `log_root_counts`
explicitly when you want a sorted frontier-size histogram.

### Verification

Tests cover acyclic and cyclic graphs, empty novel frontiers, merged and
repeated children, parent matches discovered through child covers, unreachable
matching classes, oversized nodes removed by final pruning, and multiple
previous witness classes for one current class. The end-to-end backoff fixture
has novel sizes `5, 7, 9, ...`; requesting enough terms to span three sizes
selects and builds the package at `9`.
