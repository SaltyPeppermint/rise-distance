# Vendored crates

`egg/` is reserved for the project's patched egg fork. Populate it with:

```text
git clone <egg-fork-url> vendor/egg
```

Do not point the workspace's `egg` dependency at this path until the fork has
been cloned and its crate builds locally.
