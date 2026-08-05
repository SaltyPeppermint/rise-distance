# rise-distance

## Setup

This project uses a patched version of [`egg`](https://github.com/SaltyPeppermint/egg)
as a Git submodule in `vendor/egg`.

When cloning the project, initialize the submodule at the same time:

```bash
git clone --recurse-submodules <repository-url>
cd rise-distance
```

If the project has already been cloned, initialize the submodule with:

```bash
git submodule update --init --recursive
```

Run the same update command after pulling changes that move the submodule to a
different commit.
