{
  description = "rise-distance dev shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, fenix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # toolchain = fenix.packages.${system}.stable.toolchain;

        # toolchain = fenix.packages.${system}.latest.toolchain;

        # toolchain = fenix.packages.${system}.fromToolchainFile {
        #   file = ./rust-toolchain.toml;
        #   sha256 = pkgs.lib.fakeSha256; # replace with real hash after first build
        # };

        # Stable with custom components
        toolchain = fenix.packages.${system}.stable.withComponents [
          "cargo" "rustc" "rust-src" "rustfmt" "clippy" "rust-analyzer"
        ];
      in {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = [
            toolchain
          ] ++ (with pkgs; [
            pkg-config
            lldb
            clang
            lld
            uv
          ]);

          buildInputs = with pkgs; [
            fontconfig
            freetype
            expat
          ];

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
            fontconfig freetype expat vulkan-loader libGL
          ]);
        };
      });
}