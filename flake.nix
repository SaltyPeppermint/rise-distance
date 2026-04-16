{
  description = "rise-distance dev shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [
            pkg-config
            lldb
            clang
            lld
          ];

          buildInputs = with pkgs; [
            fontconfig
            freetype
            expat
            # uncomment as needed for GUI/graphics crates:
            # xorg.libX11 xorg.libXcursor xorg.libXi xorg.libXrandr
            # libxkbcommon wayland vulkan-loader libGL
          ];

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
            fontconfig freetype expat vulkan-loader libGL
          ]);
        };
      });
}