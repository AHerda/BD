{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "aarch64-linux";
      pkgs = nixpkgs.legacyPackages."${system}";
    in {
      devShells.${system}.default = pkgs.mkShell {
        shellHook = ''
          echo "Entering python dev shell"
          nu
        '';
        packages = with pkgs; [
          (python313.withPackages(p: [
            p.matplotlib
            p.numpy
            p.seaborn
            p.wordcloud
            p.mpmath
            p.pandas
          ]))
          gcc
        ];
      };
    };
}
