{
  description = "lifer: find birds in raw photos, identify candidate species with BioCLIP, feed a lifelist";

  inputs = {
    # pinned nixpkgs rev (matches a known-good system); keep in lockstep.
    nixpkgs.url = "github:NixOS/nixpkgs/c3f15ff4df3491351fa2e0c9aa0c2f1bcd290218";
    # release-26.05: stable HM branch; the module below uses its
    # modular-services API (home.services.<name>.systemd.services."").
    home-manager = {
      url = "github:nix-community/home-manager/32904dfe1cf86039f3c8c9f41f94e1cfd76dbb34";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, home-manager }:
    let
      systems = [ "x86_64-linux" ];

      perSystem = system:
        let
          lib = nixpkgs.lib;
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true; # torch-bin ships NVIDIA CUDA wheels
          };
          # The system's default python (3.14): its package set is what the
          # nixpkgs CI / local store has built, so deps hit the binary cache
          # and store instead of building from source.
          python = pkgs.python3;
          # Prebuilt CUDA wheels: swap torch/torchvision in the package-set
          # scope (not the python derivation, which would change every
          # module's hash and defeat the binary cache). Dependents
          # (open-clip-torch, ultralytics, ...) then build against
          # torch-bin/torchvision-bin instead of compiling torch.
          #
          # A few deps are not in the store and build from source; nixpkgs
          # runs their test suites (preDistPhases / installCheckPhase) by
          # default, so turn them off for those only.
          pythonPackages = python.pkgs.override {
            overrides = _final: super: {
              torch = super."torch-bin";
              torchvision = super."torchvision-bin";
              ultralytics = super.ultralytics.overridePythonAttrs {
                dontUsePytestCheck = true;
                doInstallCheck = false;
              };
              open-clip-torch = super."open-clip-torch".overridePythonAttrs {
                dontUsePytestCheck = true;
                doInstallCheck = false;
              };
              rawpy = super.rawpy.overridePythonAttrs {
                dontUsePytestCheck = true;
                doInstallCheck = false;
              };
              # transitive dep of open-clip-torch; not in the local store
              timm = super.timm.overridePythonAttrs {
                dontUsePytestCheck = true;
                doInstallCheck = false;
              };
            };
          };
          lifer = pythonPackages.buildPythonPackage {
            pname = "lifer";
            version = "0.1.0";
            src = ./.;
            format = "pyproject";
            # PEP 517 build with setuptools (see pyproject.toml [build-system])
            build-system = [ pythonPackages.setuptools ];
            # nixpkgs no longer parses pyproject deps; list them explicitly
            dependencies = [
              pythonPackages.torch
              pythonPackages.torchvision
              pythonPackages.ultralytics
              pythonPackages."open-clip-torch"
              pythonPackages.rawpy
              pythonPackages.odfpy
              pythonPackages.openpyxl
              pythonPackages.requests
              pythonPackages.rapidfuzz
              pythonPackages.numpy
              pythonPackages.pillow
              pythonPackages.flask
            ];
            # exiftool & darktable-cli on the wrapper's runtime PATH (the
            # module shells out to them; develop stays torch-free)
            makeWrapperArgs = [ "--prefix" "PATH" ":"
                                "${pkgs.lib.makeBinPath [ pkgs.exiftool pkgs.darktable ]}" ];
            doCheck = false;
            meta = {
              description = "Bird photo pipeline: detect, identify, review, develop";
              license = lib.licenses.mit;
              platforms = [ "x86_64-linux" ];
            };
          };
        in
        {
          packages = {
            inherit lifer;
            default = lifer;
          };
          defaultPackage = lifer;
          apps = {
            default = {
              type = "app";
              program = "${lifer}/bin/lifer";
              meta = {
                description = "lifer: find birds in raw photos, identify species with BioCLIP, feed a lifelist";
              };
            };
          };
          devShells = {
            default = pkgs.mkShell {
              packages = [
                pkgs.git
                pkgs.exiftool
                pkgs.python3
                lifer
              ];
              shellHook = ''
                # site-packages of the app and all its deps, so the shell
                # python can import lifer, torch, ... like the app does
                export PYTHONPATH="${lib.concatStringsSep ":" (lib.map (d: "${d}/lib/python3.14/site-packages") (lifer.propagatedBuildInputs ++ [ lifer ]))}${PYTHONPATH:+:$PYTHONPATH}"
              '';
            };
          };
          checks = {
            # building the package is itself a check
            inherit lifer;
            # the home-manager module evaluates and produces the unit
            home-manager-module =
              let
                hm = home-manager.lib.homeManagerConfiguration {
                  pkgs = pkgs;
                  modules = [
                    homeManagerModule
                    {
                      home.homeDirectory = "/home/user";
                      home.username = "user";
                      home.stateVersion = "26.05";
                      services.lifer.enable = true;
                      # make the flake's lifer package available to the
                      # module's pkgs (HM re-imports nixpkgs internally)
                      nixpkgs.overlays = [ (final: _prev: { inherit lifer; }) ];
                    }
                  ];
                };
                svc = hm.config.systemd.user.services."lifer";
              in
              pkgs.runCommand "lifer-home-manager-check" { } ''
                mkdir -p $out
                # the module evaluated and produced the user unit
                echo "ExecStart=${lib.lists.elemAt svc.Service.ExecStart 0}" > $out/service
                echo "Environment=${lib.concatStringsSep " " svc.Service.Environment}" >> $out/service
                grep -q "lifer serve" $out/service
                grep -q "LIFER_HOME" $out/service
              '';
          };
        };

      # A home-manager module: `services.lifer = { enable; port ? 8737;
      # home ? ...; }` runs `lifer serve` as a systemd user service with
      # LIFER_HOME set. Uses HM 26.05's modular-services API
      # (home.services.<name>.systemd.services."").
      homeManagerModule =
        { config, lib, pkgs, ... }:
        let
          cfg = config.services.lifer;
        in
        {
          options.services.lifer = {
            enable = lib.mkEnableOption "the lifer review app (systemd user service)";
            host = lib.mkOption {
              type = lib.types.str;
              default = "127.0.0.1";
              description = "Interface for `lifer serve` to bind (127.0.0.1 for local-only).";
            };
            port = lib.mkOption {
              type = lib.types.port;
              default = 8737;
              description = "Port for `lifer serve`.";
            };
            home = lib.mkOption {
              type = lib.types.path;
              default = "${config.home.homeDirectory}/.local/share/lifer";
              description = "LIFER_HOME state directory (runs/, cache/, data/).";
            };
          };

          config = lib.mkIf cfg.enable {
            home.services.lifer = {
              systemd.mainExecStart =
                "${pkgs.lifer}/bin/lifer serve --host ${cfg.host} --port ${toString cfg.port} --db ${cfg.home}/data/lifer.sqlite";
              systemd.services."".description = "lifer review app";
              systemd.services."".wantedBy = [ "default.target" ];
              # the lifer wrapper already carries exiftool on its PATH
              systemd.services."".serviceConfig.Environment = [ "LIFER_HOME=${cfg.home}" ];
            };
          };
        };
    in
    {
      packages = nixpkgs.lib.genAttrs systems (s: (perSystem s).packages);
      defaultPackage = nixpkgs.lib.genAttrs systems (s: (perSystem s).defaultPackage);
      apps = nixpkgs.lib.genAttrs systems (s: (perSystem s).apps);
      devShells = nixpkgs.lib.genAttrs systems (s: (perSystem s).devShells);
      checks = nixpkgs.lib.genAttrs systems (s: (perSystem s).checks);
      homeManagerModules = nixpkgs.lib.genAttrs systems (s: {
        inherit homeManagerModule;
        default = homeManagerModule;
      });
    };
}
