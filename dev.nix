{ pkgs ? import <nixpkgs> {
    config = {
        allowUnfree = true;
        cudaSupport = true;
    };
} }:

let
    nvidiaPackage = pkgs.linuxPackages.nvidiaPackages.stable;
in
    pkgs.mkShell {
        packages = with pkgs; [
            gcc

            cudatoolkit
            cudaPackages.cuda_cudart
            nvidiaPackage
            texliveFull
        ];

        shellHook = ''
    export CUDA_PATH=${pkgs.cudatoolkit}
    export CUDAToolkit_ROOT=${pkgs.cudatoolkit}
    export LD_LIBRARY_PATH="${nvidiaPackage}/lib:$LD_LIBRARY_PATH"
        '';

    }
