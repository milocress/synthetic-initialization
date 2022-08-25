let
  mach-nix = import (builtins.fetchGit {
    url = "https://github.com/DavHau/mach-nix";
    ref = "refs/tags/3.5.0";
  }) {
    pypiDataRev = "f014c49221a49dae6f4c5f72b51b8b8486dbf326";
    pypiDataSha256 = "sha256:1fslbylgawk00hnvrl4lysn1wb67pr3ik8g6lla0afq7krgpgvcv";
  };
in

mach-nix.mkPythonShell rec {
  requirements = builtins.readFile ./requirements.txt;
  providers = {
    # allow wheels only for torch
    torch = "wheel";
  };
}
