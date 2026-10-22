"""Shell completion scripts for the lifer CLI.

Generated from the argparse parser with shtab when the package is built
(flake.nix runs ``python -m lifer.completion bash|zsh``), so they track the
CLI by construction. Paths complete as files/dirs; trip IDs complete from the
run folders under LIFER_HOME at completion time, so new trips show up
without a rebuild.
"""

from __future__ import annotations

import sys

import shtab

from .__main__ import build_parser

PREAMBLE = {
    "bash": r'''
# trip IDs: the run folders under LIFER_HOME (matching the word being completed)
_lifer_trips() {
  local runs="${LIFER_HOME:-$HOME/.local/share/lifer}/runs" d
  for d in "$runs"/*/; do
    [[ -d $d ]] || continue
    d=${d%/}; d=${d##*/}
    [[ $d == "$1"* ]] && printf '%s\n' "$d"
  done
}
_lifer_trips_or_dirs() {
  _lifer_trips "$1"
  _shtab_compgen_dirs "$1"
}
''',
    "zsh": r'''
_lifer_trips() {
  local -a trips
  trips=( ${LIFER_HOME:-$HOME/.local/share/lifer}/runs/*(N/:t) )
  _describe -t trips 'trip' trips
}
_lifer_trips_or_dirs() {
  _alternative 'trips:trip:_lifer_trips' 'dirs:folder:_files -/'
}
''',
}


def main() -> None:
    shell = sys.argv[1] if len(sys.argv) > 1 else "bash"
    print(shtab.complete(build_parser(), shell=shell, preamble=PREAMBLE))


if __name__ == "__main__":
    main()
