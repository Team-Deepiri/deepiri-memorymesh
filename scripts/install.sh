#!/usr/bin/env bash
# Install deepiri-memorymesh via curl:
#   curl -fsSL https://raw.githubusercontent.com/Team-Deepiri/deepiri-memorymesh/main/scripts/install.sh | bash
set -euo pipefail

REPO="Team-Deepiri/deepiri-memorymesh"
REPO_URL="https://github.com/${REPO}.git"
BRANCH="${DEEPIRI_MEMORYMESH_BRANCH:-main}"
KEEP_DIR="${DEEPIRI_MEMORYMESH_KEEP_DIR:-0}"
WITH_EMBEDDINGS="${DEEPIRI_MEMORYMESH_EMBEDDINGS:-1}"

usage() {
  cat <<'EOF'
Usage: install.sh [options]

Clone (when needed) and pip-install memorymesh with CLI linked to ~/.local/bin.

Options:
  -h, --help           Show this help
  --dry-run            Print actions without installing
  --no-embeddings      Skip [embeddings] extra (lighter install)

Environment:
  DEEPIRI_MEMORYMESH_SRC           Existing checkout
  DEEPIRI_MEMORYMESH_BRANCH        Git branch (default: main)
  DEEPIRI_MEMORYMESH_KEEP_DIR      Keep clone when set to 1
  DEEPIRI_MEMORYMESH_EMBEDDINGS    Set to 0 to skip embeddings extra

Requires: git, python3 (>=3.10)
Verify:   memorymesh --help
EOF
}

log() { printf '==> %s\n' "$*"; }

DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --no-embeddings) WITH_EMBEDDINGS=0; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

for cmd in git python3; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "error: $cmd is required." >&2; exit 1; }
done

ROOT=""
CLEANUP=""
LOCAL_BIN="${HOME}/.local/bin"

if [[ -n "${DEEPIRI_MEMORYMESH_SRC:-}" && -f "${DEEPIRI_MEMORYMESH_SRC}/pyproject.toml" ]]; then
  ROOT="${DEEPIRI_MEMORYMESH_SRC}"
elif [[ -n "${BASH_SOURCE[0]:-}" ]] && [[ "${BASH_SOURCE[0]}" != bash ]] && [[ -f "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/pyproject.toml" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
else
  ROOT="$(mktemp -d)"
  [[ "$KEEP_DIR" != "1" ]] && CLEANUP="$ROOT"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "Would clone ${REPO_URL} to ${ROOT}"
    log "Would pip install memorymesh and link memorymesh CLI"
    exit 0
  fi
  git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$ROOT"
fi

[[ "$DRY_RUN" -eq 1 ]] && { log "Would install from ${ROOT}"; exit 0; }

trap '[[ -n "$CLEANUP" ]] && rm -rf "$CLEANUP"' EXIT
cd "$ROOT"

VENV="${ROOT}/.venv"
log "Creating venv at ${VENV}"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -U pip wheel -q

EXTRAS=""
[[ "$WITH_EMBEDDINGS" == "1" ]] && EXTRAS="[embeddings]"
log "Installing memorymesh${EXTRAS}"
"$VENV/bin/pip" install -e ".${EXTRAS}" -q

mkdir -p "$LOCAL_BIN"
ln -sf "$VENV/bin/memorymesh" "$LOCAL_BIN/memorymesh"
export PATH="${LOCAL_BIN}:${PATH}"

memorymesh --help >/dev/null
echo ""
echo "Verify: memorymesh --help"
if [[ ":$PATH:" != *":${LOCAL_BIN}:"* ]]; then
  echo "Add to your shell profile: export PATH=\"${LOCAL_BIN}:\$PATH\""
fi
