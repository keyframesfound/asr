#!/usr/bin/env bash
# One-command installer for asr — Audio Live Transcription.
#
# Fresh Mac (nothing pre-installed — no Homebrew, no Xcode tools needed):
#   curl -fsSL https://raw.githubusercontent.com/keyframesfound/asr/main/install.sh | bash
#
# From an existing clone:
#   ./install.sh
#
# Rerunning it updates an existing install in place; .env, config.json and
# models/ are kept.
#
# Optional environment overrides:
#   ASR_DIR=~/asr            install location        (default ~/asr)
#   ASR_BRANCH=main          branch/tag to fetch     (default main)
#   ASR_PYTHON=3.12          Python version          (default 3.12)
#   ASR_BIN_DIR=/usr/local/bin  where the `asr` launcher goes (default: auto)
#   ASR_WEIGHTS_SRC=~/Documents/VS Code/asr/models  copy local model weights into the
#                    install (SenseVoice/Whisper only load from local files;
#                    already-present weights are not re-copied)
set -euo pipefail

REPO_SLUG="${ASR_REPO:-keyframesfound/asr}"
BRANCH="${ASR_BRANCH:-main}"
TARGET="${ASR_DIR:-$HOME/asr}"
PYVER="${ASR_PYTHON:-3.12}"

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Platform checks
# ---------------------------------------------------------------------------
[[ "$(uname -s)" == "Darwin" ]] || warn "This installer targets macOS; continuing anyway (Linux may need system PortAudio)."
ARCH="$(uname -m)"
SKIP_PARAKEET=0
if [[ "$ARCH" != "arm64" ]]; then
  warn "Apple Silicon (arm64) not detected ($ARCH): the Parakeet engine needs MLX and will be skipped."
  SKIP_PARAKEET=1
fi

# ---------------------------------------------------------------------------
# 2. uv (brings its own Python — nothing else to install)
# ---------------------------------------------------------------------------
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv (Python bootstrapper)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  command -v uv >/dev/null 2>&1 || die "uv installation failed; install it manually from https://docs.astral.sh/uv/"
fi

# ---------------------------------------------------------------------------
# 3. Source code: local checkout if run from one, otherwise GitHub tarball
# ---------------------------------------------------------------------------
SRC=""
if [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]:-}" ]]; then
  candidate="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  [[ -f "$candidate/main.py" ]] && SRC="$candidate"
fi
TMP=""
if [[ -z "$SRC" ]]; then
  log "Downloading asr ($BRANCH) from GitHub"
  TMP="$(mktemp -d)"
  curl -fsSL "https://codeload.github.com/$REPO_SLUG/tar.gz/refs/heads/$BRANCH" | tar -xz -C "$TMP"
  SRC="$(find "$TMP" -maxdepth 1 -type d -name "${REPO_SLUG#*/}-*" | head -n1)"
  [[ -n "$SRC" ]] || die "Could not unpack the $REPO_SLUG tarball."
fi

# ---------------------------------------------------------------------------
# 4. Copy code into the install dir (fresh install or in-place update)
# ---------------------------------------------------------------------------
if [[ -f "$TARGET/main.py" ]]; then
  log "Updating existing install at $TARGET"
elif [[ -e "$TARGET" ]]; then
  die "$TARGET already exists but is not an asr install. Set ASR_DIR to a fresh location."
else
  log "Installing asr to $TARGET"
  mkdir -p "$TARGET"
fi
[[ "$SRC" != "$TARGET" ]] || die "Source and install location are the same directory."

# Keep user state across updates: venv, secrets, settings, model weights.
# config.json is skipped only when it already exists (updates); a fresh
# install ships the repo's default config.
for item in "$SRC"/* "$SRC"/.env.example "$SRC"/.gitignore; do
  [[ -e "$item" ]] || continue
  name="$(basename "$item")"
  case "$name" in
    .venv|.git|.env|models|__pycache__|.DS_Store) continue ;;
    config.json) [[ -f "$TARGET/config.json" ]] && continue ;;
  esac
  cp -R "$item" "$TARGET/"
done
[[ -f "$TARGET/.env.example" ]] || cp "$SRC/.env.example" "$TARGET/" 2>/dev/null || true
[[ -z "$TMP" ]] || rm -rf "$TMP"

# ---------------------------------------------------------------------------
# 4b. Optional: provision local model weights (SenseVoice / Whisper)
# ---------------------------------------------------------------------------
if [[ -n "${ASR_WEIGHTS_SRC:-}" ]]; then
  [[ -d "$ASR_WEIGHTS_SRC" ]] || die "ASR_WEIGHTS_SRC=$ASR_WEIGHTS_SRC is not a directory."
  log "Provisioning model weights from $ASR_WEIGHTS_SRC"
  mkdir -p "$TARGET/models"
  for item in "$ASR_WEIGHTS_SRC"/*; do
    [[ -e "$item" ]] || continue
    name="$(basename "$item")"
    [[ "$name" == ".DS_Store" ]] && continue
    if [[ -e "$TARGET/models/$name" ]]; then
      log "  $name (already present, skipping)"
    else
      log "  $name"
      cp -R "$item" "$TARGET/models/"
    fi
  done
fi

# ---------------------------------------------------------------------------
# 5. Virtualenv + dependencies
# ---------------------------------------------------------------------------
cd "$TARGET"
if [[ ! -x .venv/bin/python ]]; then
  log "Creating Python $PYVER virtualenv (uv downloads Python itself if missing)"
  uv venv .venv --python "$PYVER"
fi
REQ="requirements.txt"
if [[ "$SKIP_PARAKEET" == 1 ]]; then
  REQ="$(mktemp)"
  grep -v '^parakeet-mlx' requirements.txt > "$REQ" || true
fi
log "Installing Python dependencies (torch, MLX, FunASR — the long step)"
uv pip install --python .venv/bin/python -r "$REQ"

# ---------------------------------------------------------------------------
# 6. Seed .env (iFlytek cloud keys — optional, local engines work without)
# ---------------------------------------------------------------------------
[[ -f .env ]] || cp .env.example .env

# ---------------------------------------------------------------------------
# 6b. Pre-download model weights so the app is ready out of the box
#     (skip with ASR_NO_MODELS=1; the in-app manager can always fetch later)
# ---------------------------------------------------------------------------
download_model() {
  code="$1"
  for attempt in 1 2 3; do
    set +e
    ./.venv/bin/python -m engines.weights_worker "$code" \
      | awk -v c="$code" '/^progress/ { printf "\r  %s: %.0f MB   ", c, $2/1048576; fflush() } END { print "" }'
    rc=$?
    set -e
    if [[ $rc == 0 ]]; then return 0; fi
    if [[ $rc != 42 ]]; then return "$rc"; fi
    printf '  %s: stalled, retrying (attempt %d/3)\n' "$code" "$((attempt + 1))"
  done
  return 42
}

if [[ "${ASR_NO_MODELS:-0}" != 1 ]]; then
  log "Pre-downloading model weights (Parakeet ~2.4 GB, SenseVoice ~1 GB, Whisper ~1.6 GB)"
  log "  skip anytime with ASR_NO_MODELS=1 — the app can also fetch them (press i)"
  MODELS_FAILED=0
  for code in parakeet sensevoice whisper; do
    present=$(./.venv/bin/python -c "from engines import weights as w; print(int(w.weights_present('$code')))")
    if [[ "$present" == 1 ]]; then
      echo "  $code: already present"
      continue
    fi
    if ! download_model "$code"; then
      warn "$code failed to download — open asr and press i on it to retry."
      MODELS_FAILED=1
    fi
  done
  unset -f download_model
fi

# ---------------------------------------------------------------------------
# 7. Global `asr` launcher
# ---------------------------------------------------------------------------
if [[ -n "${ASR_BIN_DIR:-}" ]]; then
  BIN="$ASR_BIN_DIR"
else
  if [[ -d /opt/homebrew/bin && -w /opt/homebrew/bin ]]; then
    BIN="/opt/homebrew/bin"
  elif [[ -d /usr/local/bin && -w /usr/local/bin ]]; then
    BIN="/usr/local/bin"
  else
    BIN="$HOME/.local/bin"
  fi
fi
mkdir -p "$BIN"
printf '#!/usr/bin/env bash\nexec %q/run "$@"\n' "$TARGET" > "$BIN/asr"
chmod +x "$BIN/asr"

PATH_NOTE=""
if [[ ":$PATH:" != *":$BIN:"* ]]; then
  mkdir -p "$HOME"
  if ! grep -qs '# asr installer' "$HOME/.zprofile" 2>/dev/null; then
    printf '\n# asr installer: asr command\nexport PATH="%s:$PATH"\n' "$BIN" >> "$HOME/.zprofile"
  fi
  PATH_NOTE="Open a new Terminal window so the updated PATH takes effect."
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
cat <<EOF

  asr is installed.

    run:    asr            (in a new shell if the command is not found)
    app:    $TARGET
    keys:   $TARGET/.env   (iFlytek cloud keys optional; local engines work without)

  Notes:
  - First launch downloads Parakeet weights (~1 GB) automatically.
  - macOS will ask for microphone access on first listen — allow it for your terminal.
$([[ -n "$PATH_NOTE" ]] && printf '  - %s\n' "$PATH_NOTE")
EOF
