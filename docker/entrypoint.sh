#!/usr/bin/env bash
set -Eeuo pipefail

APP_ROOT="${APP_ROOT:-/app}"
DATA_DIR="${DATA_DIR:-${APP_ROOT}/data}"
PYTHON_BIN="${PYTHON_BIN:-python}"
KAGGLE_COMPETITION="${KAGGLE_COMPETITION:-rsna-knee-abnormality-detection}"
TRAIN_MODE="${TRAIN_MODE:-train}"

log() { printf '[initializer] %s\n' "$*"; }
fail() { printf '[initializer] ERROR: %s\n' "$*" >&2; exit 2; }

has_dicom() {
  local directory="$1"
  [[ -d "$directory" && -n "$(find "$directory" -type f -iname '*.dcm' -print -quit 2>/dev/null)" ]]
}

copy_metadata() {
  local staging="$1" name="$2" source
  [[ -f "${DATA_DIR}/${name}" ]] && return 0
  source="$(find "$staging" -type f -name "$name" -print -quit)"
  [[ -n "$source" ]] || fail "download did not contain ${name}"
  cp "$source" "${DATA_DIR}/${name}"
}

normalize_local_metadata() {
  local canonical alias
  while IFS='|' read -r canonical alias; do
    if [[ ! -f "${DATA_DIR}/${canonical}" && -f "${DATA_DIR}/${alias}" ]]; then
      log "promoting ${alias} to canonical ${canonical}"
      cp "${DATA_DIR}/${alias}" "${DATA_DIR}/${canonical}"
    fi
  done <<'EOF'
train.csv|train (1).csv
test.csv|test (1).csv
train_series.csv|train_series (1).csv
test_series.csv|test_series (1).csv
sample_submission.csv|sample_submission (2).csv
EOF
}

copy_dicom_split() {
  local staging split destination source
  staging="$1"
  split="$2"
  destination="${DATA_DIR}/${split}_series"
  source=""
  has_dicom "$destination" && return 0
  for candidate in "${split}_series" "${split}_images" "$split"; do
    source="$(find "$staging" -type d -name "$candidate" -print -quit)"
    if [[ -n "$source" ]] && has_dicom "$source"; then
      mkdir -p "$destination"
      cp -a "${source}/." "$destination/"
      return 0
    fi
  done
  fail "download did not contain a DICOM tree for ${split}"
}

download_data() {
  if [[ -z "${KAGGLE_API_TOKEN:-}" && ( -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ) ]]; then
    fail "DICOM data is missing; set KAGGLE_API_TOKEN in .env (and accept the competition rules first)"
  fi
  local temporary download staging archive
  temporary="$(mktemp -d)"
  trap "rm -rf '$temporary'" EXIT
  download="${temporary}/download"
  staging="${temporary}/staging"
  mkdir -p "$download" "$staging" "$DATA_DIR"
  log "downloading Kaggle competition ${KAGGLE_COMPETITION}"
  if ! kaggle competitions download -c "$KAGGLE_COMPETITION" -p "$download"; then
    fail "Kaggle download failed; verify the token and accept the competition rules in your browser"
  fi
  while IFS= read -r archive; do
    unzip -q "$archive" -d "$staging"
  done < <(find "$download" -type f -name '*.zip' -print)
  find "$download" -maxdepth 1 -type f ! -name '*.zip' -exec cp {} "$staging/" \;
  copy_metadata "$staging" train.csv
  copy_metadata "$staging" test.csv
  copy_metadata "$staging" train_series.csv
  copy_metadata "$staging" test_series.csv
  copy_metadata "$staging" sample_submission.csv
  copy_dicom_split "$staging" train
  copy_dicom_split "$staging" test
}

validate_inputs() {
  local name fold
  for name in train.csv test.csv train_series.csv test_series.csv sample_submission.csv; do
    [[ -s "${DATA_DIR}/${name}" ]] || fail "required metadata is missing: ${DATA_DIR}/${name}"
  done
  has_dicom "${DATA_DIR}/train_series" || fail "train DICOM files are missing"
  has_dicom "${DATA_DIR}/test_series" || fail "test DICOM files are missing"
  for fold in 0 1 2 3 4; do
    [[ -s "${APP_ROOT}/weights/checkpoints/knee/m_f${fold}.pt" ]] || fail "missing DINO checkpoint m_f${fold}.pt"
  done
  [[ -e "${APP_ROOT}/weights/checkpoints/sam/submissions_epoch_8_step_11550" ]] || fail "missing SAM checkpoint"
}

cd "$APP_ROOT"
normalize_local_metadata
if has_dicom "${DATA_DIR}/train_series" && has_dicom "${DATA_DIR}/test_series"; then
  log "existing train and test DICOMs found; skipping Kaggle download"
else
  download_data
fi
validate_inputs
command -v nvidia-smi >/dev/null || fail "nvidia-smi is unavailable; start the container with NVIDIA GPU support"
nvidia-smi
"$PYTHON_BIN" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else "PyTorch cannot access CUDA")'

case "$TRAIN_MODE" in
  train) exec "$PYTHON_BIN" TrainEnsemble.py auto --config config/training.yaml ;;
  plan) exec "$PYTHON_BIN" TrainEnsemble.py auto --plan-only --config config/training.yaml ;;
  *) fail "TRAIN_MODE must be train or plan" ;;
esac
