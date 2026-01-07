
#!/usr/bin/env bash
set -Eeuo pipefail

# Resolve project root relative to this script (safe even if you move the project)
APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON="${APP_ROOT}/.venv/bin/python"
ENTRYPOINT="${APP_ROOT}/weight/run_weight.py"   # or your actual entrypoint
ENV_FILE="${APP_ROOT}/config/myapp.env"
LOCK_FILE="${APP_ROOT}/run_weight.lock"
LOG_DIR="${APP_ROOT}/logs"
OUT_LOG="${LOG_DIR}/run_weight.out.log"
ERR_LOG="${LOG_DIR}/run_weight.err.log"


mkdir -p "${LOG_DIR}"
touch "${OUT_LOG}" "${ERR_LOG}"

export LC_ALL="C.UTF-8"
export PATH="/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${APP_ROOT}:${PYTHONPATH:-}"

if [[ -f "${ENV_FILE}" ]]; then
  source "${ENV_FILE}"
fi

if [[ ! -x "${PYTHON}" ]]; then
  echo "[ERROR] Python venv not found at ${PYTHON}" | tee -a "${ERR_LOG}"
  exit 127
fi


# --- Make the working directory the same as the ENTRYPOINT's directory
ENTRY_DIR="$(dirname "${ENTRYPOINT}")"
cd "${ENTRY_DIR}"

# Show current working dir and effective paths
echo "[DBG] PWD=$(pwd)"
echo "[DBG] APP_ROOT=${APP_ROOT}"
echo "[DBG] PYTHON=${PYTHON}"
echo "[DBG] ENTRYPOINT=${ENTRYPOINT}"

# Verify the entrypoint exists and is readable
if [[ ! -r "${ENTRYPOINT}" ]]; then
  echo "[ERROR] Entrypoint not found or unreadable: ${ENTRYPOINT}" | tee -a "${ERR_LOG}"
  ls -lah "$(dirname "${ENTRYPOINT}")" | tee -a "${ERR_LOG}" || true
  exit 2
fi



{
  flock -n 9 || { echo "[WARN] Another run is in progress, skipping." ; exit 0; }
  echo "[$(date -Is)] START myapp"
  exec "${PYTHON}" "${ENTRYPOINT}" --mode=batch
  echo "[$(date -Is)] END myapp (exit=$?)"
} 9>"${LOCK_FILE}" >>"${OUT_LOG}" 2>>"${ERR_LOG}"
