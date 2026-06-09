export WANDB_API_KEY=""  # your wandb api key
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYNC_PATH="${SCRIPT_DIR}"
SYNC_INTERVAL=10

while true; do
    python -m wandb sync $SYNC_PATH/wandb/offline-run-* 2>/dev/null
    sleep $SYNC_INTERVAL
done
