#!/usr/bin/env bash
# Run this on the EC2 instance after git clone.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is missing. Install it, then rerun this script."
  exit 1
fi

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

if [[ ! -f .env && -f .env.example ]]; then
  cp .env.example .env
  echo "Created .env from .env.example. Set AWS_REGION if this instance is not in us-east-1."
fi

echo
echo "Setup finished."
echo "  source .venv/bin/activate"
echo "  python session.py --list-models"
echo "  python session.py"
