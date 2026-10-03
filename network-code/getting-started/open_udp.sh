#!/usr/bin/env bash
# open.sh <port>
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <port>" >&2
  exit 1
fi

PORT="$1"

if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "Invalid port: $PORT" >&2
  exit 1
fi

RULE=(-p udp --dport "$PORT" -j nixos-fw-accept)

# Don't add a duplicate if it's already open
if sudo iptables -C nixos-fw "${RULE[@]}" 2>/dev/null; then
  echo "UDP port $PORT is already open."
  exit 0
fi

sudo iptables -I nixos-fw "${RULE[@]}"
echo "UDP port $PORT opened."
