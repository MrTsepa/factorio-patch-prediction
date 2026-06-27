#!/usr/bin/env bash
# Start FBSR's bot-run RPC service and keep it alive. `scripts/fbsr.sh bot-render`
# and scripts/render_eval.py connect to this service to render blueprints fast
# (one warm JVM with the sprite atlas loaded, vs. cold-starting per render).
#
#   scripts/fbsr_service.sh &        # leave running in the background
#   # then render: scripts/fbsr.sh bot-render "0eN..." -o=out.png -full
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
# Feed the interactive shell `bot-run`, then hold stdin open so the service stays up.
( echo 'bot-run vanilla -r'; tail -f /dev/null ) | "$DIR/fbsr.sh"
