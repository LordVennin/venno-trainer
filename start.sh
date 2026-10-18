#!/bin/bash
# Start Venno Trainer (installs Node dependencies on first run).
cd "$(dirname "$0")"
if [ ! -d node_modules ]; then
    echo "Installing dependencies..."
    npm install || exit 1
fi
node server.js "$@"
