#!/bin/bash

# Path of Python script
SCRIPT_PATH="/home/sn27/uptime_checker/uptime_checker.py"

su - sn27 <<'EOF'
SCRIPT_PATH="/home/sn27/uptime_checker/uptime_checker.py"
/usr/bin/env python3 "$SCRIPT_PATH"
EOF