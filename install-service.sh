#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
unit_path="$unit_dir/g0v0-workflow-bot.service"

if [[ ! -x "$project_dir/.venv/bin/nb" ]]; then
    printf 'Executable not found: %s/.venv/bin/nb\n' "$project_dir" >&2
    exit 1
fi

escape_unit_value() {
    local value="$1"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    value="${value//%/%%}"
    printf '%s' "$value"
}
escaped_dir="$(escape_unit_value "$project_dir")"
mkdir -p "$unit_dir"
if [[ -L "$unit_path" ]]; then
    rm -- "$unit_path"
fi
cat > "$unit_path" <<UNIT
[Unit]
Description=g0v0 Workflow Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$project_dir
Environment="PATH=$escaped_dir/.venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart="$escaped_dir/.venv/bin/nb" run
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
UNIT
systemctl --user daemon-reload
systemctl --user enable g0v0-workflow-bot.service
systemctl --user restart g0v0-workflow-bot.service
printf 'Service installed and started: %s\n' "$unit_path"
