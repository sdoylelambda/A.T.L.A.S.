#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cat > "$DIR/Atlas.desktop" <<EOF
[Desktop Entry]
Name=Atlas
Exec=$DIR/run.sh
Icon=$DIR/_internal/icon.png
Type=Application
Terminal=false
Categories=Utility;
EOF
chmod +x "$DIR/Atlas.desktop"
echo "Atlas.desktop created — right-click it and choose 'Allow Launching'."