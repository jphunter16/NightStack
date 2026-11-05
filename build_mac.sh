#!/bin/bash
# Build NightStack.app (+ a .dmg) for Intel Macs.
#   Requirements: macOS 11+, Python 3.11 or 3.12 from python.org (or Homebrew), Xcode command line tools.
#   Usage:  ./build_mac.sh
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
echo "==> Using $($PY --version) at $(command -v $PY)  [machine: $(uname -m)]"
if [[ "$(uname -m)" == "arm64" ]]; then
  echo "   Apple Silicon detected: building an Intel app needs an x86_64 Python."
  echo "   Run:  arch -x86_64 /usr/local/bin/python3 -m venv ... or set PYTHON to an Intel python."
fi

$PY -m venv .venv-build
source .venv-build/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt "pyinstaller>=6.3"

echo "==> Making app icon"
mkdir -p build/NightStack.iconset
for s in 16 32 64 128 256 512; do
  sips -z $s $s nightstack/resources/icon.png --out build/NightStack.iconset/icon_${s}x${s}.png >/dev/null
  d=$((s*2)); sips -z $d $d nightstack/resources/icon.png --out build/NightStack.iconset/icon_${s}x${s}@2x.png >/dev/null
done
iconutil -c icns build/NightStack.iconset -o build/NightStack.icns

echo "==> Running self-test"
python tests/test_pipeline.py

echo "==> Building app"
pyinstaller --noconfirm --clean --distpath dist --workpath build/pyi packaging/NightStack.spec

echo "==> Ad-hoc code signing"
codesign --force --deep --sign - dist/NightStack.app

echo "==> Creating DMG"
rm -f dist/NightStack.dmg
mkdir -p build/dmg && rm -rf build/dmg/* && cp -R dist/NightStack.app build/dmg/ && ln -s /Applications build/dmg/Applications
hdiutil create -volname NightStack -srcfolder build/dmg -ov -format UDZO dist/NightStack.dmg >/dev/null

echo ""
echo "Done:  dist/NightStack.app   and   dist/NightStack.dmg"
echo "First launch: right-click NightStack.app > Open (it is not notarized by Apple)."
