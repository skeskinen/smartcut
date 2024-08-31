#!/bin/bash

pushd "$(dirname "$0")"

rm -rf dist

pyinstaller --distpath ./dist --workpath ./build --onefile -n smartcut smc/main_smartcut.py

popd