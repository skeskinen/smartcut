#!/bin/bash

pushd "$(dirname "$0")"

rm -rf dist

pyinstaller --distpath ./dist --workpath ./build --onefile -n smartcut_linux smartcut/__main__.py

popd
