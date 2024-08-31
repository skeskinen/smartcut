#!/bin/bash

pushd "$(dirname "$0")"

rm -rf dist

pyinstaller --distpath ./dist --workpath ./build --onefile -n smartcut smartcut/__main__.py

tar -czvf "dist/smartcut_linux.tar.gz" dist/smartcut

popd
