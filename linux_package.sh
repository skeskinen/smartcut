#!/bin/bash

pushd "$(dirname "$0")"

rm -rf build dist

pyinstaller --distpath ./dist --workpath ./build --onefile -n smartcut smartcut/__main__.py

tar -czvf "dist/smartcut_linux.tar" -C dist smartcut
./dist/smartcut --version
popd
