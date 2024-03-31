#/bin/sh

DIR=$(dirname "$0")

$DIR/venv/bin/python3.10 $DIR/main.py 2> /dev/null
