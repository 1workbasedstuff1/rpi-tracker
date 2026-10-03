#!/usr/bin/env bash
# Resets the current git repo to match origin/main exactly,
# after two confirmation prompts.

set -e

read -r -p "Are you sure you're going to erase all current progress in this directory? (yes/no): " answer1
if [ "$answer1" != "yes" ]; then
    echo "Aborted."
    exit 1
fi

read -r -p "Are you sure? (yes/no): " answer2
if [ "$answer2" != "yes" ]; then
    echo "Aborted."
    exit 1
fi

git fetch origin
git reset --hard origin/main
git clean -fd

echo "Done. Directory now matches origin/main."
