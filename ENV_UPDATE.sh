#!/bin/bash
set -e  

if [ ! -d ".git" ]; then
    echo "Run this script from the root of a git repository."
    exit 1
fi

BACKUP_DIR=".models_local_backup"

echo "[*] Backing up local models/..."
rm -rf "$BACKUP_DIR"
if [ -d "models" ]; then
    cp -r models "$BACKUP_DIR"
fi

git fetch

git reset --hard origin/main

rm -rf models
if [ -d "$BACKUP_DIR" ]; then
    mv "$BACKUP_DIR" models
fi

echo "[*] ENV updated!"
