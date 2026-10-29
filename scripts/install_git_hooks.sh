#!/usr/bin/env bash
# ==============================================================================
# Git Hooks Installer
# Configures local repository to use version-controlled .githooks directory.
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

chmod +x .githooks/* 2>/dev/null || true
git config core.hooksPath .githooks

echo "Configured git core.hooksPath to .githooks"
echo "Active hooks:"
ls -l .githooks/
echo "Local pre-push hook is active. Tests will run automatically before every git push."
