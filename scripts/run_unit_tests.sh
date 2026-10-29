#!/usr/bin/env bash
# ==============================================================================
# kn-rates Unit Test Suite Runner
# Runs all unit tests required for CI and pull request merge checks.
# ==============================================================================
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Prefer an accessible python3 executable
if [ -z "${PYTHON:-}" ]; then
    if python3 --version >/dev/null 2>&1; then
        PYTHON="python3"
    elif [ -x /opt/homebrew/bin/python3 ]; then
        PYTHON="/opt/homebrew/bin/python3"
    elif [ -x /usr/bin/python3 ]; then
        PYTHON="/usr/bin/python3"
    else
        PYTHON="python"
    fi
fi

echo "=================================================="
echo "Running kn-rates Unit Tests Suite"
echo "Python interpreter: $("$PYTHON" --version)"
echo "Working directory:  $REPO_ROOT"
echo "=================================================="

echo ""
echo "--> [1/2] Discovering tests in tests/ directory..."
"$PYTHON" -m unittest discover -s tests -v

echo ""
echo "--> [2/2] Running criteria grid and inclination discrepancy tests..."
"$PYTHON" -m unittest test_criteria_grid.py test_inclination_discrepancy.py

echo ""
echo "=================================================="
echo "SUCCESS: All unit tests passed!"
echo "=================================================="
