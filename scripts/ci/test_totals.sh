#!/usr/bin/env bash
# Sums the "test result:" lines of a cargo test log and reports the totals
# as a GitHub Actions notice, so a run shows how many tests it ran without
# opening its log.
#
#     scripts/ci/test_totals.sh LOG

set -euo pipefail

log="${1:?usage: test_totals.sh LOG}"

awk '
    /^test result:/ {
        binaries++
        for (i = 1; i <= NF; i++) {
            if ($(i + 1) ~ /^passed/) passed += $i
            if ($(i + 1) ~ /^failed/) failed += $i
            if ($(i + 1) ~ /^ignored/) ignored += $i
        }
    }
    END {
        printf "::notice title=Tests::%d passed, %d failed, %d ignored in %d test binaries\n", passed, failed, ignored, binaries
    }
' "$log"
