#!/usr/bin/env bash
# Apply the next queued milestone to the working tree and stage it.
# Run by .github/workflows/drip.yml; the workflow then commits (authored as the
# repo owner, dated the real run day) and pushes. No dates are pre-stamped — a
# milestone's commit date is whenever the scheduled run actually applies it.
set -euo pipefail

rm -f .drip/_noop .drip/_commit_msg

next=$(sed -n 's/.*"next"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' .drip/state.json)
[ -n "${next:-}" ] || { echo "ERROR: could not read .drip/state.json"; exit 1; }

dir=$(printf 'queue/%02d' "$next")
if [ ! -d "$dir" ]; then
  echo "No milestone #$next queued ($dir absent) — nothing to do."
  : > .drip/_noop
  exit 0
fi

echo "Applying milestone #$next from $dir"
git apply --whitespace=nowarn "$dir/diff.patch"

printf '{ "next": %d }\n' "$((next + 1))" > .drip/state.json
git add -A
cp "$dir/msg.txt" .drip/_commit_msg
echo "Staged milestone #$next; next is now $((next + 1))."
