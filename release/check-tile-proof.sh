#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
# Release smoke for the FlashInfer 0.7.2.post1 sm_86 FP8 32-row verify tile.
# Patch: release/patches/flashinfer-0.7.2.post1/0001-sm8x-fp8-smallq-verify-tile-072.patch.
# JIT suffix: _sm8x_smallq_072_v1. The proof-line protocol is unchanged by the 0.7.0 and 0.7.2 ports.
# Run release/apply-flashinfer-patch.sh <venv> to verify the installed version and patched hashes;
# this log check alone does not attest the FlashInfer version or JIT artifact identity.
#
#   release/check-tile-proof.sh <server-log> [q_len] --tp N      (q_len 4 or 5; default 5 = MTP K=4; N = tensor-parallel size)
#
# Exit 0 if the CURRENT boot's part of the log (everything after the last "Initializing a V1 LLM engine" line) holds, as complete
# lines (the whole line, nothing before or after it),
#   [flashinfer-sm8x-tile] smallq=cta32 q_len=<q_len> rows=<6*q_len> cta_tile_q=32 device=<d>
# exactly once for every d in 0..N-1 and for no other d. Non-zero otherwise: 1 = proof missing, a device missing, repeated or out
# of range, or no engine-start line in the log (an un-delimited log could hold an earlier boot's lines); 2 = usage / unreadable log.
#
# d is the CUDA device the planning rank was on (cudaGetDevice), i.e. relative to CUDA_VISIBLE_DEVICES: vLLM puts TP rank r on
# cuda:r of its visible devices, so the ranks of a TP=N server report 0..N-1 whatever physical cards GPUS selects. The FlashInfer
# planner prints the line once per (mode, q_len) per loaded module (once per rank process) the first time a PLAN takes the tile:
# it is planning-time evidence that every rank chose the tile, not proof that the kernel then ran. Run this AFTER the first
# generation request has finished (speculative decoding on), never on a log that only shows startup.
set -euo pipefail

USAGE="usage: check-tile-proof.sh <server-log> [q_len] --tp N"
LOG=""; QLEN=""; TP=""
while [ $# -gt 0 ]; do
  case "$1" in
    --tp) [ $# -ge 2 ] || { echo "check-tile-proof: FAIL: --tp needs a value ($USAGE)" >&2; exit 2; }; TP="$2"; shift 2 ;;
    --tp=*) TP="${1#--tp=}"; shift ;;
    -*) echo "check-tile-proof: FAIL: unknown option $1 ($USAGE)" >&2; exit 2 ;;
    *) if [ -z "$LOG" ]; then LOG="$1"; elif [ -z "$QLEN" ]; then QLEN="$1"; else echo "check-tile-proof: FAIL: too many arguments ($USAGE)" >&2; exit 2; fi; shift ;;
  esac
done
[ -n "$LOG" ] || { echo "check-tile-proof: FAIL: $USAGE" >&2; exit 2; }
QLEN="${QLEN:-5}"
case "$QLEN" in
  4|5) ;;
  *) echo "check-tile-proof: FAIL: q_len must be 4 or 5, got $QLEN" >&2; exit 2 ;;
esac
case "$TP" in
  "") echo "check-tile-proof: FAIL: --tp N is required: the number of ranks that must each show the proof line ($USAGE)" >&2; exit 2 ;;
  [1-9] | [1-9][0-9]) ;;
  *) echo "check-tile-proof: FAIL: --tp must be a positive integer (at most 99), got '$TP'" >&2; exit 2 ;;
esac
[ -r "$LOG" ] || { echo "check-tile-proof: FAIL: cannot read $LOG" >&2; exit 2; }

TAG="[flashinfer-sm8x-tile]"
STEM="$TAG smallq=cta32 q_len=$QLEN rows=$((6 * QLEN)) cta_tile_q=32"
# Whole-line pattern (ERE, anchored); the device is a decimal without leading zeros.
RE="^\[flashinfer-sm8x-tile\] smallq=cta32 q_len=$QLEN rows=$((6 * QLEN)) cta_tile_q=32 device=(0|[1-9][0-9]*)$"
MARK="Initializing a V1 LLM engine"
START="$(grep -nF -- "$MARK" "$LOG" | tail -1 | cut -d: -f1 || true)"
if [ -z "$START" ]; then
  echo "check-tile-proof: FAIL: no '$MARK' line in $LOG: cannot delimit the current boot (pass the server log from its start)" >&2
  exit 1
fi
BOOT="$(tail -n +"$((START + 1))" "$LOG")"

# device id -> count, from complete proof lines of the current boot only
DEVS="$(printf '%s\n' "$BOOT" | grep -E -- "$RE" | sed -E 's/.* device=([0-9]+)$/\1/' || true)"
fail=0
for ((d = 0; d < TP; d++)); do
  n="$(printf '%s\n' "$DEVS" | grep -cx -- "$d" || true)"
  if [ "$n" != 1 ]; then
    echo "check-tile-proof: FAIL: device=$d x$n in the current boot (want exactly 1)" >&2
    fail=1
  fi
done
EXTRA="$(printf '%s\n' "$DEVS" | grep -Ev "^$|^([0-9]|[1-9][0-9])$" || true)"
for d in $(printf '%s\n' "$DEVS" | grep -E '^[0-9]+$' | sort -un); do
  if [ "$d" -ge "$TP" ]; then EXTRA="$EXTRA $d"; fi
done
if [ -n "${EXTRA// /}" ]; then
  echo "check-tile-proof: FAIL: proof line from device(s)$EXTRA, outside 0..$((TP - 1)) (is --tp $TP the server's TP size?)" >&2
  fail=1
fi
if [ "$fail" = 0 ]; then
  echo "check-tile-proof: OK: '$STEM device=<d>' once each for device 0..$((TP - 1)) in the current boot of $LOG"
  exit 0
fi
if printf '%s\n' "$BOOT" | grep -qF -- "$TAG"; then
  echo "check-tile-proof: [flashinfer-sm8x-tile] lines in the current boot:" >&2
  printf '%s\n' "$BOOT" | grep -F -- "$TAG" | sort | uniq -c >&2
  if printf '%s\n' "$BOOT" | grep -qxF -- "$STEM"; then
    echo "check-tile-proof: proof lines without device=: the venv carries an older revision of the patch" \
         "(re-install flashinfer-python and run release/apply-flashinfer-patch.sh)" >&2
  fi
else
  echo "check-tile-proof: no [flashinfer-sm8x-tile] line in the current boot: is FLASHINFER_SM8X_FP8_SMALLQ=cta32 set on" \
       "every rank, the patch applied (release/apply-flashinfer-patch.sh), and has a request with MTP" \
       "verify run yet?" >&2
fi
exit 1
