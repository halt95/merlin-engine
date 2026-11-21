# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Tokenizer helper process for the parser self-test (vllm/merlin/parser_selftest.py).

Run as a script by the self-test, never imported: ``python -I -S parser_selftest_child.py <tokenizers dir>``.
``-I -S`` means no environment variables, no user site and no ``site`` module, so no ``.pth`` file and no
``sitecustomize`` runs; the one directory the server resolved for the ``tokenizers`` package is put on
``sys.path`` in front of the standard library and nothing else is. It imports only the standard library
and ``tokenizers`` (no vllm, no torch, no CUDA), loads the serialised tokenizer the
server hands it, and answers encode and decode requests over stdin and stdout, so the native tokenizer
lives in a process the server can kill and none of its memory is in the serving process.

Framing, both directions: a 4-byte big-endian length, then that many bytes. The first message from the
server is the serialised tokenizer (JSON text); the reply is ``ok``. Every later request is a JSON object,
``{"op": "encode", "text": ..., "special": bool}``, ``{"op": "decode", "ids": [...], "skip": bool}`` or
``{"op": "id_to_token", "ids": [...]}``, answered with ``{"ids": [...]}``, ``{"text": ...}``,
``{"tokens": [...]}`` or ``{"error": ...}``. End of input ends the process.
"""

import json
import struct
import sys


def _read(stream):
    head = stream.read(4)
    if len(head) < 4:
        return None
    (size,) = struct.unpack(">I", head)
    data = stream.read(size)
    return data if len(data) == size else None


def _write(stream, payload: bytes) -> None:
    stream.write(struct.pack(">I", len(payload)) + payload)
    stream.flush()


def main() -> int:
    # With -I -S the path is the standard library alone; add the directory the server found ``tokenizers`` in.
    sys.path[:] = sys.argv[1:2] + sys.path
    from tokenizers import Tokenizer

    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    serialised = _read(stdin)
    if serialised is None:
        return 1
    tokenizer = Tokenizer.from_str(serialised.decode("utf-8"))
    _write(stdout, b"ok")
    while True:
        raw = _read(stdin)
        if raw is None:
            return 0
        try:
            request = json.loads(raw.decode("utf-8"))
            if request["op"] == "encode":
                reply = {"ids": tokenizer.encode(request["text"], add_special_tokens=request["special"]).ids}
            elif request["op"] == "decode":
                reply = {"text": tokenizer.decode(request["ids"], skip_special_tokens=request["skip"])}
            elif request["op"] == "id_to_token":
                reply = {"tokens": [tokenizer.id_to_token(i) for i in request["ids"]]}
            else:
                reply = {"error": "unknown op"}
        except Exception as e:  # noqa: BLE001 - report to the server, keep serving requests
            reply = {"error": f"{type(e).__name__}: {e}"}
        _write(stdout, json.dumps(reply).encode("utf-8"))


if __name__ == "__main__":
    sys.exit(main())
