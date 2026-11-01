#!/usr/bin/env python3
"""Print GGUF tokenizer entries by token ID."""

import argparse
from pathlib import Path

from gguf import GGUFReader, GGUFValueType


def main():
    parser = argparse.ArgumentParser(
        description="Print token IDs, spellings, and token types from a GGUF."
    )
    parser.add_argument("gguf", type=Path, help="Path to input GGUF")
    parser.add_argument("--start", type=int, default=0, help="First token ID")
    parser.add_argument("--end", type=int, help="Last token ID (inclusive)")
    args = parser.parse_args()

    reader = GGUFReader(str(args.gguf))

    fields = reader.fields
    token_field = fields.get("tokenizer.ggml.tokens")
    type_field = fields.get("tokenizer.ggml.token_type")

    if token_field is None:
        raise SystemExit("Error: tokenizer.ggml.tokens not found")

    tokens = token_field.contents()
    token_types = type_field.contents() if type_field is not None else None

    end = args.end if args.end is not None else len(tokens) - 1
    if args.start < 0 or end >= len(tokens) or args.start > end:
        raise SystemExit(f"Invalid token range {args.start}..{end}")

    for token_id in range(args.start, end + 1):
        token = tokens[token_id]
        if isinstance(token, bytes):
            token = token.decode("utf-8", errors="backslashreplace")

        if token_types is not None:
            token_type = token_types[token_id]
            print(f"{token_id:6d}  type={token_type!s:>2}  {token!r}")
        else:
            print(f"{token_id:6d}  {token!r}")


if __name__ == "__main__":
    main()
