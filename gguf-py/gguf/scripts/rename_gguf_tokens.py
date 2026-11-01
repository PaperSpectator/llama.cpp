#!/usr/bin/env python3
"""
Surgically rename existing entries in tokenizer.ggml.tokens.

Only the selected GGUF metadata field is rewritten. Tensor info is copied
unchanged, alignment is recalculated, and the tensor-data region is copied
byte-for-byte.

Requires the gguf-py package from llama.cpp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path

from gguf import GGUFReader, GGUFValueType


TOKENS_KEY = "tokenizer.ggml.tokens"
TOKEN_TYPES_KEY = "tokenizer.ggml.token_type"
CHUNK_SIZE = 8 * 1024 * 1024


def digest_region(data, start: int, end: int) -> str:
    h = hashlib.sha256()
    for pos in range(start, end, CHUNK_SIZE):
        h.update(data[pos:min(pos + CHUNK_SIZE, end)])
    return h.hexdigest()


def write_region(out, data, start: int, end: int) -> None:
    for pos in range(start, end, CHUNK_SIZE):
        out.write(data[pos:min(pos + CHUNK_SIZE, end)])

def close_gguf_reader(reader):
    """Release the memory-mapped file held by GGUFReader."""
    data = getattr(reader, "data", None)
    mmap_obj = getattr(data, "_mmap", None)

    if mmap_obj is not None:
        mmap_obj.close()

def encode_token_array(tokens: list[str], endian: str) -> bytes:
    """Encode a GGUF ARRAY<STRING> value (including its array type/count)."""
    result = bytearray()
    result += struct.pack(endian + "I", int(GGUFValueType.STRING))
    result += struct.pack(endian + "Q", len(tokens))

    for token in tokens:
        encoded = token.encode("utf-8")
        result += struct.pack(endian + "Q", len(encoded))
        result += encoded

    return bytes(result)


def build_replacement_field(
    key: str,
    tokens: list[str],
    endian: str,
) -> bytes:
    """Encode the complete key + value-type + array field."""
    key_bytes = key.encode("utf-8")
    result = bytearray()

    result += struct.pack(endian + "Q", len(key_bytes))
    result += key_bytes
    result += struct.pack(endian + "I", int(GGUFValueType.ARRAY))
    result += encode_token_array(tokens, endian)

    return bytes(result)


def rename_tokens(
    source: Path,
    output: Path,
    mapping: dict[int, str],
    expected: dict[int, str] | None = None,
) -> None:
    if source.resolve() == output.resolve():
        raise ValueError("Source and output must be different files")

    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    if not mapping:
        raise ValueError("Mapping is empty")

    if any(i < 0 for i in mapping):
        raise ValueError("Token IDs must be non-negative")

    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Replacement spellings must be unique")

    reader = GGUFReader(source)
    fields = reader.fields

    token_field = fields.get(TOKENS_KEY)
    if token_field is None:
        raise ValueError(f"Missing GGUF field: {TOKENS_KEY}")

    if token_field.types != [
        GGUFValueType.ARRAY,
        GGUFValueType.STRING,
    ]:
        raise ValueError(
            f"{TOKENS_KEY} is not an ARRAY<STRING>: {token_field.types}"
        )

    tokens = token_field.contents()
    token_count = len(tokens)

    for token_id, replacement in mapping.items():
        if token_id >= token_count:
            raise ValueError(
                f"Token ID {token_id} is outside vocabulary "
                f"(size {token_count})"
            )

        if not isinstance(replacement, str) or not replacement:
            raise ValueError(f"Invalid replacement for token {token_id}")

        old = tokens[token_id]
        if expected is not None and token_id in expected:
            if old != expected[token_id]:
                raise ValueError(
                    f"Token {token_id} mismatch: expected "
                    f"{expected[token_id]!r}, found {old!r}"
                )

        print(f"{token_id:6d}  {old!r} -> {replacement!r}")
        tokens[token_id] = replacement

    # Refuse to introduce duplicate token spellings.
    if len(set(tokens)) != len(tokens):
        raise ValueError("Replacement would create duplicate token spellings")

    # The current prototype expects at least one tensor.
    if not reader.tensors:
        raise ValueError("This prototype requires at least one tensor")

    endian = "<" if reader.byte_order == "I" else ">"

    # ReaderField.offset points to the beginning of the complete KV entry.
    field_start = token_field.offset
    field_end = field_start + sum(
        int(part.nbytes) for part in token_field.parts
    )

    # Tensor-info region starts at the first tensor's metadata record.
    tensor_info_start = min(t.field.offset for t in reader.tensors)
    tensor_info_end = max(
        t.field.offset + sum(int(p.nbytes) for p in t.field.parts)
        for t in reader.tensors
    )

    old_data_start = reader.data_offset
    alignment = int(reader.alignment)

    if not (
        24 <= field_start < field_end <= tensor_info_start
        <= tensor_info_end <= old_data_start <= reader.data.nbytes
    ):
        raise ValueError("Unexpected GGUF section layout; refusing to edit")

    # Snapshot the token types so we can verify them after rewriting.
    type_field = fields.get(TOKEN_TYPES_KEY)
    original_types = type_field.contents() if type_field else None

    replacement_field = build_replacement_field(
        TOKENS_KEY, tokens, endian
    )

    # Construct everything before the tensor data.
    #
    # Keep:
    #   header
    #   all KV bytes before the target field
    #   replacement field
    #   all remaining KV bytes
    #   original tensor-info records
    #
    # Recalculate padding to the next tensor-data alignment boundary.
    pre_data_size = (
        field_start
        + len(replacement_field)
        + (tensor_info_start - field_end)
        + (tensor_info_end - tensor_info_start)
    )
    new_data_start = (
        (pre_data_size + alignment - 1) // alignment
    ) * alignment

    padding = new_data_start - pre_data_size

    output.parent.mkdir(parents=True, exist_ok=True)

    original_tensor_hash = digest_region(
        reader.data, old_data_start, reader.data.nbytes
    )

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            prefix=output.name + ".",
            suffix=".tmp",
            delete=False,
        ) as out:
            temp_path = Path(out.name)

            # GGUF header, unchanged.
            write_region(out, reader.data, 0, field_start)

            # Re-encoded vocabulary field.
            out.write(replacement_field)

            # Remaining metadata KV entries.
            write_region(
                out, reader.data, field_end, tensor_info_start
            )

            # Original tensor-info records, unchanged.
            write_region(
                out, reader.data, tensor_info_start, tensor_info_end
            )

            # New alignment padding.
            if padding:
                out.write(b"\x00" * padding)

            # Original tensor-data region, unchanged.
            write_region(
                out, reader.data, old_data_start, reader.data.nbytes
            )

            out.flush()
            os.fsync(out.fileno())

        # Validate the temporary output before publishing it.
        check = GGUFReader(temp_path)
        try:
            check_tokens = check.fields[TOKENS_KEY].contents()
            for token_id, replacement in mapping.items():
                if check_tokens[token_id] != replacement:
                    raise RuntimeError(
                        f"Post-write verification failed at token {token_id}"
                    )

            if original_types is not None:
                check_types = check.fields[TOKEN_TYPES_KEY].contents()
                if check_types != original_types:
                    raise RuntimeError("Token types changed unexpectedly")

            new_tensor_hash = digest_region(
                check.data, check.data_offset, check.data.nbytes
            )

            if original_tensor_hash != new_tensor_hash:
                raise RuntimeError("Tensor-data hash mismatch!")

            if len(check.tensors) != len(reader.tensors):
                raise RuntimeError("Tensor count changed unexpectedly")

            for before, after in zip(reader.tensors, check.tensors):
                if (
                    before.name != after.name
                    or before.shape.tolist() != after.shape.tolist()
                    or before.tensor_type != after.tensor_type
                    or before.n_bytes != after.n_bytes
                ):
                    raise RuntimeError(
                        f"Tensor metadata mismatch: {before.name}"
                    )
        finally:
            # free the reader and working copy reader, it's done tweaking the working copy
            close_gguf_reader(reader)
            close_gguf_reader(check)

        # Publish only after all checks pass.
        os.replace(temp_path, output)
        temp_path = None

        print()
        print(f"Output: {output}")
        print(f"Vocabulary size: {len(check_tokens)}")
        print(f"Tensor count: {len(check.tensors)}")
        print(f"Tensor-data SHA256: {new_tensor_hash}")
        print("Verification: PASSED")

    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except PermissionError:
                print(
                    f"Warning: could not remove temporary file:\n"
                    f"  {temp_path}\n"
                    f"It may still be held open by another process."
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rename existing token spellings in a GGUF."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = json.loads(args.map.read_text(encoding="utf-8"))

    mapping = {
        int(token_id): value["new"]
        if isinstance(value, dict)
        else value
        for token_id, value in config.items()
    }

    expected = {
        int(token_id): value["old"]
        for token_id, value in config.items()
        if isinstance(value, dict) and "old" in value
    }

    rename_tokens(args.input, args.output, mapping, expected)


if __name__ == "__main__":
    main()
