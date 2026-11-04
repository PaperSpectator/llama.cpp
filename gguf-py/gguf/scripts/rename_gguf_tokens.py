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
import tempfile
from pathlib import Path

from gguf import GGUFReader, GGUFWriter, GGUFValueType, Keys
from tqdm import tqdm

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

    token_field = fields.get(Keys.Tokenizer.LIST)
    if token_field is None:
        raise ValueError(f"Missing GGUF field: {Keys.Tokenizer.LIST}")

    if token_field.types != [
        GGUFValueType.ARRAY,
        GGUFValueType.STRING,
    ]:
        raise ValueError(
            f"{Keys.Tokenizer.LIST} is not an ARRAY<STRING>: {token_field.types}"
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

    old_data_start = reader.data_offset
    alignment = int(reader.alignment)

    # Snapshot the token types so we can verify them after rewriting.
    type_field = fields.get(Keys.Tokenizer.TOKEN_TYPE)
    original_types = type_field.contents() if type_field else None

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
            arch = reader.get_field(Keys.General.ARCHITECTURE).contents()
            writer = GGUFWriter(out.name, arch=arch, endianess=reader.endianess)
            alignment_f = reader.get_field(Keys.General.ALIGNMENT)
            if alignment_f is not None:
                writer.data_alignment = alignment.contents()
            for field in reader.fields.values():
                if field.name == Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
                    # if we copy these, they'll get dupped and the checker will complain
                    continue
                if field.name == Keys.Tokenizer.LIST:
                    writer.add_key_value(Keys.Tokenizer.LIST, tokens, GGUFValueType.ARRAY, sub_type=GGUFValueType.STRING)
                else:
                    val_type = field.types[0]
                    sub_type = field.types[-1] if val_type == GGUFValueType.ARRAY else None
                    val = field.contents()
                    writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

            total_bytes = 0
            for tensor in reader.tensors:
                total_bytes += tensor.n_bytes
                writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type)

            bar = tqdm(desc="Writing", total=total_bytes, unit="byte", unit_scale=True)

            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_ti_data_to_file()
            for tensor in reader.tensors:
                writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
                bar.update(tensor.n_bytes)
            writer.close()

        # Validate the temporary output before publishing it.
        check = GGUFReader(temp_path)
        try:
            check_tokens = check.fields[Keys.Tokenizer.LIST].contents()
            for token_id, replacement in mapping.items():
                if check_tokens[token_id] != replacement:
                    raise RuntimeError(
                        f"Post-write verification failed at token {token_id}"
                    )

            if original_types is not None:
                check_types = check.fields[Keys.Tokenizer.TOKEN_TYPE].contents()
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
