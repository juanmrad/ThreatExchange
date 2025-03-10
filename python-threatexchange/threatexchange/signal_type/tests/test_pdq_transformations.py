# Copyright (c) Meta Platforms, Inc. and affiliates.

import pytest
import typing as t

from threatexchange.signal_type.pdq.pdq_index import PDQIndex
from threatexchange.signal_type.pdq.pdq_index2 import PDQIndex2
from threatexchange.signal_type.pdq.pdq_utils import (
    hex_to_binary_str,
    binary_str_to_hex,
)


# Test data: PDQ hash and its known transformations
# This is a simple pattern that makes transformations easy to verify visually
ORIGINAL_HASH = "f" * 32 + "0" * 32  # 32 'f's followed by 32 '0's
ROTATED_90_HASH = "c30c30c30c30c30c30c30c30c30c30c30c30c30c30c30c30c30c30c30c30c30c30c30c30c30c30c30c30c30c30c30c30c"
ROTATED_180_HASH = "0" * 32 + "f" * 32  # 32 '0's followed by 32 'f's
FLIPPED_H_HASH = "f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0"


def create_test_hash_with_pattern() -> str:
    """Create a test hash with a pattern that's easy to verify after transformations."""
    # Create a 16x16 grid where the top half is all 1s and bottom half is all 0s
    grid = []
    for i in range(8):
        grid.append("1" * 16)
    for i in range(8):
        grid.append("0" * 16)

    binary = "".join(grid)
    return binary_str_to_hex(binary)


def manually_rotate_90(hash_str: str) -> str:
    """Manually rotate a PDQ hash 90 degrees clockwise for testing."""
    binary = hex_to_binary_str(hash_str)

    # Create a 16x16 grid from the binary string
    grid = []
    for i in range(16):
        row = []
        for j in range(16):
            idx = i * 16 + j
            row.append(binary[idx])
        grid.append(row)

    # Rotate 90 degrees clockwise
    rotated = []
    for j in range(16):
        row = []
        for i in range(15, -1, -1):
            row.append(grid[i][j])
        rotated.append(row)

    # Convert back to binary and then to hex
    binary_rotated = ""
    for row in rotated:
        binary_rotated += "".join(row)

    return binary_str_to_hex(binary_rotated)


def manually_rotate_180(hash_str: str) -> str:
    """Manually rotate a PDQ hash 180 degrees for testing."""
    binary = hex_to_binary_str(hash_str)

    # Create a 16x16 grid from the binary string
    grid = []
    for i in range(16):
        row = []
        for j in range(16):
            idx = i * 16 + j
            row.append(binary[idx])
        grid.append(row)

    # Rotate 180 degrees
    rotated = []
    for i in range(15, -1, -1):
        row = []
        for j in range(15, -1, -1):
            row.append(grid[i][j])
        rotated.append(row)

    # Convert back to binary and then to hex
    binary_rotated = ""
    for row in rotated:
        binary_rotated += "".join(row)

    return binary_str_to_hex(binary_rotated)


def manually_flip_horizontal(hash_str: str) -> str:
    """Manually flip a PDQ hash horizontally for testing."""
    binary = hex_to_binary_str(hash_str)

    # Create a 16x16 grid from the binary string
    grid = []
    for i in range(16):
        row = []
        for j in range(16):
            idx = i * 16 + j
            row.append(binary[idx])
        grid.append(row)

    # Flip horizontally
    flipped = []
    for i in range(16):
        row = []
        for j in range(15, -1, -1):
            row.append(grid[i][j])
        flipped.append(row)

    # Convert back to binary and then to hex
    binary_flipped = ""
    for row in flipped:
        binary_flipped += "".join(row)

    return binary_str_to_hex(binary_flipped)


class TestPDQTransformations:
    """Test PDQ hash transformations in both PDQIndex implementations."""

    @pytest.mark.parametrize("index_class", [PDQIndex, PDQIndex2])
    def test_get_transformed_hashes(self, index_class):
        """Test that _get_transformed_hashes generates the expected transformations."""
        index = index_class()
        test_hash = create_test_hash_with_pattern()

        transformed_hashes = index._get_transformed_hashes(test_hash)

        # Should have 8 hashes (original + 7 transformations)
        assert (
            len(transformed_hashes) == 8
        ), f"Expected 8 transformed hashes, got {len(transformed_hashes)}"

        # Original hash should be the first one
        assert transformed_hashes[0] == test_hash, "First hash should be the original"

        # Verify that our manually calculated transformations match
        expected_90_rotation = manually_rotate_90(test_hash)
        expected_180_rotation = manually_rotate_180(test_hash)
        expected_h_flip = manually_flip_horizontal(test_hash)

        assert (
            expected_90_rotation in transformed_hashes
        ), "90-degree rotation not found"
        assert (
            expected_180_rotation in transformed_hashes
        ), "180-degree rotation not found"
        assert expected_h_flip in transformed_hashes, "Horizontal flip not found"

    @pytest.mark.parametrize("index_class", [PDQIndex, PDQIndex2])
    def test_query_with_transformations(self, index_class):
        """Test that querying finds matches with transformed hashes."""
        # Create an index with just the original hash
        original_hash = create_test_hash_with_pattern()
        index = index_class(entries=[(original_hash, "original")])

        # Query with the original hash
        results = index.query(original_hash)
        assert len(results) == 1, "Should find 1 match with original hash"
        assert results[0].metadata == "original"

        # Query with a 90-degree rotated hash
        rotated_hash = manually_rotate_90(original_hash)
        results = index.query(rotated_hash)
        assert len(results) == 1, "Should find 1 match with rotated hash"
        assert results[0].metadata == "original"

        # Query with a 180-degree rotated hash
        rotated_hash = manually_rotate_180(original_hash)
        results = index.query(rotated_hash)
        assert len(results) == 1, "Should find 1 match with rotated hash"
        assert results[0].metadata == "original"

        # Query with a horizontally flipped hash
        flipped_hash = manually_flip_horizontal(original_hash)
        results = index.query(flipped_hash)
        assert len(results) == 1, "Should find 1 match with flipped hash"
        assert results[0].metadata == "original"

    @pytest.mark.parametrize("index_class", [PDQIndex, PDQIndex2])
    def test_query_with_multiple_matches(self, index_class):
        """Test that querying finds all matches without duplicates."""
        # Create an index with multiple entries
        original_hash = create_test_hash_with_pattern()
        rotated_hash = manually_rotate_90(original_hash)

        index = index_class(
            entries=[(original_hash, "original"), (rotated_hash, "rotated")]
        )

        # Query with the original hash - should find both entries
        results = index.query(original_hash)
        assert len(results) == 2, "Should find 2 matches with original hash"

        # Extract metadata from results
        metadata = [r.metadata for r in results]
        assert "original" in metadata, "Should find 'original' entry"
        assert "rotated" in metadata, "Should find 'rotated' entry"

        # Query with a different transformation - should still find both entries
        flipped_hash = manually_flip_horizontal(original_hash)
        results = index.query(flipped_hash)
        assert len(results) == 2, "Should find 2 matches with flipped hash"

        # Extract metadata from results
        metadata = [r.metadata for r in results]
        assert "original" in metadata, "Should find 'original' entry"
        assert "rotated" in metadata, "Should find 'rotated' entry"
