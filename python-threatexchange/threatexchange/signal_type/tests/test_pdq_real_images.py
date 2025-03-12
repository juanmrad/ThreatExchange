# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import pathlib
import pytest

from threatexchange.signal_type.pdq.pdq_hasher import pdq_from_file
from threatexchange.signal_type.pdq.pdq_index import PDQIndex
from threatexchange.signal_type.pdq.pdq_index2 import PDQIndex2


def get_resource_path(filename: str) -> pathlib.Path:
    """Get the path to a resource file."""
    # Get the directory where this test file is located
    test_dir = pathlib.Path(__file__).parent
    # Return path to the resources directory within the test directory
    return test_dir / "resources" / filename


class TestPDQRealImages:
    """Test PDQ hash transformations with real images."""

    @pytest.mark.parametrize("index_class", [PDQIndex, PDQIndex2])
    def test_flipped_image_match(self, index_class):
        """Test that a flipped image matches the original using PDQ transformations."""
        # Get paths to the test images
        original_path = get_resource_path("original.jpg")
        flipped_path = get_resource_path("flipped.jpg")

        # Ensure the test images exist
        assert original_path.exists(), f"Original image not found at {original_path}"
        assert flipped_path.exists(), f"Flipped image not found at {flipped_path}"

        # Compute PDQ hashes for both images
        original_hash, original_quality = pdq_from_file(original_path)
        flipped_hash, flipped_quality = pdq_from_file(flipped_path)

        # Ensure we got valid hashes
        assert len(original_hash) == 64, "Invalid original hash"
        assert len(flipped_hash) == 64, "Invalid flipped hash"

        # Create an index with the original hash
        index = index_class(entries=[(original_hash, "original")])

        # Query with the flipped hash - should find the original
        results = index.query(flipped_hash)
        assert (
            len(results) > 0
        ), "Should find a match between original and flipped images"
        assert results[0].metadata == "original", "Should match with the original image"

        # Now test the reverse - index with flipped, query with original
        index = index_class(entries=[(flipped_hash, "flipped")])
        results = index.query(original_hash)
        assert (
            len(results) > 0
        ), "Should find a match between flipped and original images"
        assert results[0].metadata == "flipped", "Should match with the flipped image"

    @pytest.mark.parametrize("index_class", [PDQIndex, PDQIndex2])
    def test_without_transformations(self, index_class):
        """
        Test that without transformations, flipped images might not match.
        This test demonstrates the value of our transformation feature.
        """
        # Get paths to the test images
        original_path = get_resource_path("original.jpg")
        flipped_path = get_resource_path("flipped.jpg")

        # Compute PDQ hashes for both images
        original_hash, _ = pdq_from_file(original_path)
        flipped_hash, _ = pdq_from_file(flipped_path)

        # Create a mock version of _get_transformed_hashes that only returns the original hash
        original_get_transformed = index_class._get_transformed_hashes

        try:
            # Override the transformation method to only return the original hash
            index_class._get_transformed_hashes = lambda self, hash: [hash]

            # Create an index with the original hash
            index = index_class(entries=[(original_hash, "original")])

            # Query with the flipped hash - should NOT find the original
            # because we disabled transformations
            results = index.query(flipped_hash)

            # The test passes if either:
            # 1. No matches are found (expected behavior)
            # 2. The distance is higher than it would be with transformations
            if results:
                # If we do find a match, it should be at a higher distance
                # than we would get with transformations
                with_transformations = original_get_transformed(index, flipped_hash)
                assert (
                    len(with_transformations) > 1
                ), "Transformation function not working"

                # Create a new index to test with transformations
                index_class._get_transformed_hashes = original_get_transformed
                index_with_transform = index_class(
                    entries=[(original_hash, "original")]
                )
                results_with_transform = index_with_transform.query(flipped_hash)

                # The distance with transformations should be lower (better match)
                assert (
                    results[0].similarity_info.distance
                    > results_with_transform[0].similarity_info.distance
                ), "Transformations should improve match quality"

        finally:
            # Restore the original method
            index_class._get_transformed_hashes = original_get_transformed
