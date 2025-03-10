# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Implementation of SignalTypeIndex abstraction for PDQ by wrapping
hashing.pdq_faiss_matcher.
"""

import typing as t

from threatexchange.signal_type.index import (
    IndexMatchUntyped,
    SignalSimilarityInfoWithIntDistance,
    SignalTypeIndex,
    T as IndexT,
)
from threatexchange.signal_type.pdq.pdq_faiss_matcher import (
    PDQMultiHashIndex,
    PDQFlatHashIndex,
    PDQHashIndex,
)
from threatexchange.signal_type.pdq.pdq_utils import (
    hex_to_binary_str,
    binary_str_to_hex,
)

PDQIndexMatch = IndexMatchUntyped[SignalSimilarityInfoWithIntDistance, IndexT]


class PDQIndex(SignalTypeIndex[IndexT]):
    """
    Wrapper around the pdq faiss index lib using PDQMultiHashIndex
    """

    @classmethod
    def get_match_threshold(cls):
        return 31  # PDQ_CONFIDENT_MATCH_THRESHOLD

    @classmethod
    def _get_empty_index(cls) -> PDQHashIndex:
        return PDQMultiHashIndex()

    def __init__(self, entries: t.Iterable[t.Tuple[str, IndexT]] = ()) -> None:
        super().__init__()
        self.local_id_to_entry: t.List[t.Tuple[str, IndexT]] = []
        self.index: PDQHashIndex = self._get_empty_index()
        self.add_all(entries=entries)

    def __len__(self) -> int:
        return len(self.local_id_to_entry)

    def _get_transformed_hashes(self, hash: str) -> t.List[str]:
        """
        Generate transformed versions of the hash to account for rotations and flips.

        This includes:
        - Original hash
        - 90, 180, 270 degree rotations
        - Horizontal flip
        - Vertical flip
        - Horizontal flip + rotations

        PDQ hashes are 256-bit (64 hex character) representations where bits encode
        spatial information in a 16x16 grid. To transform them, we need to rearrange
        these bits according to the desired transformation.
        """
        # Convert hex hash to binary for easier manipulation
        binary = hex_to_binary_str(hash)

        # Create a 16x16 grid from the binary string
        grid = []
        for i in range(16):
            row = []
            for j in range(16):
                idx = i * 16 + j
                row.append(binary[idx])
            grid.append(row)

        # Original hash
        transformed_hashes = [hash]

        # Rotate 90 degrees clockwise
        rotated_90 = []
        for j in range(16):
            row = []
            for i in range(15, -1, -1):
                row.append(grid[i][j])
            rotated_90.append(row)

        # Rotate 180 degrees
        rotated_180 = []
        for i in range(15, -1, -1):
            row = []
            for j in range(15, -1, -1):
                row.append(grid[i][j])
            rotated_180.append(row)

        # Rotate 270 degrees clockwise
        rotated_270 = []
        for j in range(15, -1, -1):
            row = []
            for i in range(16):
                row.append(grid[i][j])
            rotated_270.append(row)

        # Horizontal flip
        flipped_h = []
        for i in range(16):
            row = []
            for j in range(15, -1, -1):
                row.append(grid[i][j])
            flipped_h.append(row)

        # Vertical flip
        flipped_v = []
        for i in range(15, -1, -1):
            row = []
            for j in range(16):
                row.append(grid[i][j])
            flipped_v.append(row)

        # Horizontal flip + 90 degree rotation
        flipped_h_90 = []
        for j in range(16):
            row = []
            for i in range(16):
                row.append(flipped_h[i][j])
            flipped_h_90.append(row)

        # Horizontal flip + 270 degree rotation
        flipped_h_270 = []
        for j in range(15, -1, -1):
            row = []
            for i in range(15, -1, -1):
                row.append(flipped_h[i][j])
            flipped_h_270.append(row)

        # Convert all grids back to binary strings and then to hex
        for grid_variant in [
            rotated_90,
            rotated_180,
            rotated_270,
            flipped_h,
            flipped_v,
            flipped_h_90,
            flipped_h_270,
        ]:
            binary_variant = ""
            for row in grid_variant:
                binary_variant += "".join(row)
            hex_variant = binary_str_to_hex(binary_variant)
            transformed_hashes.append(hex_variant)

        return transformed_hashes

    def query(self, hash: str) -> t.Sequence[PDQIndexMatch[IndexT]]:
        """
        Look up entries against the index, up to the max supported distance.
        Consider rotated and flipped versions of the image.
        """
        transformed_hashes = self._get_transformed_hashes(hash)
        all_matches = []
        seen_ids = set()  # To avoid duplicate matches

        # Query the index with all transformed hashes at once
        results = self.index.search_with_distance_in_result(
            transformed_hashes, self.get_match_threshold()
        )

        # Process results for all transformations
        for transformed_hash in transformed_hashes:
            for id, _, distance in results[transformed_hash]:
                if id not in seen_ids:
                    seen_ids.add(id)
                    all_matches.append(
                        IndexMatchUntyped(
                            SignalSimilarityInfoWithIntDistance(int(distance)),
                            self.local_id_to_entry[id][1],
                        )
                    )

        return all_matches

    def add(self, signal_str: str, entry: IndexT) -> None:
        self.add_all(((signal_str, entry),))

    def add_all(self, entries: t.Iterable[t.Tuple[str, IndexT]]) -> None:
        start = len(self.local_id_to_entry)
        self.local_id_to_entry.extend(entries)
        if start != len(self.local_id_to_entry):
            # This function signature is very silly
            self.index.add(
                (e[0] for e in self.local_id_to_entry[start:]),
                range(start, len(self.local_id_to_entry)),
            )


class PDQFlatIndex(PDQIndex):
    """
    Wrapper around the pdq faiss index lib
    that uses PDQFlatHashIndex instead of PDQMultiHashIndex
    It also uses a high match threshold to increase recall
    possibly as the cost of precision.
    """

    @classmethod
    def get_match_threshold(cls):
        return 52  # larger PDQ_MATCH_THRESHOLD for flatindexes

    @classmethod
    def _get_empty_index(cls) -> PDQHashIndex:
        return PDQFlatHashIndex()
