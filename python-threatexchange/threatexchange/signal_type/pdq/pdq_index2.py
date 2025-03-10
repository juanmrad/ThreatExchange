# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Implementation of SignalTypeIndex abstraction for PDQ
"""

import typing as t
import faiss
import numpy as np


from threatexchange.signal_type.index import (
    IndexMatchUntyped,
    SignalSimilarityInfoWithIntDistance,
    SignalTypeIndex,
    T as IndexT,
)
from threatexchange.signal_type.pdq.pdq_utils import (
    BITS_IN_PDQ,
    PDQ_CONFIDENT_MATCH_THRESHOLD,
    convert_pdq_strings_to_ndarray,
    hex_to_binary_str,
    binary_str_to_hex,
)

PDQIndexMatch = IndexMatchUntyped[SignalSimilarityInfoWithIntDistance, IndexT]


class PDQIndex2(SignalTypeIndex[IndexT]):
    """
    Indexing and querying PDQ signals using Faiss for approximate nearest neighbor search.

    This is a redo of the existing PDQ index,
    designed to be simpler and fix hard-to-squash bugs in the existing implementation.
    Purpose of this class: to replace the original index in pytx 2.0
    """

    def __init__(
        self,
        index: t.Optional[faiss.Index] = None,
        entries: t.Iterable[t.Tuple[str, IndexT]] = (),
        *,
        threshold: int = PDQ_CONFIDENT_MATCH_THRESHOLD,
    ) -> None:
        super().__init__()
        self.threshold = threshold

        if index is None:
            index = faiss.IndexFlatL2(BITS_IN_PDQ)
        self._index = _PDQFaissIndex(index)

        # Matches hash to Faiss index
        self._deduper: t.Dict[str, int] = {}
        # Entry mapping: Each list[entries]'s index is its hash's index
        self._idx_to_entries: t.List[t.List[IndexT]] = []

        self.add_all(entries=entries)

    def __len__(self) -> int:
        return len(self._idx_to_entries)

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
        Look up entries against the index, up to the threshold.
        Consider rotated and flipped versions of the image.
        """
        transformed_hashes = self._get_transformed_hashes(hash)
        all_matches = []
        seen_ids = set()  # To avoid duplicate matches

        for transformed_hash in transformed_hashes:
            matches_list = self._index.search(
                queries=[transformed_hash], threshold=self.threshold
            )

            for match, distance in matches_list:
                if match not in seen_ids:
                    seen_ids.add(match)
                    entries = self._idx_to_entries[match]
                    # Create match objects for each entry
                    all_matches.extend(
                        PDQIndexMatch(
                            SignalSimilarityInfoWithIntDistance(distance=distance),
                            entry,
                        )
                        for entry in entries
                    )

        return all_matches

    def add(self, signal_str: str, entry: IndexT) -> None:
        self.add_all(((signal_str, entry),))

    def add_all(self, entries: t.Iterable[t.Tuple[str, IndexT]]) -> None:
        for h, i in entries:
            existing_faiss_id = self._deduper.get(h)
            if existing_faiss_id is None:
                self._index.add([h])
                self._idx_to_entries.append([i])
                next_id = len(self._deduper)  # Because faiss index starts from 0 up
                self._deduper[h] = next_id
            else:
                # Since this already exists, we don't add it to Faiss because Faiss cannot handle duplication
                self._idx_to_entries[existing_faiss_id].append(i)


class _PDQFaissIndex:
    """
    A wrapper around the faiss index for pickle serialization
    """

    def __init__(self, faiss_index: faiss.Index) -> None:
        self.faiss_index = faiss_index

    def add(self, pdq_strings: t.Sequence[str]) -> None:
        """
        Add PDQ hashes to the FAISS index.
        """
        vectors = convert_pdq_strings_to_ndarray(pdq_strings)
        self.faiss_index.add(vectors)

    def search(
        self, queries: t.Sequence[str], threshold: int
    ) -> t.List[t.Tuple[int, int]]:
        """
        Search the FAISS index for matches to the given PDQ queries.
        """
        query_array: np.ndarray = convert_pdq_strings_to_ndarray(queries)
        limits, distances, indices = self.faiss_index.range_search(
            query_array, threshold + 1
        )

        results: t.List[t.Tuple[int, int]] = []
        for i in range(len(queries)):
            matches = [idx.item() for idx in indices[limits[i] : limits[i + 1]]]
            dists = [dist for dist in distances[limits[i] : limits[i + 1]]]
            for j in range(len(matches)):
                results.append((matches[j], dists[j]))
        return results

    def __getstate__(self):
        return faiss.serialize_index(self.faiss_index)

    def __setstate__(self, data):
        self.faiss_index = faiss.deserialize_index(data)
