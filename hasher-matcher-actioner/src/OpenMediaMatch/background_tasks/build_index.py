# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import time
import gc
import typing as t
from typing import Optional

from threatexchange.signal_type.signal_base import SignalType

from OpenMediaMatch.background_tasks.development import get_apscheduler
from OpenMediaMatch.persistence import get_storage
from OpenMediaMatch.storage.interface import (
    ISignalTypeIndexStore,
    ISignalTypeConfigStore,
    IBankStore,
    SignalTypeIndexBuildCheckpoint,
)
from OpenMediaMatch.utils.time_utils import duration_to_human_str
from OpenMediaMatch.utils.memory_monitoring import MemoryMonitor

logger = logging.getLogger(__name__)


def apscheduler_build_all_indices() -> None:
    with get_apscheduler().app.app_context():
        storage = get_storage()
        build_all_indices(storage, storage, storage)


def build_all_indices(
    signal_type_cfgs: ISignalTypeConfigStore,
    bank_store: IBankStore,
    index_store: ISignalTypeIndexStore,
) -> None:
    """
    Build all indices from scratch from current bank contents and persist them

    Any additional indices (for disabled SignalTypes) are deleted.
    """
    start = time.time()
    logger.info("Running the %s background task", build_all_indices.__name__)
    enabled = signal_type_cfgs.get_enabled_signal_types()
    
    # Monitor memory before starting all indices
    monitor = MemoryMonitor()
    monitor.log_snapshot("Before building all indices")
    
    for st in enabled.values():
        # Force rebuild by clearing the checkpoint
        index_store.store_signal_type_index(
            st,
            st.get_index_cls().build([]),
            SignalTypeIndexBuildCheckpoint.get_empty()
        )
        build_index(st, bank_store, index_store)

    # Monitor memory after building all indices
    monitor.log_snapshot("After building all indices")
    monitor.log_memory_trend()

    logger.info(
        "Completed %s background task - %s",
        build_all_indices.__name__,
        duration_to_human_str(int(time.time() - start)),
    )
    # TODO cleanup disabled / deleted signal types


def build_index(
    for_signal_type: t.Type[SignalType],
    bank_store: IBankStore,
    index_store: ISignalTypeIndexStore,
) -> None:
    """
    Build one index from scratch with the current bank contents and persist it.
    """
    start = time.time()
    monitor = MemoryMonitor()
    logger.info(f"Starting index build for {for_signal_type.get_name()}")
    logger.info(monitor.log_snapshot(f"Starting index build for {for_signal_type.get_name()}"))
    
    # First check to see if new signals have appeared since the last build
    idx_checkpoint = index_store.get_last_index_build_checkpoint(for_signal_type)
    bank_checkpoint = bank_store.get_current_index_build_target(for_signal_type)
    
    logger.info(
        "Building index for %s (%d signals)",
        for_signal_type.get_name(),
        0 if bank_checkpoint is None else bank_checkpoint.total_hash_count,
    )
    
    # Monitor memory during signal collection
    signal_list = []
    last_cs = None
    signal_count = 0
    logger.info(monitor.log_snapshot(f"Before signal collection for {for_signal_type.get_name()}"))
    
    # Collect signals in a more memory-efficient way
    for last_cs in bank_store.bank_yield_content(for_signal_type):
        signal_list.append((last_cs.signal_val, last_cs.bank_content_id))
        signal_count += 1
        if signal_count % 1000 == 0:  # Log memory every 1000 signals
            logger.info(monitor.log_snapshot(f"After collecting {signal_count} signals for {for_signal_type.get_name()}"))
    
    logger.info(monitor.log_snapshot(f"After signal collection for {for_signal_type.get_name()}"))
    
    # Monitor memory during index building
    logger.info(monitor.log_snapshot(f"Before index construction for {for_signal_type.get_name()}"))
    index_cls = for_signal_type.get_index_cls()
    
    # Build index and store it
    built_index: Optional[t.Any] = index_cls.build(signal_list)
    logger.info(monitor.log_snapshot(f"After index construction for {for_signal_type.get_name()}"))
    
    
    checkpoint = SignalTypeIndexBuildCheckpoint.get_empty()
    if last_cs is not None:
        checkpoint = SignalTypeIndexBuildCheckpoint(
            last_item_timestamp=last_cs.bank_content_timestamp,
            last_item_id=last_cs.bank_content_id,
            total_hash_count=len(signal_list),
        )
        
    # Monitor memory during index storage
    logger.info(monitor.log_snapshot(f"Before index storage for {for_signal_type.get_name()}"))
    
    if built_index is not None:
        index_store.store_signal_type_index(for_signal_type, built_index, checkpoint)
    logger.info(monitor.log_snapshot(f"After index storage for {for_signal_type.get_name()}"))
    

    # Clean up temporary objects and force GC after index build
    signal_list.clear()
    gc.collect()
    logger.info(monitor.log_snapshot(f"After cleanup before storage for {for_signal_type.get_name()}"))

    # Log final memory trends
    logger.info(monitor.log_memory_trend())
    
    # Final memory snapshot
    logger.info(monitor.log_snapshot(f"After index build for {for_signal_type.get_name()}"))
    
    logger.info(
        "Indexed %d signals for %s - %s",
        signal_count,
        for_signal_type.get_name(),
        duration_to_human_str(int(time.time() - start)),
    )
