// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <algorithm>
#ifdef _WINDOWS
#include <numeric>
#endif
#include <string>
#include <vector>

#include "distance.h"
#include "parameters.h"

namespace diskann {
  struct QueryStats {
    float total_us = 0;  // total time to process query in micros
    float io_us = 0;     // total time spent in IO
    float cpu_us = 0;    // total time spent in CPU
    float preprocess_us = 0;    // total time spent for pre process (in-mem nev)
    float postprocess_us = 0;    // total time spent for post process (sort and lim-k)

    float dispatch_us = 0;    // total time spent for dispatch nodes
    float read_disk_us = 0;    // total time spent for read disk process
    float cache_proc_us = 0;    // total time spent for cache node process
    float disk_proc_us = 0;    // total time spent for disk node process
    float page_proc_us = 0;    // total time spent for page process in PageSearch

    unsigned n_ios = 0;         // total # of search IOs issued
    unsigned n_emb_ios = 0;     // total # of refine IOs issued
    unsigned n_cmps = 0;        // # cmps
    unsigned n_ext_cmps = 0;    // # exact cmps
    unsigned n_cache_hits = 0;  // # cache_hits
    unsigned n_hops = 0;        // # search hops

    uint64_t region_prefetch_enabled = 0;
    uint64_t region_prefetch_overflow = 0;
    uint64_t region_logical_graph_ios = 0;
    uint64_t region_device_read_submits = 0;
    uint64_t region_device_read_bytes = 0;
    uint64_t region_device_4kb_reads = 0;
    uint64_t region_device_16kb_reads = 0;
    uint64_t region_first_triggers = 0;
    uint64_t region_cache_hits = 0;
    uint64_t region_loaded_regions = 0;
    uint64_t region_additional_pages_read = 0;
    uint64_t region_additional_pages_used = 0;
    uint64_t region_unused_additional_pages = 0;
    uint64_t region_duplicate_load_attempts = 0;
    uint64_t region_same_region_pending_frontier_count = 0;
    uint64_t region_pending_waiter_registered_count = 0;
    uint64_t region_pending_waiter_processed_count = 0;
    uint64_t region_pending_waiter_duplicate_count = 0;
    uint64_t region_requested_page_parse_count = 0;
    uint64_t region_prefetched_only_page_parse_count = 0;
    uint64_t region_pending_region_leftover_at_query_end = 0;
    uint64_t region_waiter_registered_but_not_processed = 0;
    uint64_t region_duplicate_region_submission_count = 0;
    uint64_t region_duplicate_logical_page_process_count = 0;
    uint64_t region_slice_byte_mismatch_count = 0;
    uint64_t region_cache_entries_released_at_query_end = 0;
    uint64_t region_cache_leftover_after_cleanup = 0;
    uint64_t region_cache_peak_bytes = 0;
    uint64_t region_cache_allocations = 0;
    uint64_t region_cache_frees = 0;
    float region_lookup_us = 0;
    float region_cache_manage_us = 0;

    uint64_t deterministic_logical_page_processing_enabled = 0;
    uint64_t deterministic_logical_page_request_count = 0;
    uint64_t deterministic_logical_request_seq_assigned_count = 0;
    uint64_t deterministic_logical_page_ready_count = 0;
    uint64_t deterministic_logical_page_processed_count = 0;
    uint64_t deterministic_logical_page_duplicate_request_count = 0;
    uint64_t deterministic_logical_page_duplicate_process_count = 0;
    uint64_t deterministic_queue_insert_count = 0;
    uint64_t deterministic_queue_max_depth = 0;
    uint64_t deterministic_queue_leftover_at_query_end = 0;
    uint64_t deterministic_requested_but_not_processed_count = 0;
    uint64_t deterministic_processed_without_request_count = 0;
    uint64_t deterministic_physical_io_submission_count = 0;
    uint64_t deterministic_logical_to_physical_coalescing_count = 0;

#ifdef ENABLE_REPLICA_REDUNDANCY_STATS
    uint64_t replica_graph_page_ios = 0;
    uint64_t replica_first_read_pages = 0;
    uint64_t replica_repeated_page_reads = 0;
    uint64_t replica_attempts = 0;
    uint64_t replica_attempts_on_first_read_pages = 0;
    uint64_t unique_replica_adjacencies = 0;
    uint64_t all_replica_duplicates = 0;
    uint64_t cross_page_replica_duplicates = 0;
    uint64_t replica_to_replica_duplicates = 0;
    uint64_t owner_to_replica_duplicates = 0;
    uint64_t pages_with_any_replica_duplicate = 0;
    uint64_t fully_redundant_replica_pages = 0;
    double replica_redundancy = 0.0;
    double duplicate_replicas_per_graph_io = 0.0;
#endif
  };

  template<typename T>
  inline T get_percentile_stats(
      QueryStats *stats, uint64_t len, float percentile,
      const std::function<T(const QueryStats &)> &member_fn) {
    std::vector<T> vals(len);
    for (uint64_t i = 0; i < len; i++) {
      vals[i] = member_fn(stats[i]);
    }

    std::sort(vals.begin(), vals.end(),
              [](const T &left, const T &right) { return left < right; });

    auto retval = vals[(uint64_t)(percentile * len)];
    vals.clear();
    return retval;
  }

  template<typename T>
  inline double get_mean_stats(
      QueryStats *stats, uint64_t len, 
      const std::function<T(const QueryStats &)> &member_fn) {
    double avg = 0;
    for (uint64_t i = 0; i < len; i++) {
      avg += (double) member_fn(stats[i]);
    }
    return avg / len;
  }

  template<typename T>
  inline double get_mean_stats(
      QueryStats *stats, uint64_t len, uint64_t warmup, 
      const std::function<T(const QueryStats &)> &member_fn) {
    double avg = 0;
    for (uint64_t i = warmup; i < len; i++) {
      avg += (double) member_fn(stats[i]);
    }
    return avg / (len - warmup);
  }

  template<typename T>
  inline T get_percentile_stats(
      QueryStats *stats, uint64_t len, uint64_t warmup, float percentile,
      const std::function<T(const QueryStats &)> &member_fn) {
    uint64_t n = len > warmup ? len - warmup : 0;
    if (n == 0) return T();
    std::vector<T> vals(n);
    for (uint64_t i = warmup; i < len; i++) {
      vals[i - warmup] = member_fn(stats[i]);
    }
    std::sort(vals.begin(), vals.end(),
              [](const T &left, const T &right) { return left < right; });
    uint64_t idx = static_cast<uint64_t>(percentile * n);
    if (idx >= n) idx = n - 1;
    auto retval = vals[idx];
    vals.clear();
    return retval;
  }

  // The following two functions are used when getting statistics while range searching on only queries with
  // non-zero gt lengths
  template<typename T>
  inline T get_percentile_stats_gt(
      QueryStats *stats, uint64_t len, float percentile,
      const std::function<T(const QueryStats &)> &member_fn, std::vector<std::vector<uint32_t>> &gt) {
    std::vector<T> vals;
    for (uint64_t i = 0; i < len; i++) {
      if (gt[i].size()) vals.push_back(member_fn(stats[i]));
    }

    std::sort(vals.begin(), vals.end(),
              [](const T &left, const T &right) { return left < right; });

    auto retval = vals[(uint64_t)(percentile * vals.size())];
    vals.clear();
    return retval;
  }

  template<typename T>
  inline double get_mean_stats_gt(
      QueryStats *stats, uint64_t len,
      const std::function<T(const QueryStats &)> &member_fn, std::vector<std::vector<uint32_t>> &gt) {
    uint32_t cnt = 0;
    double avg = 0;
    for (uint64_t i = 0; i < len; i++) {
      if (gt[i].size()) {
        ++cnt;
        avg += (double) member_fn(stats[i]);
      }
    }
    return avg / cnt;
  }
}  // namespace diskann
