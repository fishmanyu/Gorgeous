// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <atomic>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <omp.h>
#include <pq_flash_index.h>
#include <set>
#include <string.h>
#include <time.h>
#include <boost/program_options.hpp>

#include "aux_utils.h"
#include "index.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "partition_and_pq.h"
#include "timer.h"
#include "utils.h"
#include "percentile_stats.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "linux_aligned_file_reader.h"

// use engine
#include "thread_pool.h"
#include "file_io_manager.h"
#include "deco_index.h"

namespace po = boost::program_options;

void print_stats(std::string category, std::vector<float> percentiles,
                 std::vector<float> results) {
  diskann::cout << std::setw(20) << category << ": " << std::flush;
  for (uint32_t s = 0; s < percentiles.size(); s++) {
    diskann::cout << std::setw(8) << percentiles[s] << "%";
  }
  diskann::cout << std::endl;
  diskann::cout << std::setw(22) << " " << std::flush;
  for (uint32_t s = 0; s < percentiles.size(); s++) {
    diskann::cout << std::setw(9) << results[s];
  }
  diskann::cout << std::endl;
}

template<typename T>
int search_disk_index(
    diskann::Metric& metric, const std::string& index_path_prefix,
    const std::string& pq_path_prefix,
    const std::string& mem_index_path,
    const std::string& mem_sample_path,
    const std::string& result_output_prefix, const std::string& query_file,
    const std::string& gt_file,
    const std::string& disk_file_path,
    const std::string& disk_graph_prefix,
    const std::string& graph_rep_index_prefix,
    const unsigned num_threads, const unsigned recall_at,
    const unsigned beamwidth, const unsigned num_nodes_to_cache,
    const _u32 search_io_limit, const std::vector<unsigned>& Lvec,
    const _u32 mem_L,
    const _u64 sector_len,
    const bool use_page_search=true,
    const float use_ratio=1.0,
    const float pq_ratio=0.9,
    const bool deco_impl = false,
    const bool use_graph_rep_index = false,
    const bool enable_region_layout = false,
    const float mem_graph_use_ratio = 1.0,
    const float mem_emb_use_ratio = 1.0,
    const float emb_search_ratio = 1.0,
    const bool collect_transition_trace = false,
    const unsigned enable_trace_v2 = 0,
    const std::string& trace_v2_output_dir = "logs",
    const bool enable_region_io_trace = false,
    const std::string& region_io_trace_dir = "logs/region_io_trace",
    const std::string& region_file = "",
    const bool enable_global_qd_control = false,
    const unsigned global_qd_cap = 0,
    const bool qd_control_trace = false,
    const bool enable_region_prefetch = false,
    const unsigned region_prefetch_size = 4,
    const std::string& region_buffer_policy = "query_lifetime",
    const uint64_t region_cache_limit_bytes_per_query = 16ULL * 1024ULL * 1024ULL,
    const std::string& region_cache_overflow_policy = "abort",
    const std::string& region_prefetch_stats_file = "",
    const bool deterministic_logical_page_processing = false) {
  diskann::cout << "Search parameters: #threads: " << num_threads << ", ";
  if (beamwidth <= 0)
    diskann::cout << "beamwidth to be optimized for each L value" << std::flush;
  else
    diskann::cout << " beamwidth: " << beamwidth << std::flush;
  if (search_io_limit == std::numeric_limits<_u32>::max())
    diskann::cout << "." << std::endl;
  else
    diskann::cout << ", io_limit: " << search_io_limit << "." << std::endl;

  std::string warmup_query_file = index_path_prefix + "_sample_data.bin";

  // load query bin
  T*        query = nullptr;
  unsigned* gt_ids = nullptr;
  float*    gt_dists = nullptr;
  size_t    query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
  diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim,
                               query_aligned_dim);

  bool calc_recall_flag = false;
  if (gt_file != std::string("null") && gt_file != std::string("NULL") &&
      file_exists(gt_file)) {
    diskann::load_truthset(gt_file, gt_ids, gt_dists, gt_num, gt_dim);
    if (gt_num != query_num) {
      diskann::cout
          << "Error. Mismatch in number of queries and ground truth data"
          << std::endl;
    }
    calc_recall_flag = true;
  }

  // default index (Starling)
  std::shared_ptr<AlignedFileReader> reader = nullptr;
  std::shared_ptr<diskann::PQFlashIndex<T>> _pFlashIndex;

  // use disk separation
  std::shared_ptr<FileIOManager> fio_reader = nullptr;
  std::shared_ptr<diskann::DecoIndex<T>> _decoIndex;

  int res;
  if (deco_impl) {
    fio_reader.reset(new FileIOManager());
    _decoIndex = std::make_shared<diskann::DecoIndex<T>>(fio_reader, metric, use_graph_rep_index, sector_len, enable_region_layout);
    res = _decoIndex->load(num_threads, index_path_prefix.c_str(), pq_path_prefix.c_str(),
                           disk_file_path, graph_rep_index_prefix, disk_graph_prefix);
  }  else {
    reader.reset(new LinuxAlignedFileReader());
    _pFlashIndex = std::make_shared<diskann::PQFlashIndex<T>>(reader, use_page_search, metric, sector_len);
    res = _pFlashIndex->load(num_threads, index_path_prefix.c_str(), pq_path_prefix.c_str(), disk_file_path);
  }

  if (res != 0) {
    return res;
  }
  if (deco_impl) {
    _decoIndex->enable_transition_trace(collect_transition_trace);
    _decoIndex->enable_trace_v2(enable_trace_v2, trace_v2_output_dir);
    _decoIndex->enable_region_io_trace(enable_region_io_trace, region_io_trace_dir, region_file);
    _decoIndex->configure_region_prefetch(enable_region_prefetch, region_file, region_prefetch_size,
                                          region_cache_limit_bytes_per_query, region_buffer_policy,
                                          region_cache_overflow_policy);
    _decoIndex->configure_global_qd_control(enable_global_qd_control, global_qd_cap, qd_control_trace);
    _decoIndex->configure_deterministic_logical_page_processing(deterministic_logical_page_processing);
  }

  // load in-memory navigation graph
  if (deco_impl) {
    if (mem_L) {
      _decoIndex->load_mem_index(metric, query_dim, mem_index_path, num_threads, mem_L);
    }
    if (mem_graph_use_ratio > 0) {
      std::vector<unsigned> tags;
      if (mem_L) {
        std::string tags_path = mem_sample_path + "_ids.bin";
        _decoIndex->load_sampled_tags(tags_path, tags);
      }
      _decoIndex->load_mem_graph(disk_graph_prefix, tags, mem_graph_use_ratio);
      _decoIndex->load_mem_emb(tags, mem_emb_use_ratio);
    }
  } else {
    if (mem_L) {
      _pFlashIndex->load_mem_index(metric, query_dim, mem_index_path, num_threads, mem_L);
    }
    // cache bfs levels
    std::vector<uint32_t> node_list;
    diskann::cout << "Caching " << num_nodes_to_cache
                  << " BFS nodes around medoid(s)" << std::endl;
    if (num_nodes_to_cache > 0){
      _pFlashIndex->generate_cache_list_from_sample_queries(
          warmup_query_file, 15, 6, num_nodes_to_cache, num_threads, node_list, use_page_search, mem_L);
      _pFlashIndex->load_cache_list(node_list);
    }
    node_list.clear();
    node_list.shrink_to_fit();
  }

  omp_set_num_threads(num_threads);

  diskann::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  diskann::cout.precision(2);

  std::string recall_string = "Recall@" + std::to_string(recall_at);
  diskann::cout << std::setw(4) << "L"
                << std::setw(4) << "BW"
                << std::setw(9) << "QPS"
                << std::setw(9) << "Mean Ltc"
                << std::setw(9) << "P50"
                << std::setw(9) << "P95"
                << std::setw(9) << "P99"
                << std::setw(9) << "999 Ltc" 
                << std::setw(9) << "Graph IO"
                << std::setw(9) << "Emb IO"
                << std::setw(9) << "Ext Cmp"
                << std::setw(9) << "PQ Cmp"
                << std::setw(9) << "Pre(T)"
                << std::setw(9) << "Disp(T)"
                << std::setw(9) << "Read(T)"
                << std::setw(9) << "Page(T)"
                << std::setw(9) << "Cache(T)"
                << std::setw(9) << "DiskN(T)"
                << std::setw(9) << "Post(T)"
                << std::setw(9) << "Mem(MB)";
  if (calc_recall_flag) {
    diskann::cout << std::setw(10) << recall_string << std::endl;
  } else
    diskann::cout << std::endl;
  diskann::cout
      << "==============================================================="
         "======================================================="
      << std::endl;

  std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
  std::vector<std::vector<float>>    query_result_dists(Lvec.size());

  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
    _u64 L = Lvec[test_id];

    if (L < recall_at) {
      diskann::cout << "Ignoring search with L:" << L
                    << " since it's smaller than K:" << recall_at << std::endl;
      continue;
    }

    if (beamwidth <= 0) {
      diskann::cout << "beamwidth <= 0" << std::endl;
      exit(-1);
    }

    query_result_ids[test_id].resize(recall_at * query_num);
    query_result_dists[test_id].resize(recall_at * query_num);

    auto stats = new diskann::QueryStats[query_num];

    std::vector<uint64_t> query_result_ids_64(recall_at * query_num);
    auto                  s = std::chrono::high_resolution_clock::now();

    // Using branching outside the for loop instead of inside and 
    // std::function/std::mem_fn for less switching and function calling overhead
    if (deco_impl && !use_graph_rep_index) {
      // Gorgeous with Starling layout
      _decoIndex->page_search(
        query, query_num, query_aligned_dim, recall_at, mem_L, L, 
        query_result_ids_64, query_result_dists[test_id],
        beamwidth, search_io_limit, pq_ratio, emb_search_ratio, stats);
    } else if (deco_impl && use_graph_rep_index) {
      // Gorgeous graph replicated
      _decoIndex->page_search_dup_graph(
        query, query_num, query_aligned_dim, recall_at, mem_L, L, 
        query_result_ids_64, query_result_dists[test_id],
        beamwidth, search_io_limit, pq_ratio, emb_search_ratio, stats);
    } else {
      if (use_page_search) {
        // Starling
#pragma omp parallel for schedule(dynamic, 1)
        for (_s64 i = 0; i < (int64_t) query_num; i++) {
          _pFlashIndex->page_search(
              query + (i * query_aligned_dim), recall_at, mem_L, L,
              query_result_ids_64.data() + (i * recall_at),
              query_result_dists[test_id].data() + (i * recall_at),
              beamwidth, search_io_limit, use_ratio, stats + i);
        }
      } else {
        // DiskANN (starling without page search)
#pragma omp parallel for schedule(dynamic, 1)
        for (_s64 i = 0; i < (int64_t) query_num; i++) {
          _pFlashIndex->cached_beam_search(
              query + (i * query_aligned_dim), recall_at, L,
              query_result_ids_64.data() + (i * recall_at),
              query_result_dists[test_id].data() + (i * recall_at),
              beamwidth, search_io_limit, stats + i, mem_L);
        }
      }
    }
    auto                          e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    float qps = (1.0 * query_num) / (1.0 * diff.count());

    diskann::convert_types<uint64_t, uint32_t>(query_result_ids_64.data(),
                                               query_result_ids[test_id].data(),
                                               query_num, recall_at);

    uint64_t warmup_cnt = query_num > 20 ? 20 : 0;

    int mean_latency = (int)(diskann::get_mean_stats<float>(
        stats, query_num, warmup_cnt,
        [](const diskann::QueryStats& stats) { return stats.total_us; }));

    int latency_p50 = (int)(diskann::get_percentile_stats<float>(
        stats, query_num, warmup_cnt, 0.50,
        [](const diskann::QueryStats& stats) { return stats.total_us; }));

    int latency_p95 = (int)(diskann::get_percentile_stats<float>(
        stats, query_num, warmup_cnt, 0.95,
        [](const diskann::QueryStats& stats) { return stats.total_us; }));

    int latency_p99 = (int)(diskann::get_percentile_stats<float>(
        stats, query_num, warmup_cnt, 0.99,
        [](const diskann::QueryStats& stats) { return stats.total_us; }));

    int latency_999 = (int)(diskann::get_percentile_stats<float>(
        stats, query_num, warmup_cnt, 0.999,
        [](const diskann::QueryStats& stats) { return stats.total_us; }));

    auto mean_ios = diskann::get_mean_stats<unsigned>(
        stats, query_num, warmup_cnt,
        [](const diskann::QueryStats& stats) { return stats.n_ios; });

    auto mean_emb_ios = diskann::get_mean_stats<unsigned>(
        stats, query_num, warmup_cnt,
        [](const diskann::QueryStats& stats) { return stats.n_emb_ios; });

    auto mean_ext_cmps = diskann::get_mean_stats<float>(
        stats, query_num, warmup_cnt,
        [](const diskann::QueryStats& stats) { return stats.n_ext_cmps; });

    auto mean_cmps = diskann::get_mean_stats<float>(
        stats, query_num, warmup_cnt,
        [](const diskann::QueryStats& stats) { return stats.n_cmps; });

    auto mean_preprocess = diskann::get_mean_stats<float>(
        stats, query_num, warmup_cnt,
        [](const diskann::QueryStats& stats) { return stats.preprocess_us; });

    auto mean_postprocess = diskann::get_mean_stats<float>(
        stats, query_num, warmup_cnt,
        [](const diskann::QueryStats& stats) { return stats.postprocess_us; });

    auto mean_dispatch_time = diskann::get_mean_stats<float>(
        stats, query_num, warmup_cnt,
        [](const diskann::QueryStats& stats) { return stats.dispatch_us; });

    auto mean_read_disk_time = diskann::get_mean_stats<float>(
        stats, query_num, warmup_cnt,
        [](const diskann::QueryStats& stats) { return stats.read_disk_us; });

    auto mean_page_proc_time = diskann::get_mean_stats<float>(
        stats, query_num, warmup_cnt,
        [](const diskann::QueryStats& stats) { return stats.page_proc_us; });

    auto mean_cache_proc_time = diskann::get_mean_stats<float>(
        stats, query_num, warmup_cnt,
        [](const diskann::QueryStats& stats) { return stats.cache_proc_us; });

    auto mean_disk_proc_time = diskann::get_mean_stats<float>(
        stats, query_num, warmup_cnt,
        [](const diskann::QueryStats& stats) { return stats.disk_proc_us; });

    float recall = 0;
    if (calc_recall_flag) {
      recall = diskann::calculate_recall(query_num, gt_ids, gt_dists, gt_dim,
                                         query_result_ids[test_id].data(),
                                         recall_at, recall_at);
    }

    diskann::cout << std::setw(4) << L
                  << std::setw(4) << beamwidth
                  << std::setw(9) << qps
                  << std::setw(9) << mean_latency
                  << std::setw(9) << latency_p50
                  << std::setw(9) << latency_p95
                  << std::setw(9) << latency_p99
                  << std::setw(9) << latency_999
                  << std::setw(9) << mean_ios
                  << std::setw(9) << mean_emb_ios
                  << std::setw(9) << mean_ext_cmps
                  << std::setw(9) << mean_cmps
                  << std::setw(9) << mean_preprocess
                  << std::setw(9) << mean_dispatch_time
                  << std::setw(9) << mean_read_disk_time
                  << std::setw(9) << mean_page_proc_time
                  << std::setw(9) << mean_cache_proc_time
                  << std::setw(9) << mean_disk_proc_time
                  << std::setw(9) << mean_postprocess
                  << std::setw(9) << getProcessPeakRSS();
    if (calc_recall_flag) {
      diskann::cout << std::setw(10) << recall << std::endl;
    } else
      diskann::cout << std::endl;

    if (!region_prefetch_stats_file.empty()) {
      const bool append = test_id > 0;
      std::ofstream pf(region_prefetch_stats_file, append ? std::ios::app : std::ios::out);
      if (!append) {
        pf << "query_id,L,region_prefetch_enabled,overflow,logical_graph_ios,device_read_submits,"
           << "device_read_bytes,device_4kb_reads,device_16kb_reads,region_first_triggers,"
           << "region_cache_hits,loaded_regions,additional_pages_read,additional_pages_used,"
           << "unused_additional_pages,duplicate_load_attempts,same_region_pending_frontier_count,"
           << "pending_waiter_registered_count,pending_waiter_processed_count,pending_waiter_duplicate_count,"
           << "requested_page_parse_count,prefetched_only_page_parse_count,pending_region_leftover_at_query_end,"
           << "waiter_registered_but_not_processed,duplicate_region_submission_count,"
           << "duplicate_logical_page_process_count,region_slice_byte_mismatch_count,"
           << "cache_entries_released_at_query_end,cache_leftover_after_cleanup,"
           << "cache_peak_bytes,cache_allocations,"
           << "cache_frees,region_lookup_us,region_cache_manage_us,"
           << "deterministic_enabled,det_logical_page_requests,det_seq_assigned,det_ready,"
           << "det_processed,det_duplicate_requests,det_duplicate_processes,det_queue_inserts,"
           << "det_queue_max_depth,det_queue_leftover,det_requested_but_not_processed,"
           << "det_processed_without_request,det_physical_io_submissions,det_logical_to_physical_coalescing,"
           << "total_us,read_disk_us,disk_proc_us,GraphIO,EmbIO\n";
      }
      for (size_t qi = 0; qi < query_num; qi++) {
        const auto &st = stats[qi];
        pf << qi << "," << L << "," << st.region_prefetch_enabled << ","
           << st.region_prefetch_overflow << "," << st.region_logical_graph_ios << ","
           << st.region_device_read_submits << "," << st.region_device_read_bytes << ","
           << st.region_device_4kb_reads << "," << st.region_device_16kb_reads << ","
           << st.region_first_triggers << "," << st.region_cache_hits << ","
           << st.region_loaded_regions << "," << st.region_additional_pages_read << ","
           << st.region_additional_pages_used << "," << st.region_unused_additional_pages << ","
           << st.region_duplicate_load_attempts << ","
           << st.region_same_region_pending_frontier_count << ","
           << st.region_pending_waiter_registered_count << ","
           << st.region_pending_waiter_processed_count << ","
           << st.region_pending_waiter_duplicate_count << ","
           << st.region_requested_page_parse_count << ","
           << st.region_prefetched_only_page_parse_count << ","
           << st.region_pending_region_leftover_at_query_end << ","
           << st.region_waiter_registered_but_not_processed << ","
           << st.region_duplicate_region_submission_count << ","
           << st.region_duplicate_logical_page_process_count << ","
           << st.region_slice_byte_mismatch_count << ","
           << st.region_cache_entries_released_at_query_end << ","
           << st.region_cache_leftover_after_cleanup << ","
           << st.region_cache_peak_bytes << ","
           << st.region_cache_allocations << "," << st.region_cache_frees << ","
           << st.region_lookup_us << "," << st.region_cache_manage_us << ","
           << st.deterministic_logical_page_processing_enabled << ","
           << st.deterministic_logical_page_request_count << ","
           << st.deterministic_logical_request_seq_assigned_count << ","
           << st.deterministic_logical_page_ready_count << ","
           << st.deterministic_logical_page_processed_count << ","
           << st.deterministic_logical_page_duplicate_request_count << ","
           << st.deterministic_logical_page_duplicate_process_count << ","
           << st.deterministic_queue_insert_count << ","
           << st.deterministic_queue_max_depth << ","
           << st.deterministic_queue_leftover_at_query_end << ","
           << st.deterministic_requested_but_not_processed_count << ","
           << st.deterministic_processed_without_request_count << ","
           << st.deterministic_physical_io_submission_count << ","
           << st.deterministic_logical_to_physical_coalescing_count << ","
           << st.total_us << "," << st.read_disk_us << "," << st.disk_proc_us << ","
           << st.n_ios << "," << st.n_emb_ios << "\n";
      }
    }

#ifdef ENABLE_REPLICA_REDUNDANCY_STATS
    mkdir("logs", 0755);
    const std::string replica_csv_path = "logs/replica_redundancy_stats.csv";
    std::ofstream replica_csv(replica_csv_path,
                              test_id == 0 ? std::ios::out : std::ios::app);
    if (test_id == 0) {
      replica_csv << "query_id,L,graph_page_ios,first_read_pages,repeated_page_reads,"
                  << "replica_attempts,replica_attempts_on_first_read_pages,"
                  << "unique_replica_adjacencies,all_replica_duplicates,"
                  << "cross_page_replica_duplicates,replica_to_replica_duplicates,"
                  << "owner_to_replica_duplicates,pages_with_any_replica_duplicate,"
                  << "fully_redundant_replica_pages,replica_redundancy,duplicate_replicas_per_graph_io,cross_page_replica_redundancy,cross_page_duplicates_per_graph_io\n";
    }

    uint64_t total_graph_page_ios = 0;
    uint64_t total_first_read_pages = 0;
    uint64_t total_replica_attempts_on_first_read_pages = 0;
    uint64_t total_cross_page_replica_duplicates = 0;
    uint64_t total_replica_to_replica_duplicates = 0;
    uint64_t total_pages_with_any_replica_duplicate = 0;
    uint64_t total_fully_redundant_replica_pages = 0;
    const uint64_t stats_begin = std::min<uint64_t>(warmup_cnt, query_num);

    for (uint64_t q = 0; q < query_num; q++) {
      const auto &st = stats[q];
      replica_csv << q << "," << L << ","
                  << st.replica_graph_page_ios << ","
                  << st.replica_first_read_pages << ","
                  << st.replica_repeated_page_reads << ","
                  << st.replica_attempts << ","
                  << st.replica_attempts_on_first_read_pages << ","
                  << st.unique_replica_adjacencies << ","
                  << st.all_replica_duplicates << ","
                  << st.cross_page_replica_duplicates << ","
                  << st.replica_to_replica_duplicates << ","
                  << st.owner_to_replica_duplicates << ","
                  << st.pages_with_any_replica_duplicate << ","
                  << st.fully_redundant_replica_pages << ","
                  << st.replica_redundancy << ","
                  << st.duplicate_replicas_per_graph_io << ","
                  << (st.replica_attempts_on_first_read_pages == 0 ? 0.0 : static_cast<double>(st.cross_page_replica_duplicates) / static_cast<double>(st.replica_attempts_on_first_read_pages)) << ","
                  << (st.replica_graph_page_ios == 0 ? 0.0 : static_cast<double>(st.cross_page_replica_duplicates) / static_cast<double>(st.replica_graph_page_ios)) << "\n";

      if (q >= stats_begin) {
        total_graph_page_ios += st.replica_graph_page_ios;
        total_first_read_pages += st.replica_first_read_pages;
        total_replica_attempts_on_first_read_pages +=
            st.replica_attempts_on_first_read_pages;
        total_cross_page_replica_duplicates +=
            st.cross_page_replica_duplicates;
        total_replica_to_replica_duplicates +=
            st.replica_to_replica_duplicates;
        total_pages_with_any_replica_duplicate +=
            st.pages_with_any_replica_duplicate;
        total_fully_redundant_replica_pages +=
            st.fully_redundant_replica_pages;
      }
    }

    auto safe_div = [](uint64_t num, uint64_t den) -> double {
      return den == 0 ? 0.0 : static_cast<double>(num) / static_cast<double>(den);
    };
    const double replica_duplicate_rate =
        safe_div(total_replica_to_replica_duplicates,
                 total_replica_attempts_on_first_read_pages);
    const double cross_page_duplicate_rate =
        safe_div(total_cross_page_replica_duplicates,
                 total_replica_attempts_on_first_read_pages);
    const double duplicates_per_graph_io =
        safe_div(total_replica_to_replica_duplicates, total_graph_page_ios);
    const double effective_replica_utilization = 1.0 - replica_duplicate_rate;
    const double pages_with_any_duplicate_rate =
        safe_div(total_pages_with_any_replica_duplicate, total_first_read_pages);
    const double fully_redundant_page_rate =
        safe_div(total_fully_redundant_replica_pages, total_first_read_pages);

    diskann::cout << "Replica redundancy stats L=" << L
                  << " csv=" << replica_csv_path
                  << " replica_duplicate_rate=" << replica_duplicate_rate
                  << " cross_page_duplicate_rate=" << cross_page_duplicate_rate
                  << " duplicates_per_graph_io=" << duplicates_per_graph_io
                  << " pages_with_any_duplicate_rate="
                  << pages_with_any_duplicate_rate
                  << " effective_replica_utilization="
                  << effective_replica_utilization
                  << " fully_redundant_page_rate="
                  << fully_redundant_page_rate << std::endl;
#endif

    delete[] stats;
  }

  diskann::cout << "Done searching. Now saving results " << std::endl;

  _u64 test_id = 0;
  for (auto L : Lvec) {
    if (L < recall_at)
      continue;

    std::string cur_result_path =
        result_output_prefix + "_" + std::to_string(L) + "_idx_uint32.bin";
    diskann::save_bin<_u32>(cur_result_path, query_result_ids[test_id].data(),
                            query_num, recall_at);

    cur_result_path =
        result_output_prefix + "_" + std::to_string(L) + "_dists_float.bin";
    diskann::save_bin<float>(cur_result_path,
                             query_result_dists[test_id++].data(), query_num,
                             recall_at);
  }

  diskann::aligned_free(query);
  delete[] gt_ids;
  delete[] gt_dists;
  return 0;
}

int main(int argc, char** argv) {
  std::string data_type, dist_fn, index_path_prefix, pq_path_prefix, result_path_prefix,
      query_file, gt_file, disk_file_path, mem_index_path, mem_sample_path;
  std::string disk_graph_prefix, graph_rep_index_prefix;
  unsigned              num_threads, K, W, num_nodes_to_cache, search_io_limit;
  unsigned              mem_L;
  std::vector<unsigned> Lvec;
  bool                  use_page_search = true;
  float                 use_ratio = 1.0;
  float                 pq_ratio = 1.0;
  bool deco_impl = false;
  bool use_graph_rep_index = false;
  bool enable_region_layout = false;
  bool enable_region_physical_reorder = false;
  bool collect_transition_trace = false;
  unsigned enable_trace_v2 = 0;
  std::string trace_v2_output_dir = "logs";
  bool enable_region_io_trace = false;
  std::string region_io_trace_dir = "logs/region_io_trace";
  std::string region_file = "";
  bool enable_global_qd_control = false;
  unsigned global_qd_cap = 0;
  bool qd_control_trace = false;
  bool enable_region_prefetch = false;
  unsigned region_prefetch_size = 4;
  std::string region_buffer_policy = "query_lifetime";
  uint64_t region_cache_limit_bytes_per_query = 16ULL * 1024ULL * 1024ULL;
  std::string region_cache_overflow_policy = "abort";
  std::string region_prefetch_stats_file = "";
  bool deterministic_logical_page_processing = false;
  float mem_graph_use_ratio = 0.0;
  float mem_emb_use_ratio = 0.0;
  float emb_search_ratio = 1.0;
  _u64 sector_len;

  po::options_description desc{"Arguments"};
  try {
    desc.add_options()("help,h", "Print information on arguments");
    desc.add_options()("data_type",
                       po::value<std::string>(&data_type)->required(),
                       "data type <int8/uint8/float>");
    desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                       "distance function <l2/mips/fast_l2>");
    desc.add_options()("index_path_prefix",
                       po::value<std::string>(&index_path_prefix)->required(),
                       "Path prefix to the index");
    desc.add_options()("pq_path_prefix",
                       po::value<std::string>(&pq_path_prefix)->required(),
                       "Path prefix to the pq");
    desc.add_options()("result_path",
                       po::value<std::string>(&result_path_prefix)->required(),
                       "Path prefix for saving results of the queries");
    desc.add_options()("query_file",
                       po::value<std::string>(&query_file)->required(),
                       "Query file in binary format");
    desc.add_options()(
        "gt_file",
        po::value<std::string>(&gt_file)->default_value(std::string("null")),
        "ground truth file for the queryset");
    desc.add_options()("recall_at,K", po::value<uint32_t>(&K)->required(),
                       "Number of neighbors to be returned");
    desc.add_options()("search_list,L",
                       po::value<std::vector<unsigned>>(&Lvec)->multitoken(),
                       "List of L values of search");
    desc.add_options()("beamwidth,W", po::value<uint32_t>(&W)->default_value(2),
                       "Beamwidth for search. Set 0 to optimize internally.");
    desc.add_options()(
        "num_nodes_to_cache",
        po::value<uint32_t>(&num_nodes_to_cache)->default_value(0),
        "Number nodes cached. DiskANN's method.");
    desc.add_options()("search_io_limit",
                       po::value<uint32_t>(&search_io_limit)
                           ->default_value(std::numeric_limits<_u32>::max()),
                       "Max #IOs for search");
    desc.add_options()("sector_len",
                      po::value<_u64>(&sector_len)
                          ->default_value(std::numeric_limits<_u64>::max()),
                      "sector len");
    desc.add_options()(
        "num_threads,T",
        po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
        "Number of threads used for building index (defaults to "
        "omp_get_num_procs())");
    desc.add_options()("mem_L", po::value<unsigned>(&mem_L)->default_value(0),
                       "The L of the in-memory navigation graph while searching. Use 0 to disable");
    desc.add_options()("use_page_search", po::value<bool>(&use_page_search)->default_value(1),
                       "Use 1 for page search (default), 0 for DiskANN beam search");
    desc.add_options()("use_ratio", po::value<float>(&use_ratio)->default_value(1.0f),
                       "The percentage of how many vectors in a page to search each time");
    desc.add_options()("disk_file_path", po::value<std::string>(&disk_file_path)->required(),
                       "The path of the disk file (_disk.index in the original DiskANN)");
    desc.add_options()("disk_graph_prefix", po::value<std::string>(&disk_graph_prefix)->required(),
                       "graph prefix");
    desc.add_options()("graph_rep_index_prefix", po::value<std::string>(&graph_rep_index_prefix)->required(),
                       "graph cache index prefix");
    desc.add_options()("mem_index_path", po::value<std::string>(&mem_index_path)->default_value(""),
                       "The prefix path of the mem_index");
    desc.add_options()("mem_sample_path", po::value<std::string>(&mem_sample_path)->default_value(""),
                       "The mem_sample_path path of the mem_sample_path");
    desc.add_options()("deco_impl", po::value<bool>(&deco_impl)->default_value(0),
                       "whether use disk separation");
    desc.add_options()("pq_ratio", po::value<float>(&pq_ratio)->default_value(1.0f),
                       "The percentage of how many vectors in a page to search each time");
    desc.add_options()("use_graph_rep_index", po::value<bool>(&use_graph_rep_index)->default_value(0),
                       "whether use graph cache index");
    desc.add_options()("enable_region_layout", po::value<bool>(&enable_region_layout)->default_value(0),
                       "deprecated alias for enable_region_physical_reorder");
    desc.add_options()("enable_region_physical_reorder", po::value<bool>(&enable_region_physical_reorder)->default_value(0),
                       "use owner node to physical page mapping for region-ordered graph-replicated index");
    desc.add_options()("collect_transition_trace", po::value<bool>(&collect_transition_trace)->default_value(0),
                       "whether collect graph traversal trace to logs/search_trace.csv");
    desc.add_options()("enable_trace_v2", po::value<unsigned>(&enable_trace_v2)->default_value(0),
                       "whether collect semantic trace v2 for incoming-direction analysis");
    desc.add_options()("trace_v2_output_dir", po::value<std::string>(&trace_v2_output_dir)->default_value("logs"),
                       "directory for expansion_trace_v2.csv and edge_scan_trace_v2.csv");
    desc.add_options()("enable_region_io_trace", po::value<bool>(&enable_region_io_trace)->default_value(0),
                       "collect low-overhead graph I/O batch locality/timing trace");
    desc.add_options()("region_io_trace_dir", po::value<std::string>(&region_io_trace_dir)->default_value("logs/region_io_trace"),
                       "directory for graph I/O batch trace outputs");
    desc.add_options()("region_file", po::value<std::string>(&region_file)->default_value(""),
                       "optional regions.tsv for annotating graph I/O batches");
    desc.add_options()("enable_global_qd_control", po::value<bool>(&enable_global_qd_control)->default_value(0),
                       "enable experimental process-wide graph-read request QD controller");
    desc.add_options()("global_qd_cap", po::value<unsigned>(&global_qd_cap)->default_value(0),
                       "process-wide graph-read request cap for QD controller");
    desc.add_options()("qd_control_trace", po::value<bool>(&qd_control_trace)->default_value(0),
                       "record QD controller wait/split counters in region I/O trace");
    desc.add_options()("enable_region_prefetch", po::value<bool>(&enable_region_prefetch)->default_value(0),
                       "enable query-lifetime R=4 online Region prefetch prototype");
    desc.add_options()("region_prefetch_size", po::value<unsigned>(&region_prefetch_size)->default_value(4),
                       "Region prefetch size in 4KB pages; task 3A supports 4");
    desc.add_options()("region_buffer_policy", po::value<std::string>(&region_buffer_policy)->default_value("query_lifetime"),
                       "Region buffer lifetime policy; task 3A supports query_lifetime");
    desc.add_options()("region_cache_limit_bytes", po::value<uint64_t>(&region_cache_limit_bytes_per_query)->default_value(16ULL * 1024ULL * 1024ULL),
                       "per-query Region cache safety limit in bytes");
    desc.add_options()("region_cache_overflow_policy", po::value<std::string>(&region_cache_overflow_policy)->default_value("abort"),
                       "Region cache overflow policy; task 3A supports abort");
    desc.add_options()("region_prefetch_stats_file", po::value<std::string>(&region_prefetch_stats_file)->default_value(""),
                       "optional per-query Region prefetch stats CSV");
    desc.add_options()("deterministic_logical_page_processing", po::value<bool>(&deterministic_logical_page_processing)->default_value(0),
                       "process completed graph pages in stable logical request order; default off");
    desc.add_options()("mem_graph_use_ratio", po::value<float>(&mem_graph_use_ratio)->default_value(1.0f),
                       "ratio of using memory graph");
    desc.add_options()("mem_emb_use_ratio", po::value<float>(&mem_emb_use_ratio)->default_value(1.0f),
                       "ratio of using memory emb");
    desc.add_options()("emb_search_ratio", po::value<float>(&emb_search_ratio)->default_value(1.0f),
                       "ratio of embedding search when using memory graph");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc;
      return 0;
    }
    po::notify(vm);
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return -1;
  }

  diskann::Metric metric;
  if (dist_fn == std::string("mips")) {
    metric = diskann::Metric::INNER_PRODUCT;
  } else if (dist_fn == std::string("l2")) {
    metric = diskann::Metric::L2;
  } else if (dist_fn == std::string("cosine")) {
    metric = diskann::Metric::COSINE;
  } else {
    std::cout << "Unsupported distance function. Currently only L2/ Inner "
                 "Product/Cosine are supported."
              << std::endl;
    return -1;
  }

  if (use_ratio < 0 || use_ratio > 1.0f) {
    std::cout << "use_ratio should be in the range [0, 1] (inclusive)." << std::endl;
    return -1;
  }

  if ((data_type != std::string("float")) &&
      (metric == diskann::Metric::INNER_PRODUCT)) {
    std::cout << "Currently support only floating point data for Inner Product."
              << std::endl;
    return -1;
  }

  if (!use_page_search && deco_impl) {
    std::cout << "[Warning] deco_impl not support diskann." << std::endl;
  }
  if (mem_graph_use_ratio > 1.0 || mem_graph_use_ratio < 0) {
    std::cout << "mem graph use ratio should betweem 0 and 1." << std::endl;
    return -1;
  }
  if (emb_search_ratio > 1.0 || emb_search_ratio < 0) {
    std::cout << "emb_search_ratio should betweem 0 and 1." << std::endl;
    return -1;
  }
  if (mem_emb_use_ratio > 1.0 || mem_emb_use_ratio < 0) {
    std::cout << "mem_emb_use_ratio should betweem 0 and 1." << std::endl;
    return -1;
  }
  std::cout << data_type << std::endl;

  // try {
    if (data_type == std::string("float"))
      return search_disk_index<float>(
          metric, index_path_prefix, pq_path_prefix, mem_index_path, mem_sample_path, result_path_prefix,
          query_file, gt_file, disk_file_path, disk_graph_prefix, graph_rep_index_prefix,
          num_threads, K, W, num_nodes_to_cache, search_io_limit, Lvec, mem_L, sector_len,
          use_page_search, use_ratio, pq_ratio, deco_impl,
          use_graph_rep_index, (enable_region_layout || enable_region_physical_reorder), mem_graph_use_ratio, mem_emb_use_ratio, emb_search_ratio, collect_transition_trace, enable_trace_v2, trace_v2_output_dir, enable_region_io_trace, region_io_trace_dir, region_file, enable_global_qd_control, global_qd_cap, qd_control_trace, enable_region_prefetch, region_prefetch_size, region_buffer_policy, region_cache_limit_bytes_per_query, region_cache_overflow_policy, region_prefetch_stats_file, deterministic_logical_page_processing);
    else if (data_type == std::string("int8"))
      return search_disk_index<int8_t>(
          metric, index_path_prefix, pq_path_prefix, mem_index_path, mem_sample_path, result_path_prefix,
          query_file, gt_file, disk_file_path, disk_graph_prefix, graph_rep_index_prefix,
          num_threads, K, W, num_nodes_to_cache, search_io_limit, Lvec, mem_L, sector_len,
          use_page_search, use_ratio, pq_ratio, deco_impl,
          use_graph_rep_index, (enable_region_layout || enable_region_physical_reorder), mem_graph_use_ratio, mem_emb_use_ratio, emb_search_ratio, collect_transition_trace, enable_trace_v2, trace_v2_output_dir, enable_region_io_trace, region_io_trace_dir, region_file, enable_global_qd_control, global_qd_cap, qd_control_trace, enable_region_prefetch, region_prefetch_size, region_buffer_policy, region_cache_limit_bytes_per_query, region_cache_overflow_policy, region_prefetch_stats_file, deterministic_logical_page_processing);
    else if (data_type == std::string("uint8"))
      return search_disk_index<uint8_t>(
          metric, index_path_prefix, pq_path_prefix, mem_index_path, mem_sample_path, result_path_prefix,
          query_file, gt_file, disk_file_path, disk_graph_prefix, graph_rep_index_prefix,
          num_threads, K, W, num_nodes_to_cache, search_io_limit, Lvec, mem_L, sector_len,
          use_page_search, use_ratio, pq_ratio, deco_impl,
          use_graph_rep_index, (enable_region_layout || enable_region_physical_reorder), mem_graph_use_ratio, mem_emb_use_ratio, emb_search_ratio, collect_transition_trace, enable_trace_v2, trace_v2_output_dir, enable_region_io_trace, region_io_trace_dir, region_file, enable_global_qd_control, global_qd_cap, qd_control_trace, enable_region_prefetch, region_prefetch_size, region_buffer_policy, region_cache_limit_bytes_per_query, region_cache_overflow_policy, region_prefetch_stats_file, deterministic_logical_page_processing);
    else {
      std::cerr << "Unsupported data type. Use float or int8 or uint8"
                << std::endl;
      return -1;
    }
  // } catch (const std::exception& e) {
  //   std::cout << std::string(e.what()) << std::endl;
  //   diskann::cerr << "Index search failed." << std::endl;
  //   return -1;
  // }
}
