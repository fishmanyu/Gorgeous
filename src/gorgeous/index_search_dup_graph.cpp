#include <immintrin.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <numeric>
#include <map>
#include <sstream>
#include <time.h>
#include "logger.h"
#include "percentile_stats.h"
#include "replica_redundancy_stats.h"
#include "deco_index.h"
#include "timer.h"

namespace diskann {

  struct CacheNode {
    unsigned id;
    unsigned nb_size;
    unsigned* node_nbrs;
    unsigned parent_id;
    bool is_replica;
    unsigned retset_position;

    CacheNode(unsigned id, unsigned nb_size, unsigned* node_nbrs, unsigned parent_id = INF,
              bool is_replica = false, unsigned retset_position = INF) {
      this->id = id;
      this->nb_size = nb_size;
      this->node_nbrs = node_nbrs;
      this->parent_id = parent_id;
      this->is_replica = is_replica;
      this->retset_position = retset_position;
    }
  };

  struct TransitionTraceRow {
    size_t query_id;
    unsigned expand_order;
    unsigned parent;
    unsigned current;
    unsigned neighbor;
    bool accepted;
    float pq_dist;
    bool is_replica;
    bool graph_io;
    unsigned depth_from_entry;
  };

  enum TraceV2Kind : unsigned {
    TRACE_V2_KIND_NONE = 0,
    TRACE_V2_KIND_ENTRY = 1,
    TRACE_V2_KIND_OWNER = 2,
    TRACE_V2_KIND_REPLICA = 3,
    TRACE_V2_KIND_MEM = 4,
    TRACE_V2_KIND_REPLICA_TRIGGERED = 5
  };

  static const char *trace_v2_kind_name(unsigned kind) {
    switch (kind) {
      case TRACE_V2_KIND_ENTRY: return "entry";
      case TRACE_V2_KIND_OWNER: return "owner";
      case TRACE_V2_KIND_REPLICA: return "replica";
      case TRACE_V2_KIND_MEM: return "mem";
      case TRACE_V2_KIND_REPLICA_TRIGGERED: return "replica_triggered";
      default: return "none";
    }
  }

  struct ExpansionTraceV2Row {
    size_t query_id;
    unsigned expansion_event_id;
    unsigned expansion_order;
    unsigned current;
    unsigned arrival_parent;
    unsigned arrival_kind;
    unsigned current_first_insert_event;
    unsigned current_first_insert_order;
    float current_distance;
    unsigned expansion_kind;
    unsigned triggering_page_owner;
    bool graph_io;
    unsigned physical_page_id;
    unsigned current_adjacency_source;
    bool current_adjacency_from_replica;
    unsigned retset_position;
    bool query_finished_flag;
  };

  struct EdgeScanTraceV2Row {
    size_t query_id;
    unsigned expansion_event_id;
    unsigned adjacency_scan_id;
    unsigned trigger_current;
    unsigned adjacency_source;
    unsigned page_owner;
    unsigned scan_kind;
    unsigned neighbor;
    bool accepted;
    bool first_accepted;
    unsigned neighbor_arrival_parent_after_scan;
    float neighbor_distance;
    bool graph_io;
    bool is_replica;
    unsigned physical_page_id;
  };
  struct GraphIORequestTraceRow {
    size_t query_id;
    unsigned thread_id;
    unsigned batch_id;
    unsigned request_index;
    unsigned owner_node_id;
    unsigned physical_page_id;
    uint64_t file_offset;
    unsigned region_id;
  };

  struct GraphIOBatchTraceRow {
    size_t query_id;
    unsigned thread_id;
    unsigned batch_id;
    unsigned batch_size;
    std::string logical_owner_key;
    unsigned owner_min;
    unsigned owner_max;
    unsigned physical_min;
    unsigned physical_max;
    uint64_t address_span;
    double adjacent_gap_mean;
    double adjacent_gap_median;
    double adjacent_gap_p95;
    double adjacent_gap_p99;
    unsigned contiguous_adjacent_pairs;
    double le1_pair_ratio;
    double le4_pair_ratio;
    double le16_pair_ratio;
    double le64_pair_ratio;
    double same_region_pair_ratio;
    double construct_us;
    double prep_us;
    double submit_us;
    double submit_to_first_completion_us;
    double submit_to_all_completion_us;
    double getevents_wait_us;
    double page_process_us;
    double query_read_disk_us;
    double query_total_us;
    unsigned remaining;
    bool saw_first;
    double submit_return_us;
  };

  struct GraphIOQueryTraceRow {
    size_t query_id;
    unsigned thread_id;
    unsigned graph_io_batches;
    unsigned graph_io_requests;
    double submit_us;
    double completion_wait_us;
    double page_process_us;
    double read_disk_us;
    double total_us;
    double qd_control_wait_us;
    unsigned qd_control_wait_events;
    unsigned qd_control_split_submit_groups;
  };

  struct GraphIOTimestampBatchRow {
    size_t query_id;
    unsigned thread_id;
    unsigned logical_batch_id;
    unsigned submit_group_id;
    unsigned batch_size;
    uint64_t submit_before_ns;
    uint64_t submit_after_ns;
    unsigned submitted_count;
    int64_t per_context_inflight_before;
    int64_t per_context_inflight_after_submit;
    int64_t global_inflight_before;
    int64_t global_inflight_after_submit;
    uint64_t first_completion_ns;
    uint64_t last_completion_ns;
    uint64_t first_getevents_begin_ns;
    uint64_t last_getevents_end_ns;
    std::string logical_owner_key;
    std::string physical_page_key;
    std::string request_order_key;
  };

  struct GraphIOTimestampRequestRow {
    uint64_t request_id;
    size_t query_id;
    unsigned thread_id;
    unsigned submit_group_id;
    unsigned request_index;
    unsigned owner_node_id;
    unsigned physical_page_id;
    uint64_t submit_ns;
    uint64_t completion_ns;
    unsigned completion_order;
    int result_code;
  };

  struct GraphIOQDEventRow {
    uint64_t timestamp_ns;
    unsigned thread_id;
    int delta;
    int64_t global_inflight_after;
    int64_t per_context_inflight_after;
    const char *event;
  };

  static inline double region_io_now_us() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::micro>(clock::now().time_since_epoch()).count();
  }

  static inline uint64_t region_io_now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
  }


  // data could be parse streamingly from queue.
  template<typename T>
  void DecoIndex<T>::page_search_dup_graph(
      const T *query_ptr, const _u64 query_num, const _u64 query_aligned_dim, const _u64 k_search, const _u32 mem_L,
      const _u64 l_search, std::vector<_u64>& indices_vec, std::vector<float>& distances_vec,
      const _u64 beam_width, const _u32 io_limit,
      const float pq_filter_ratio, const float emb_search_ratio, QueryStats* stats_ptr) {
    // here are global checks / global data init.
    if (beam_width > MAX_N_SECTOR_READS)
      throw ANNException("Beamwidth can not be higher than MAX_N_SECTOR_READS",
                         -1, __FUNCSIG__, __FILE__, __LINE__);

    uint32_t query_dim = metric == diskann::Metric::INNER_PRODUCT ? this-> data_dim - 1: this-> data_dim;

    // atomic pointer to query.
    std::atomic_int cur_task = 0;
    // Trace v2 mode 4 uses batch-end postwrite: worker threads only append POD
    // records to per-thread memory buffers; CSV serialization happens after
    // pool->runTask returns. Capacities are conservative for the 1k trace-v2 run.
    const unsigned trace_v2_mode_global = trace_v2_mode_;
    const bool trace_v2_batch_postwrite = trace_v2_mode_global >= 4;
    const size_t trace_v2_thread_count = std::max<size_t>(1, ctxs.size());
    const size_t trace_v2_expansion_capacity_per_thread = trace_v2_batch_postwrite ? std::max<size_t>(65536, query_num * 512) : 0;
    const size_t trace_v2_edge_capacity_per_thread = trace_v2_batch_postwrite ? std::max<size_t>(1048576, query_num * 8192) : 0;
    std::vector<std::vector<ExpansionTraceV2Row>> trace_v2_thread_expansions(trace_v2_thread_count);
    std::vector<std::vector<EdgeScanTraceV2Row>> trace_v2_thread_edges(trace_v2_thread_count);
    std::vector<unsigned> trace_v2_thread_overflow(trace_v2_thread_count, 0);
    const bool collect_region_io_trace = collect_region_io_trace_;
    std::vector<std::vector<GraphIOBatchTraceRow>> region_io_thread_batches(trace_v2_thread_count);
    std::vector<std::vector<GraphIORequestTraceRow>> region_io_thread_requests(trace_v2_thread_count);
    std::vector<std::vector<GraphIOQueryTraceRow>> region_io_thread_queries(trace_v2_thread_count);
    std::vector<std::vector<GraphIOTimestampBatchRow>> region_io_thread_ts_batches(trace_v2_thread_count);
    std::vector<std::vector<GraphIOTimestampRequestRow>> region_io_thread_ts_requests(trace_v2_thread_count);
    std::vector<std::vector<GraphIOQDEventRow>> region_io_thread_qd_events(trace_v2_thread_count);
    std::atomic<int64_t> region_io_global_inflight{0};
    std::atomic<uint64_t> region_io_next_request_id{0};
    const bool enable_global_qd_control = enable_global_qd_control_ && global_qd_cap_ > 0;
    const unsigned global_qd_cap = global_qd_cap_ == 0 ? 1 : global_qd_cap_;
    std::mutex global_qd_control_mutex;
    std::condition_variable global_qd_control_cv;
    int64_t global_qd_control_inflight = 0;
    if (collect_region_io_trace) {
      for (size_t i = 0; i < trace_v2_thread_count; i++) {
        region_io_thread_batches[i].reserve(std::max<size_t>(4096, query_num * 96));
        region_io_thread_requests[i].reserve(std::max<size_t>(8192, query_num * 128));
        region_io_thread_queries[i].reserve(std::max<size_t>(1024, query_num / trace_v2_thread_count + 8));
        region_io_thread_ts_batches[i].reserve(std::max<size_t>(4096, query_num * 96));
        region_io_thread_ts_requests[i].reserve(std::max<size_t>(8192, query_num * 128));
        region_io_thread_qd_events[i].reserve(std::max<size_t>(8192, query_num * 128));
      }
    }
    if (trace_v2_batch_postwrite) {
      for (size_t i = 0; i < trace_v2_thread_count; i++) {
        trace_v2_thread_expansions[i].reserve(trace_v2_expansion_capacity_per_thread);
        trace_v2_thread_edges[i].reserve(trace_v2_edge_capacity_per_thread);
      }
    }

    // parallel for
    pool->runTask([&, this](int tid) {
      // thread local data init
      IOContext& ctx = ctxs[tid];
      auto scratch = scratchs[tid];
      auto query_scratch = &(scratchs[tid]);
      // these pointers can init earlier
      const T *    query = scratch.aligned_query_T;
      const float *query_float = scratch.aligned_query_float;
      // sector scratch
      _u64 &sector_scratch_idx = query_scratch->sector_idx;
      char *sector_scratch = query_scratch->sector_scratch;
      float *pq_dists = query_scratch->aligned_pqtable_dist_scratch;
      // query <-> neighbor list
      float *dist_scratch = query_scratch->aligned_dist_scratch;
      _u8 *  pq_coord_scratch = query_scratch->aligned_pq_coord_scratch;
      // visited map/set
      tsl::robin_set<_u32> &page_visited = *(query_scratch->page_visited);
      tsl::robin_set<_u32> &visited = *(query_scratch->visited);
      tsl::robin_map<_u32, char*> loaded; // loaded nodes by cached.
      tsl::robin_map<_u32, unsigned> loaded_parent;
      loaded.reserve(2048);

      // this is a ring queue for storing sector buffers ptr.
      // when read done, push a sector_buf to here, wait for execute
      CircleQueue<char*> sector_buffers(MAX_N_SECTOR_READS);
      // pre-allocated buffer, will clear up each iter/step.
      std::vector<char*> tmp_bufs(MAX_N_SECTOR_READS);
      std::vector<AlignedRead> frontier_read_reqs(MAX_N_SECTOR_READS);
      std::vector<unsigned> nbr_buf(max_degree);
      std::vector<std::pair<unsigned, char*>> cached_id_bufs(MAX_N_SECTOR_READS);
      CircleQueue<std::shared_ptr<CacheNode>> cached_node(MAX_N_SECTOR_READS);

      while(true) {
        size_t task_id = cur_task++;
        if (task_id >= query_num) {
          break;
        }
        Timer query_timer, tmp_timer, part_timer;

        // record the frontier node, also used to record search path.
        std::vector<std::shared_ptr<FrontierNode>> frontier;
        size_t ftr_id = 0; // frontier id
        tsl::robin_map<_u32, unsigned> selected_retset_position;
        std::vector<GraphIOBatchTraceRow> region_io_batches;
        std::vector<GraphIORequestTraceRow> region_io_requests;
        tsl::robin_map<char*, size_t> region_io_buf_to_batch;
        std::vector<GraphIOTimestampBatchRow> region_io_ts_batches;
        std::vector<GraphIOTimestampRequestRow> region_io_ts_requests;
        std::vector<GraphIOQDEventRow> region_io_qd_events;
        tsl::robin_map<char*, size_t> region_io_buf_to_ts_request;
        int64_t region_io_context_inflight = 0;
        unsigned region_io_completion_order = 0;
        unsigned region_io_next_batch_id = 0;
        unsigned region_io_request_count = 0;
        double region_io_submit_us_total = 0.0;
        double region_io_wait_us_total = 0.0;
        double region_io_page_process_us_total = 0.0;
        double qd_control_wait_us_total = 0.0;
        unsigned qd_control_wait_events = 0;
        unsigned qd_control_split_submit_groups = 0;

        struct QueryRegionCacheEntry {
          unsigned region_id = INF;
          unsigned trigger_pid = INF;
          unsigned physical_base = INF;
          char *region_buf = nullptr;
          bool complete = false;
          std::array<bool, 4> used{{false, false, false, false}};
          std::array<bool, 4> trigger{{false, false, false, false}};
        };
        std::vector<QueryRegionCacheEntry> query_region_cache;
        tsl::robin_map<unsigned, size_t> region_id_to_cache_idx;
        tsl::robin_map<unsigned, char*> region_physical_page_to_buf;
        tsl::robin_set<unsigned> pending_region_ids;
        tsl::robin_map<unsigned, std::vector<std::shared_ptr<FrontierNode>>> pending_region_waiters;
        tsl::robin_map<char*, size_t> region_read_buf_to_cache_idx;
        uint64_t region_cache_bytes = 0;
        uint64_t region_cache_peak_bytes = 0;
        uint64_t region_cache_allocations = 0;
        uint64_t region_cache_frees = 0;
        uint64_t region_cache_manage_us_accum = 0;
        uint64_t region_lookup_us_accum = 0;
        uint64_t region_same_region_pending_frontier_count = 0;
        uint64_t region_pending_waiter_registered_count = 0;
        uint64_t region_pending_waiter_processed_count = 0;
        uint64_t region_pending_waiter_duplicate_count = 0;
        uint64_t region_requested_page_parse_count = 0;
        uint64_t region_prefetched_only_page_parse_count = 0;
        uint64_t region_duplicate_region_submission_count = 0;
        uint64_t region_duplicate_logical_page_process_count = 0;
        uint64_t region_slice_byte_mismatch_count = 0;
        uint64_t region_cache_entries_released_at_query_end = 0;
        uint64_t region_cache_leftover_after_cleanup = 0;
        tsl::robin_set<unsigned> region_processed_logical_pages;
        uint64_t deterministic_logical_page_request_count = 0;
        uint64_t deterministic_logical_request_seq_assigned_count = 0;
        uint64_t deterministic_logical_page_ready_count = 0;
        uint64_t deterministic_logical_page_processed_count = 0;
        uint64_t deterministic_logical_page_duplicate_request_count = 0;
        uint64_t deterministic_logical_page_duplicate_process_count = 0;
        uint64_t deterministic_queue_insert_count = 0;
        uint64_t deterministic_queue_max_depth = 0;
        uint64_t deterministic_processed_without_request_count = 0;
        uint64_t deterministic_physical_io_submission_count = 0;
        uint64_t deterministic_logical_to_physical_coalescing_count = 0;
        uint64_t deterministic_next_request_seq = 0;
        uint64_t deterministic_next_process_seq = 0;
        tsl::robin_set<uint64_t> deterministic_processed_seqs;
        struct DeterministicReadyPage {
          char *page_buf = nullptr;
          std::shared_ptr<FrontierNode> fn;
        };
        std::map<uint64_t, DeterministicReadyPage> deterministic_ready_pages;
        bool region_prefetch_overflow = false;
        if (enable_region_prefetch_) {
          query_region_cache.reserve(256);
          region_id_to_cache_idx.reserve(256);
          region_physical_page_to_buf.reserve(1024);
          pending_region_ids.reserve(256);
          pending_region_waiters.reserve(256);
          region_read_buf_to_cache_idx.reserve(256);
          region_processed_logical_pages.reserve(1024);
        }
        if (collect_region_io_trace) {
          region_io_batches.reserve(256);
          region_io_requests.reserve(512);
          region_io_buf_to_batch.reserve(512);
          region_io_ts_batches.reserve(256);
          region_io_ts_requests.reserve(512);
          region_io_qd_events.reserve(512);
          region_io_buf_to_ts_request.reserve(512);
        }

        // get the current query pointers
        const T* query1 = query_ptr + (task_id * query_aligned_dim);
        _u64* indices = indices_vec.data() + (task_id * k_search);
        float* distances = distances_vec.data() + (task_id * k_search);
        QueryStats* stats = stats_ptr + task_id;
        if (stats != nullptr) {
          if (enable_region_prefetch_) stats->region_prefetch_enabled = 1;
          if (deterministic_logical_page_processing_) stats->deterministic_logical_page_processing_enabled = 1;
        }
#ifdef ENABLE_REPLICA_REDUNDANCY_STATS
        ReplicaRedundancyTracker replica_redundancy_tracker;
#endif
        const bool collect_trace = collect_transition_trace_;
        const unsigned trace_v2_mode = trace_v2_mode_;
        const bool trace_v2_state = trace_v2_mode >= 1;
        const bool trace_v2_count = trace_v2_mode >= 2;
        const bool collect_trace_v2 = trace_v2_mode >= 3;
        const bool trace_v2_postwrite = trace_v2_mode >= 4;
        std::vector<TransitionTraceRow> trace_rows;
        std::vector<ExpansionTraceV2Row> trace_v2_expansion_rows;
        std::vector<EdgeScanTraceV2Row> trace_v2_edge_rows;
        tsl::robin_set<_u32> trace_expanded;
        tsl::robin_map<_u32, unsigned> trace_depth;
        tsl::robin_map<_u32, unsigned> first_arrival_parent;
        tsl::robin_map<_u32, unsigned> first_insert_event;
        tsl::robin_map<_u32, unsigned> first_insert_order;
        tsl::robin_map<_u32, unsigned> first_arrival_kind;
        unsigned expand_order = 0;
        unsigned trace_v2_next_expansion_event_id = 0;
        unsigned trace_v2_next_adjacency_scan_id = 0;
        unsigned trace_v2_next_insert_order = 0;
        if (collect_trace_v2) {
          trace_v2_expansion_rows.reserve(4096);
          trace_v2_edge_rows.reserve(262144);
          first_arrival_parent.reserve(8192);
          first_insert_event.reserve(8192);
          first_insert_order.reserve(8192);
          first_arrival_kind.reserve(8192);
          selected_retset_position.reserve(4096);
        } else if (trace_v2_state) {
          first_arrival_parent.reserve(8192);
          first_insert_event.reserve(8192);
          first_insert_order.reserve(8192);
          first_arrival_kind.reserve(8192);
          selected_retset_position.reserve(4096);
        }

        _mm_prefetch((char *) query1, _MM_HINT_T1);
        // copy query to thread specific aligned and allocated memory (for distance
        // calculations we need aligned data)
        float query_norm = 0;
        for (_u32 i = 0; i < query_dim; i++) {
          scratch.aligned_query_float[i] = query1[i];
          scratch.aligned_query_T[i] = query1[i];
          query_norm += query1[i] * query1[i];
        }
        // if inner product, we also normalize the query and set the last coordinate
        // to 0 (this is the extra coordindate used to convert MIPS to L2 search)
        if (metric == diskann::Metric::INNER_PRODUCT) {
          query_norm = std::sqrt(query_norm);
          scratch.aligned_query_T[this->data_dim - 1] = 0;
          scratch.aligned_query_float[this->data_dim - 1] = 0;
          for (_u32 i = 0; i < this->data_dim - 1; i++) {
            scratch.aligned_query_T[i] /= query_norm;
            scratch.aligned_query_float[i] /= query_norm;
          }
        }
        // reset query
        query_scratch->reset();
        loaded.clear();
        loaded_parent.clear();

        // query <-> PQ chunk centers distances
        pq_table.populate_chunk_distances(query_float, pq_dists);

        std::vector<Neighbor> retset(l_search + 1);
        unsigned cur_list_size = 0;

        std::vector<Neighbor> full_retset;
        full_retset.reserve(4096);

        // lambda to batch compute query<-> node distances in PQ space
        auto graph_page_id = [this](const unsigned id) -> unsigned {
          return enable_region_layout_ ? id2page_[id] : id;
        };

        auto graph_region_id = [this](const unsigned owner) -> unsigned {
          return owner < region_io_owner_to_region_.size() ? region_io_owner_to_region_[owner] : INF;
        };

        auto prefetch_region_id = [this](const unsigned owner) -> unsigned {
          return owner < region_prefetch_owner_to_region_.size() ? region_prefetch_owner_to_region_[owner] : INF;
        };

        auto region_cache_slot_for_physical = [this](const unsigned region_id, const unsigned physical_pid) -> unsigned {
          if (region_id == INF || region_id >= region_prefetch_physical_base_.size()) return INF;
          const unsigned base = region_prefetch_physical_base_[region_id];
          if (base == INF || physical_pid < base || physical_pid >= base + region_prefetch_size_) return INF;
          return physical_pid - base;
        };

        auto region_cache_record_use = [&](const unsigned physical_pid) {
          auto buf_it = region_physical_page_to_buf.find(physical_pid);
          if (buf_it == region_physical_page_to_buf.end()) return;
          for (auto &entry : query_region_cache) {
            if (!entry.complete || entry.region_buf == nullptr) continue;
            const unsigned slot = region_cache_slot_for_physical(entry.region_id, physical_pid);
            if (slot != INF && slot < region_prefetch_size_ && entry.region_buf + static_cast<size_t>(slot) * GR_SECTOR_LEN == buf_it->second) {
              entry.used[slot] = true;
              return;
            }
          }
        };

        auto register_pending_region_waiter = [&](const unsigned rid, const std::shared_ptr<FrontierNode> &fn) {
          auto &waiters = pending_region_waiters[rid];
          for (const auto &existing : waiters) {
            if (existing && existing->pid == fn->pid && existing->id == fn->id) {
              region_pending_waiter_duplicate_count++;
              return false;
            }
          }
          waiters.push_back(fn);
          region_pending_waiter_registered_count++;
          return true;
        };

        auto region_cache_finalize_query = [&]() {
          if (!enable_region_prefetch_) return;
          uint64_t additional_pages_read = 0;
          uint64_t additional_pages_used = 0;
          for (auto &entry : query_region_cache) {
            for (unsigned pos = 0; pos < region_prefetch_size_; pos++) {
              if (!entry.trigger[pos]) {
                additional_pages_read++;
                if (entry.used[pos]) additional_pages_used++;
              }
            }
            if (entry.region_buf != nullptr) {
              diskann::aligned_free(entry.region_buf);
              entry.region_buf = nullptr;
              region_cache_frees++;
              region_cache_entries_released_at_query_end++;
            }
          }
          uint64_t pending_waiters_left = 0;
          for (const auto &kv : pending_region_waiters) {
            pending_waiters_left += static_cast<uint64_t>(kv.second.size());
          }
          if (stats != nullptr) {
            stats->region_prefetch_overflow = region_prefetch_overflow ? 1 : 0;
            stats->region_loaded_regions = static_cast<uint64_t>(query_region_cache.size());
            stats->region_additional_pages_read = additional_pages_read;
            stats->region_additional_pages_used = additional_pages_used;
            stats->region_unused_additional_pages = additional_pages_read >= additional_pages_used ? additional_pages_read - additional_pages_used : 0;
            stats->region_same_region_pending_frontier_count = region_same_region_pending_frontier_count;
            stats->region_pending_waiter_registered_count = region_pending_waiter_registered_count;
            stats->region_pending_waiter_processed_count = region_pending_waiter_processed_count;
            stats->region_pending_waiter_duplicate_count = region_pending_waiter_duplicate_count;
            stats->region_requested_page_parse_count = region_requested_page_parse_count;
            stats->region_prefetched_only_page_parse_count = region_prefetched_only_page_parse_count;
            stats->region_pending_region_leftover_at_query_end = static_cast<uint64_t>(pending_region_ids.size()) + pending_waiters_left;
            stats->region_waiter_registered_but_not_processed = region_pending_waiter_registered_count >= region_pending_waiter_processed_count ? region_pending_waiter_registered_count - region_pending_waiter_processed_count : 0;
            stats->region_duplicate_region_submission_count = region_duplicate_region_submission_count;
            stats->region_duplicate_logical_page_process_count = region_duplicate_logical_page_process_count;
            stats->region_slice_byte_mismatch_count = region_slice_byte_mismatch_count;
            stats->region_cache_entries_released_at_query_end = region_cache_entries_released_at_query_end;
            stats->region_cache_leftover_after_cleanup = 0;
            stats->region_cache_peak_bytes = region_cache_peak_bytes;
            stats->region_cache_allocations = region_cache_allocations;
            stats->region_cache_frees = region_cache_frees;
            stats->region_lookup_us = static_cast<float>(region_lookup_us_accum);
            stats->region_cache_manage_us = static_cast<float>(region_cache_manage_us_accum);
          }
          query_region_cache.clear();
          region_id_to_cache_idx.clear();
          region_physical_page_to_buf.clear();
          pending_region_ids.clear();
          pending_region_waiters.clear();
          region_read_buf_to_cache_idx.clear();
          region_cache_bytes = 0;
          region_cache_leftover_after_cleanup = static_cast<uint64_t>(query_region_cache.size()) +
              static_cast<uint64_t>(region_id_to_cache_idx.size()) +
              static_cast<uint64_t>(region_physical_page_to_buf.size()) +
              static_cast<uint64_t>(pending_region_ids.size()) +
              static_cast<uint64_t>(pending_region_waiters.size()) +
              static_cast<uint64_t>(region_read_buf_to_cache_idx.size()) + region_cache_bytes;
          if (stats != nullptr) {
            stats->region_cache_leftover_after_cleanup = region_cache_leftover_after_cleanup;
          }
        };

        auto percentile_from_sorted = [](const std::vector<uint64_t> &vals, double p) -> double {
          if (vals.empty()) return 0.0;
          size_t idx = static_cast<size_t>(p * static_cast<double>(vals.size()));
          if (idx >= vals.size()) idx = vals.size() - 1;
          return static_cast<double>(vals[idx]);
        };

        auto append_region_io_batch = [&](const std::vector<std::shared_ptr<FrontierNode>> &frontier_nodes,
                                          const size_t begin_idx, const size_t end_idx,
                                          const double construct_us, const double prep_us,
                                          const double submit_us, const double submit_return_us,
                                          const uint64_t submit_before_ns, const uint64_t submit_after_ns,
                                          const int64_t per_context_inflight_before,
                                          const int64_t per_context_inflight_after_submit,
                                          const int64_t global_inflight_before,
                                          const int64_t global_inflight_after_submit) {
          if (!collect_region_io_trace || begin_idx >= end_idx) return;
          const unsigned batch_id = region_io_next_batch_id++;
          std::vector<unsigned> owners;
          std::vector<unsigned> physicals;
          std::vector<unsigned> regions;
          owners.reserve(end_idx - begin_idx);
          physicals.reserve(end_idx - begin_idx);
          regions.reserve(end_idx - begin_idx);
          std::ostringstream key;
          std::ostringstream physical_key;
          std::ostringstream request_order_key;
          for (size_t i = begin_idx; i < end_idx; i++) {
            const unsigned owner = frontier_nodes[i]->id;
            const unsigned physical = frontier_nodes[i]->pid;
            const unsigned region = graph_region_id(owner);
            if (i > begin_idx) key << '|';
            key << owner;
            owners.push_back(owner);
            physicals.push_back(physical);
            regions.push_back(region);
            const unsigned request_index = static_cast<unsigned>(i - begin_idx);
            if (i > begin_idx) {
              physical_key << '|';
              request_order_key << '|';
            }
            physical_key << physical;
            const uint64_t request_id = region_io_next_request_id.fetch_add(1, std::memory_order_relaxed);
            request_order_key << request_id;
            region_io_requests.push_back({task_id, static_cast<unsigned>(tid), batch_id,
                                          request_index, owner, physical,
                                          static_cast<uint64_t>(physical) * GR_SECTOR_LEN + GR_SECTOR_LEN, region});
            region_io_ts_requests.push_back({request_id, task_id, static_cast<unsigned>(tid), batch_id,
                                             request_index, owner, physical, submit_before_ns, 0, INF, 0});
            region_io_buf_to_ts_request[frontier_nodes[i]->sector_buf] = region_io_ts_requests.size() - 1;
          }
          std::vector<unsigned> sorted_phys = physicals;
          std::sort(sorted_phys.begin(), sorted_phys.end());
          std::vector<uint64_t> gaps;
          unsigned contiguous = 0;
          for (size_t i = 1; i < sorted_phys.size(); i++) {
            uint64_t gap = static_cast<uint64_t>(sorted_phys[i]) - static_cast<uint64_t>(sorted_phys[i - 1]);
            gaps.push_back(gap);
            if (gap == 1) contiguous++;
          }
          std::sort(gaps.begin(), gaps.end());
          double gap_mean = 0.0;
          if (!gaps.empty()) {
            gap_mean = static_cast<double>(std::accumulate(gaps.begin(), gaps.end(), static_cast<uint64_t>(0))) / static_cast<double>(gaps.size());
          }
          uint64_t pair_count = 0, le1 = 0, le4 = 0, le16 = 0, le64 = 0, same_region = 0;
          for (size_t i = 0; i < physicals.size(); i++) {
            for (size_t j = i + 1; j < physicals.size(); j++) {
              uint64_t d = physicals[i] > physicals[j] ? physicals[i] - physicals[j] : physicals[j] - physicals[i];
              pair_count++;
              if (d <= 1) le1++;
              if (d <= 4) le4++;
              if (d <= 16) le16++;
              if (d <= 64) le64++;
              if (regions[i] != INF && regions[i] == regions[j]) same_region++;
            }
          }
          auto ratio = [&](uint64_t v) -> double { return pair_count == 0 ? 0.0 : static_cast<double>(v) / static_cast<double>(pair_count); };
          region_io_batches.push_back({task_id, static_cast<unsigned>(tid), batch_id, static_cast<unsigned>(physicals.size()),
                                       key.str(), *std::min_element(owners.begin(), owners.end()),
                                       *std::max_element(owners.begin(), owners.end()),
                                       sorted_phys.front(), sorted_phys.back(),
                                       static_cast<uint64_t>(sorted_phys.back()) - static_cast<uint64_t>(sorted_phys.front()),
                                       gap_mean, percentile_from_sorted(gaps, 0.50), percentile_from_sorted(gaps, 0.95),
                                       percentile_from_sorted(gaps, 0.99), contiguous, ratio(le1), ratio(le4), ratio(le16),
                                       ratio(le64), ratio(same_region), construct_us, prep_us, submit_us, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, static_cast<unsigned>(physicals.size()), false, submit_return_us});
          region_io_ts_batches.push_back({task_id, static_cast<unsigned>(tid), batch_id, batch_id,
                                        static_cast<unsigned>(physicals.size()), submit_before_ns, submit_after_ns,
                                        static_cast<unsigned>(physicals.size()), per_context_inflight_before,
                                        per_context_inflight_after_submit, global_inflight_before,
                                        global_inflight_after_submit, 0, 0, 0, 0, key.str(), physical_key.str(),
                                        request_order_key.str()});
          const size_t batch_index = region_io_batches.size() - 1;
          for (size_t i = begin_idx; i < end_idx; i++) {
            region_io_buf_to_batch[frontier_nodes[i]->sector_buf] = batch_index;
          }
          region_io_request_count += static_cast<unsigned>(physicals.size());
          region_io_submit_us_total += submit_us;
        };

        auto compute_pq_dists = [this, pq_coord_scratch, pq_dists](const unsigned *ids,
                                                                const _u64 n_ids,
                                                                float *dists_out) {
          search_utils::aggregate_coords(ids, n_ids, this->data, this->n_chunks,
                            pq_coord_scratch);
          search_utils::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists,
                          dists_out);
        };

        auto compute_exact_dists_and_push = [&](const char* node_buf, const unsigned id) -> float {
          tmp_timer.reset();
          float cur_expanded_dist = dist_cmp->compare(query, (T*)node_buf,
                                                (unsigned) aligned_dim);
          if (stats != nullptr) {
            stats->n_ext_cmps++;
          }
          full_retset.push_back(Neighbor(id, cur_expanded_dist, true));
          return cur_expanded_dist;
        };

        auto add_to_retset = [&](const int nbor_id, const float nbor_dist, const bool flag) -> unsigned {
          if (nbor_dist >= retset[cur_list_size - 1].distance && (cur_list_size == l_search)) {
            return INF;
          }
          Neighbor nn(nbor_id, nbor_dist, flag);
          // Return position in sorted list where nn inserted
          auto r = InsertIntoPool(retset.data(), cur_list_size, nn);
          if (cur_list_size < l_search) ++cur_list_size;
          return r;
        };

        auto begin_trace_expand = [&](const unsigned current_id, const unsigned parent_id) -> unsigned {
          if (!collect_trace) return 0;
          if (trace_depth.find(current_id) == trace_depth.end()) {
            auto parent_depth = trace_depth.find(parent_id);
            trace_depth[current_id] = parent_depth == trace_depth.end() ? 0 : parent_depth->second + 1;
          }
          trace_expanded.insert(current_id);
          return expand_order++;
        };

        auto append_trace_row = [&](const unsigned order, const unsigned parent_id, const unsigned current_id,
                                    const unsigned neighbor_id, const bool accepted,
                                    const float pq_dist, const bool is_replica, const bool graph_io) {
          if (!collect_trace) return;
          auto current_depth = trace_depth.find(current_id);
          unsigned depth = current_depth == trace_depth.end() ? 0 : current_depth->second;
          if (accepted && trace_depth.find(neighbor_id) == trace_depth.end()) {
            trace_depth[neighbor_id] = depth + 1;
          }
          trace_rows.push_back({task_id, order, parent_id, current_id, neighbor_id, accepted, pq_dist,
                                is_replica, graph_io, depth});
        };


        auto trace_v2_register_entry = [&](const unsigned node_id) {
          if (!trace_v2_state) return;
          if (first_arrival_parent.find(node_id) == first_arrival_parent.end()) {
            first_arrival_parent[node_id] = INF;
            first_insert_event[node_id] = INF;
            first_insert_order[node_id] = trace_v2_next_insert_order++;
            first_arrival_kind[node_id] = TRACE_V2_KIND_ENTRY;
          }
        };

        auto trace_v2_record_first_accept = [&](const unsigned neighbor_id, const unsigned adjacency_source,
                                                const unsigned event_id, const unsigned arrival_kind) -> bool {
          if (!collect_trace_v2) return false;
          if (first_arrival_parent.find(neighbor_id) != first_arrival_parent.end()) return false;
          first_arrival_parent[neighbor_id] = adjacency_source;
          first_insert_event[neighbor_id] = event_id;
          first_insert_order[neighbor_id] = trace_v2_next_insert_order++;
          first_arrival_kind[neighbor_id] = arrival_kind;
          return true;
        };

        auto trace_v2_begin_expansion = [&](const unsigned current_id, const unsigned expansion_kind,
                                            const unsigned triggering_page_owner, const bool graph_io,
                                            const unsigned physical_page_id, const unsigned adjacency_source,
                                            const bool adjacency_from_replica, const unsigned retset_position,
                                            const float current_distance) -> unsigned {
          if (!trace_v2_count) return INF;
          const unsigned event_id = trace_v2_next_expansion_event_id++;
          const unsigned arrival_parent = first_arrival_parent.find(current_id) == first_arrival_parent.end() ? INF : first_arrival_parent[current_id];
          const unsigned arrival_kind = first_arrival_kind.find(current_id) == first_arrival_kind.end() ? TRACE_V2_KIND_NONE : first_arrival_kind[current_id];
          const unsigned insert_event = first_insert_event.find(current_id) == first_insert_event.end() ? INF : first_insert_event[current_id];
          const unsigned insert_order = first_insert_order.find(current_id) == first_insert_order.end() ? INF : first_insert_order[current_id];
          if (collect_trace_v2) {
            trace_v2_expansion_rows.push_back({task_id, event_id, event_id, current_id, arrival_parent, arrival_kind,
                                               insert_event, insert_order, current_distance, expansion_kind,
                                               triggering_page_owner, graph_io, physical_page_id, adjacency_source,
                                               adjacency_from_replica, retset_position, false});
          }
          return event_id;
        };

        auto trace_v2_append_edge = [&](const unsigned event_id, const unsigned scan_id, const unsigned trigger_current,
                                        const unsigned adjacency_source, const unsigned page_owner, const unsigned scan_kind,
                                        const unsigned neighbor_id, const bool accepted, const bool first_accepted,
                                        const float neighbor_distance, const bool graph_io, const bool is_replica,
                                        const unsigned physical_page_id) {
          if (!trace_v2_state) return;
          const unsigned parent_after = first_arrival_parent.find(neighbor_id) == first_arrival_parent.end() ? INF : first_arrival_parent[neighbor_id];
          trace_v2_edge_rows.push_back({task_id, event_id, scan_id, trigger_current, adjacency_source, page_owner,
                                        scan_kind, neighbor_id, accepted, first_accepted, parent_after,
                                        neighbor_distance, graph_io, is_replica, physical_page_id});
        };

        auto compute_and_push_nbrs = [&](const char *node_buf, const unsigned current_id,
                                         const unsigned parent_id, const bool is_replica, const bool graph_io,
                                         const unsigned trigger_event_id = INF, const unsigned trigger_current = INF,
                                         const unsigned page_owner = INF, const unsigned physical_page_id = INF,
                                         const unsigned scan_kind = TRACE_V2_KIND_REPLICA) {
          unsigned *node_nbrs = (unsigned*)node_buf;
          unsigned nnbrs = *(node_nbrs++);
          unsigned nbors_cand_size = 0;
          const unsigned scan_id = collect_trace_v2 ? trace_v2_next_adjacency_scan_id++ : INF;
          for (unsigned m = 0; m < nnbrs; ++m) {
            const unsigned nbor_id = node_nbrs[m];
            if (visited.insert(nbor_id).second) {
              nbr_buf[nbors_cand_size++] = nbor_id;
            }
          }
          const unsigned order = begin_trace_expand(current_id, parent_id);
          if (nbors_cand_size) {
            _mm_prefetch((char *) nbr_buf.data(), _MM_HINT_T1);
            compute_pq_dists(nbr_buf.data(), nbors_cand_size, dist_scratch);
            if (stats != nullptr) {
              stats->n_cmps += (double) nbors_cand_size;
            }
            for (unsigned m = 0; m < nbors_cand_size; ++m) {
              const unsigned nbor_id = nbr_buf[m];
              const float nbor_dist = dist_scratch[m];
              auto r = add_to_retset(nbor_id, nbor_dist, true);
              const bool inserted = r < cur_list_size;
              const bool first_accepted = inserted && trace_v2_record_first_accept(nbor_id, current_id, trigger_event_id, scan_kind);
              trace_v2_append_edge(trigger_event_id, scan_id, trigger_current, current_id, page_owner, scan_kind,
                                   nbor_id, inserted, first_accepted, nbor_dist, graph_io, is_replica, physical_page_id);
              if (collect_trace) append_trace_row(order, parent_id, current_id, nbor_id, inserted, nbor_dist, is_replica, graph_io);
            }
          } else if (collect_trace) {
            append_trace_row(order, parent_id, current_id, INF, false, 0.0f, is_replica, graph_io);
          }
        };

        auto compute_and_push_nbrs_target_update = [&](const char *node_buf, const unsigned current_id,
                                                       const unsigned parent_id, const bool is_replica, const bool graph_io,
                                                       const unsigned physical_page_id = INF,
                                                       const unsigned retset_position = INF,
                                                       const float current_distance = -1.0f) {
          unsigned *node_nbrs = (unsigned*)node_buf;
          unsigned nnbrs = *(node_nbrs++);
          unsigned nbors_cand_size = 0;
          const unsigned event_id = trace_v2_begin_expansion(current_id, TRACE_V2_KIND_OWNER, current_id, graph_io,
                                                             physical_page_id, current_id, false, retset_position, current_distance);
          const unsigned scan_id = collect_trace_v2 ? trace_v2_next_adjacency_scan_id++ : INF;
          for (unsigned m = 0; m < nnbrs; ++m) {
            const unsigned nbor_id = node_nbrs[m];
            if (visited.insert(nbor_id).second) {
              nbr_buf[nbors_cand_size++] = nbor_id;
            }
          }
          const unsigned order = begin_trace_expand(current_id, parent_id);
          if (nbors_cand_size) {
            _mm_prefetch((char *) nbr_buf.data(), _MM_HINT_T1);
            compute_pq_dists(nbr_buf.data(), nbors_cand_size, dist_scratch);
            if (stats != nullptr) {
              stats->n_cmps += (double) nbors_cand_size;
            }
            std::vector<unsigned> expand_nb_ids;
            for (unsigned m = 0; m < nbors_cand_size; ++m) {
              const unsigned nbor_id = nbr_buf[m];
              const float nbor_dist = dist_scratch[m];
              auto r = add_to_retset(nbor_id, nbor_dist, true);
              const bool inserted = r < cur_list_size;
              const bool first_accepted = inserted && trace_v2_record_first_accept(nbor_id, current_id, event_id, TRACE_V2_KIND_OWNER);
              trace_v2_append_edge(event_id, scan_id, current_id, current_id, current_id, TRACE_V2_KIND_OWNER,
                                   nbor_id, inserted, first_accepted, nbor_dist, graph_io, is_replica, physical_page_id);
              if (collect_trace) append_trace_row(order, parent_id, current_id, nbor_id, inserted, nbor_dist, is_replica, graph_io);
              if (dist_scratch[m] < retset[cur_list_size - 1].distance * pq_filter_ratio &&
                  loaded.find(nbor_id) != loaded.end()) {
                expand_nb_ids.push_back(nbor_id);
                if (r < cur_list_size) {
                  retset[r].flag = false;
                }
              }
            }
            for (unsigned m = 0; m < expand_nb_ids.size(); m++) {
              compute_and_push_nbrs(loaded[expand_nb_ids[m]], expand_nb_ids[m], current_id, true, false,
                                    event_id, current_id, loaded_parent.find(expand_nb_ids[m]) == loaded_parent.end() ? INF : loaded_parent[expand_nb_ids[m]],
                                    physical_page_id, TRACE_V2_KIND_REPLICA_TRIGGERED);
            }
          } else if (collect_trace) {
            append_trace_row(order, parent_id, current_id, INF, false, 0.0f, is_replica, graph_io);
          }
        };

        auto process_graph_page = [&](char *page_buf, const std::shared_ptr<FrontierNode> &fn) {
          const _u32 exact_id = fn->id;
          const float exact_dist = compute_exact_dists_and_push(page_buf, exact_id);
          char *node_buf = page_buf + emb_node_len + sizeof(unsigned) * (1 + n_gc_node_per_sector);
          unsigned *p_layout = (unsigned*)(page_buf + emb_node_len);
          unsigned p_size = *(p_layout++);
#ifdef ENABLE_REPLICA_REDUNDANCY_STATS
          std::vector<uint32_t> replica_adjacency_ids;
          if (p_size > 1) {
            replica_adjacency_ids.reserve(p_size - 1);
          }
          for (unsigned j = 1; j < p_size; j++) {
            replica_adjacency_ids.push_back(static_cast<uint32_t>(p_layout[j]));
          }
          replica_redundancy_tracker.record_page(
              static_cast<uint32_t>(fn->pid), static_cast<uint32_t>(exact_id),
              replica_adjacency_ids);
#endif
          for (unsigned j = 1; j < p_size; j++) {
            if (node_in_mem_pos(p_layout[j]) == INF) {
              char *nnbr_buf = node_buf + j * graph_node_len;
              loaded.insert({p_layout[j], nnbr_buf});
              loaded_parent.insert({p_layout[j], exact_id});
            }
          }
          const unsigned retset_pos = selected_retset_position.find(exact_id) == selected_retset_position.end() ? INF : selected_retset_position[exact_id];
          compute_and_push_nbrs_target_update(node_buf, exact_id, INF, false, true, static_cast<unsigned>(fn->pid), retset_pos, exact_dist);
        };

        auto compute_and_add_to_retset = [&](const unsigned *node_ids, const _u64 n_ids) {
          compute_pq_dists(node_ids, n_ids, dist_scratch);
          for (_u64 i = 0; i < n_ids; ++i) {
            retset[cur_list_size].id = node_ids[i];
            retset[cur_list_size].distance = dist_scratch[i];
            retset[cur_list_size++].flag = true;
            visited.insert(node_ids[i]);
            trace_v2_register_entry(node_ids[i]);
          }
        };

        part_timer.reset();
        if (mem_L) {
          std::vector<unsigned> mem_tags(mem_L);
          std::vector<float> mem_dists(mem_L);
          std::vector<T*> res = std::vector<T*>();
          mem_index_->search_with_tags(query, mem_L, mem_L, mem_tags.data(), mem_dists.data(), nullptr, res);
          compute_and_add_to_retset(mem_tags.data(), std::min((unsigned)mem_L, (unsigned)l_search));
        } else {
          // we only have one medoid.
          compute_and_add_to_retset(&medoids[0], 1);
        }

        std::sort(retset.begin(), retset.begin() + cur_list_size);

        if (stats != nullptr) {
          stats->preprocess_us += (double) part_timer.elapsed();
        }
        unsigned num_ios = 0;

        // map unfinished sector_buf to the frontier node.
        tsl::robin_map<char*, std::shared_ptr<FrontierNode>> sec_buf2ftr;

        // these data are count seperately
        _u32 n_io_in_q = 0; // how many io left
        _u32 n_cached_in_q = 0; // how many proc left
        _u32 n_proc_in_q = 0; // how many proc left

        auto assign_logical_request_seq = [&](const std::shared_ptr<FrontierNode> &fn) {
          if (!fn) return;
          if (fn->logical_request_seq != UINT64_MAX) {
            deterministic_logical_page_duplicate_request_count++;
            return;
          }
          fn->logical_request_seq = deterministic_next_request_seq++;
          deterministic_logical_page_request_count++;
          deterministic_logical_request_seq_assigned_count++;
        };

        auto queue_page_for_processing = [&](char *page_buf, const std::shared_ptr<FrontierNode> &fn) {
          if (page_buf == nullptr || !fn) return;
          sec_buf2ftr[page_buf] = fn;
          if (!deterministic_logical_page_processing_) {
            sector_buffers.push(page_buf);
            n_proc_in_q++;
            return;
          }
          if (fn->logical_request_seq == UINT64_MAX) {
            deterministic_processed_without_request_count++;
            fn->logical_request_seq = deterministic_next_request_seq++;
            deterministic_logical_page_request_count++;
            deterministic_logical_request_seq_assigned_count++;
          }
          deterministic_logical_page_ready_count++;
          auto inserted = deterministic_ready_pages.emplace(fn->logical_request_seq, DeterministicReadyPage{page_buf, fn});
          if (!inserted.second) {
            deterministic_logical_page_duplicate_process_count++;
            inserted.first->second = DeterministicReadyPage{page_buf, fn};
          }
          deterministic_queue_insert_count++;
          deterministic_queue_max_depth = std::max<uint64_t>(deterministic_queue_max_depth, deterministic_ready_pages.size());
          while (true) {
            auto it = deterministic_ready_pages.find(deterministic_next_process_seq);
            if (it == deterministic_ready_pages.end()) break;
            sector_buffers.push(it->second.page_buf);
            n_proc_in_q++;
            deterministic_ready_pages.erase(it);
            deterministic_next_process_seq++;
          }
        };

        while (num_ios < io_limit) {

          if (n_proc_in_q > 0) {
            part_timer.reset();
            auto sector_buf = sector_buffers.get();
            if (sec_buf2ftr.find(sector_buf) == sec_buf2ftr.end()) {
              std::cout << "(bug) read error!" << std::endl;
              exit(-1);
            }

            auto fn = sec_buf2ftr[sector_buf];
            const double region_io_page_begin_us = collect_region_io_trace ? region_io_now_us() : 0.0;
            process_graph_page(sector_buf, fn);
            if (enable_region_prefetch_ && !region_processed_logical_pages.insert(fn->pid).second) {
              region_duplicate_logical_page_process_count++;
            }
            if (deterministic_logical_page_processing_) {
              if (fn->logical_request_seq == UINT64_MAX) {
                deterministic_processed_without_request_count++;
              } else if (!deterministic_processed_seqs.insert(fn->logical_request_seq).second) {
                deterministic_logical_page_duplicate_process_count++;
              }
              deterministic_logical_page_processed_count++;
            }
            if (stats != nullptr) stats->disk_proc_us += (double) part_timer.elapsed();
            if (collect_region_io_trace) {
              const double region_io_page_us = region_io_now_us() - region_io_page_begin_us;
              auto batch_it = region_io_buf_to_batch.find(sector_buf);
              if (batch_it != region_io_buf_to_batch.end() && batch_it->second < region_io_batches.size()) {
                region_io_batches[batch_it->second].page_process_us += region_io_page_us;
                region_io_buf_to_batch.erase(batch_it);
              }
              region_io_page_process_us_total += region_io_page_us;
            }

            sec_buf2ftr.erase(sector_buf);
            n_proc_in_q--;
          }

          // calculate in memory node.
          while (n_cached_in_q > 0) {
            part_timer.reset();
            auto cn = cached_node.get();

            unsigned nbors_size = 0;
            const unsigned event_id = trace_v2_begin_expansion(cn->id, cn->is_replica ? TRACE_V2_KIND_REPLICA : TRACE_V2_KIND_MEM,
                                                               cn->parent_id, false, INF, cn->id, cn->is_replica,
                                                               cn->retset_position, -1.0f);
            const unsigned scan_id = collect_trace_v2 ? trace_v2_next_adjacency_scan_id++ : INF;
            const unsigned scan_kind = cn->is_replica ? TRACE_V2_KIND_REPLICA : TRACE_V2_KIND_MEM;
            for (unsigned m = 0; m < cn->nb_size; ++m) {
              const unsigned nbor_id = cn->node_nbrs[m];
              if (visited.insert(nbor_id).second) {
                nbr_buf[nbors_size++] = nbor_id;
              }
            }
            const unsigned order = begin_trace_expand(cn->id, cn->parent_id);
            compute_pq_dists(nbr_buf.data(), nbors_size, dist_scratch);
            if (stats != nullptr) {
              stats->n_cmps += (double) nbors_size;
            }
            for (unsigned m = 0; m < nbors_size; ++m) {
              const unsigned nbor_id = nbr_buf[m];
              const float nbor_dist = dist_scratch[m];
              auto r = add_to_retset(nbor_id, nbor_dist, true);
              const bool inserted = r < cur_list_size;
              const bool first_accepted = inserted && trace_v2_record_first_accept(nbor_id, cn->id, event_id, scan_kind);
              trace_v2_append_edge(event_id, scan_id, cn->id, cn->id, cn->parent_id, scan_kind,
                                   nbor_id, inserted, first_accepted, nbor_dist, false, cn->is_replica, INF);
              if (collect_trace) append_trace_row(order, cn->parent_id, cn->id, nbor_id, inserted, nbor_dist, cn->is_replica, false);
            }
            if (nbors_size == 0 && collect_trace) {
              append_trace_row(order, cn->parent_id, cn->id, INF, false, 0.0f, cn->is_replica, false);
            }
            n_cached_in_q--;
            if (stats != nullptr) stats->cache_proc_us += (double) part_timer.elapsed();
          }

          if (n_io_in_q > 0) {
            unsigned min_r = 0;
            if (n_proc_in_q == 0) min_r = 1;
            part_timer.reset();
            const double region_io_getevents_begin_us = collect_region_io_trace ? region_io_now_us() : 0.0;
            const uint64_t region_io_getevents_begin_ns = collect_region_io_trace ? region_io_now_ns() : 0;
            int n_read_blks = io_manager->get_events(ctx, min_r, n_io_in_q, tmp_bufs);
            const uint64_t region_io_getevents_end_ns = collect_region_io_trace ? region_io_now_ns() : 0;
            const double region_io_getevents_end_us = collect_region_io_trace ? region_io_now_us() : 0.0;
            if (collect_region_io_trace) {
              const double wait_us = region_io_getevents_end_us - region_io_getevents_begin_us;
              region_io_wait_us_total += wait_us;
              tsl::robin_set<size_t> completed_batches;
              for (int i = 0; i < n_read_blks; i++) {
                auto batch_it = region_io_buf_to_batch.find(tmp_bufs[i]);
                if (batch_it != region_io_buf_to_batch.end()) {
                  completed_batches.insert(batch_it->second);
                }
              }
              for (auto batch_index : completed_batches) {
                if (batch_index >= region_io_batches.size()) continue;
                auto &batch = region_io_batches[batch_index];
                batch.getevents_wait_us += wait_us;
                if (!batch.saw_first) {
                  batch.submit_to_first_completion_us = region_io_getevents_end_us - batch.submit_return_us;
                  batch.saw_first = true;
                }
                if (batch_index < region_io_ts_batches.size()) {
                  auto &ts_batch = region_io_ts_batches[batch_index];
                  if (ts_batch.first_getevents_begin_ns == 0) {
                    ts_batch.first_getevents_begin_ns = region_io_getevents_begin_ns;
                  }
                  ts_batch.last_getevents_end_ns = region_io_getevents_end_ns;
                  if (ts_batch.first_completion_ns == 0) {
                    ts_batch.first_completion_ns = region_io_getevents_end_ns;
                  }
                }
              }
            }
            for (int i = n_read_blks - 1; i >= 0; i--) {
              char *completed_buf = tmp_bufs[i];
              char *page_buf_to_process = completed_buf;
              auto region_read_it = region_read_buf_to_cache_idx.find(completed_buf);
              if (region_read_it != region_read_buf_to_cache_idx.end() && region_read_it->second < query_region_cache.size()) {
                auto &entry = query_region_cache[region_read_it->second];
                entry.complete = true;
                pending_region_ids.erase(entry.region_id);
                for (unsigned pos = 0; pos < region_prefetch_size_; pos++) {
                  const unsigned physical = entry.physical_base + pos;
                  region_physical_page_to_buf[physical] = entry.region_buf + static_cast<size_t>(pos) * GR_SECTOR_LEN;
                }
                const unsigned trigger_slot = region_cache_slot_for_physical(entry.region_id, entry.trigger_pid);
                if (trigger_slot == INF || trigger_slot >= region_prefetch_size_) {
                  region_slice_byte_mismatch_count++;
                  std::cerr << "Completed Region buffer has invalid trigger slot: query_id=" << task_id
                            << " region=" << entry.region_id << " trigger_pid=" << entry.trigger_pid << std::endl;
                  std::exit(3);
                }
                page_buf_to_process = entry.region_buf + static_cast<size_t>(trigger_slot) * GR_SECTOR_LEN;
                region_requested_page_parse_count++;
                auto fn_it = sec_buf2ftr.find(completed_buf);
                if (fn_it == sec_buf2ftr.end()) {
                  std::cerr << "Missing FrontierNode for completed Region buffer" << std::endl;
                  std::exit(3);
                }
                fn_it->second->sector_buf = page_buf_to_process;
                sec_buf2ftr.insert({page_buf_to_process, fn_it->second});
                if (collect_region_io_trace) {
                  auto batch_it = region_io_buf_to_batch.find(completed_buf);
                  if (batch_it != region_io_buf_to_batch.end()) {
                    region_io_buf_to_batch[page_buf_to_process] = batch_it->second;
                  }
                }
                auto wait_it = pending_region_waiters.find(entry.region_id);
                if (wait_it != pending_region_waiters.end()) {
                  for (auto &waiting_fn : wait_it->second) {
                    const unsigned waiting_slot = region_cache_slot_for_physical(entry.region_id, waiting_fn->pid);
                    if (waiting_slot == INF || waiting_slot >= region_prefetch_size_) {
                      region_slice_byte_mismatch_count++;
                      std::cerr << "Pending Region waiter outside contiguous range: query_id=" << task_id
                                << " region=" << entry.region_id << " pid=" << waiting_fn->pid << std::endl;
                      std::exit(3);
                    }
                    waiting_fn->sector_buf = entry.region_buf + static_cast<size_t>(waiting_slot) * GR_SECTOR_LEN;
                    queue_page_for_processing(waiting_fn->sector_buf, waiting_fn);
                    region_cache_record_use(waiting_fn->pid);
                    region_pending_waiter_processed_count++;
                    region_requested_page_parse_count++;
                  }
                  pending_region_waiters.erase(wait_it);
                }
                region_read_buf_to_cache_idx.erase(region_read_it);
              }
              if (collect_region_io_trace) {
                auto batch_it = region_io_buf_to_batch.find(tmp_bufs[i]);
                if (batch_it != region_io_buf_to_batch.end() && batch_it->second < region_io_batches.size()) {
                  auto &batch = region_io_batches[batch_it->second];
                  if (batch.remaining > 0) {
                    batch.remaining--;
                    if (batch.remaining == 0) {
                      batch.submit_to_all_completion_us = region_io_getevents_end_us - batch.submit_return_us;
                      if (batch_it->second < region_io_ts_batches.size()) {
                        region_io_ts_batches[batch_it->second].last_completion_ns = region_io_getevents_end_ns;
                      }
                    }
                  }
                }
                auto ts_req_it = region_io_buf_to_ts_request.find(tmp_bufs[i]);
                if (ts_req_it != region_io_buf_to_ts_request.end() && ts_req_it->second < region_io_ts_requests.size()) {
                  auto &ts_req = region_io_ts_requests[ts_req_it->second];
                  ts_req.completion_ns = region_io_getevents_end_ns;
                  ts_req.completion_order = region_io_completion_order++;
                  ts_req.result_code = 0;
                  region_io_buf_to_ts_request.erase(ts_req_it);
                }
              }
              // update to sector buffers
              auto fn_for_process = sec_buf2ftr.find(page_buf_to_process);
              if (fn_for_process == sec_buf2ftr.end()) {
                fn_for_process = sec_buf2ftr.find(completed_buf);
              }
              if (fn_for_process == sec_buf2ftr.end()) {
                std::cerr << "Missing FrontierNode for completed graph page" << std::endl;
                std::exit(3);
              }
              queue_page_for_processing(page_buf_to_process, fn_for_process->second);
            }
            if (collect_region_io_trace && n_read_blks > 0) {
              region_io_context_inflight -= n_read_blks;
              const int64_t global_after = region_io_global_inflight.fetch_sub(n_read_blks, std::memory_order_relaxed) - n_read_blks;
              region_io_qd_events.push_back({region_io_getevents_end_ns, static_cast<unsigned>(tid), -n_read_blks,
                                             global_after, region_io_context_inflight, "complete"});
            }
            if (enable_global_qd_control && n_read_blks > 0) {
              {
                std::lock_guard<std::mutex> lock(global_qd_control_mutex);
                global_qd_control_inflight -= n_read_blks;
                if (global_qd_control_inflight < 0) global_qd_control_inflight = 0;
              }
              global_qd_control_cv.notify_all();
            }
            if (stats != nullptr) stats->read_disk_us += (double) part_timer.elapsed();
            n_io_in_q -= n_read_blks;
          }

          _u32 disk_batch_size = beam_width;
          if (n_io_in_q == 0 && n_cached_in_q == 0 && n_proc_in_q <= beam_width / 2) {
            part_timer.reset();
            // clear iteration state
            frontier_read_reqs.clear();
            _u32 marker = 0;
            _u32 num_seen = 0;
            _u32 disk_seen = 0;

            // distribute read nodes
            while (marker < cur_list_size && num_seen < beam_width && disk_seen < disk_batch_size) {
              if (retset[marker].flag) {
                auto id = retset[marker].id;
                selected_retset_position[id] = marker;
                unsigned mem_pos = node_in_mem_pos(id);
                if (mem_pos != INF) {
                  cached_node.push(std::make_shared<CacheNode>(id, mem_graph_[mem_pos].size(), mem_graph_[mem_pos].data(), INF, false, marker));
                  if (disk_batch_size > 0)
                    disk_seen++;
                  else
                    num_seen++;
                  n_cached_in_q++;
                  retset[marker].flag = false;
                } else if (loaded.find(id) != loaded.end()) {
                  unsigned* node_nbrs = (unsigned*)loaded[id];
                  unsigned nb_size = *(node_nbrs++);
                  cached_node.push(std::make_shared<CacheNode>(id, nb_size, node_nbrs, loaded_parent.find(id) == loaded_parent.end() ? INF : loaded_parent[id], true, marker));
                  num_seen++;
                  n_cached_in_q++;
                  retset[marker].flag = false;
                } else {
                  const unsigned pid = graph_page_id(id);
                  bool handled_by_region_path = false;
                  bool satisfied_from_region_cache = false;
                  const double lookup_begin_us = enable_region_prefetch_ ? region_io_now_us() : 0.0;
                  const unsigned rid = enable_region_prefetch_ ? prefetch_region_id(id) : INF;
                  if (enable_region_prefetch_ && rid != INF && region_prefetch_mapping_valid_) {
                    auto cache_it = region_id_to_cache_idx.find(rid);
                    if (cache_it != region_id_to_cache_idx.end() && cache_it->second < query_region_cache.size() &&
                        query_region_cache[cache_it->second].complete) {
                      auto buf_it = region_physical_page_to_buf.find(pid);
                      if (buf_it != region_physical_page_to_buf.end() && page_visited.insert(pid).second) {
                        auto fn = std::make_shared<FrontierNode>(id, pid, gc_index_fid);
                        fn->sector_buf = buf_it->second;
                        assign_logical_request_seq(fn);
                        queue_page_for_processing(fn->sector_buf, fn);
                        region_cache_record_use(pid);
                        region_requested_page_parse_count++;
                        deterministic_logical_to_physical_coalescing_count++;
                        num_seen++;
                        num_ios++;
                        if (stats != nullptr) {
                          stats->n_ios++;
                          stats->region_logical_graph_ios++;
                          stats->region_cache_hits++;
                        }
                        satisfied_from_region_cache = true;
                      } else {
                        satisfied_from_region_cache = true;
                      }
                    } else if (pending_region_ids.find(rid) != pending_region_ids.end()) {
                      if (page_visited.insert(pid).second) {
                        auto fn = std::make_shared<FrontierNode>(id, pid, gc_index_fid);
                        assign_logical_request_seq(fn);
                        register_pending_region_waiter(rid, fn);
                        deterministic_logical_to_physical_coalescing_count++;
                        num_seen++;
                        num_ios++;
                        if (stats != nullptr) {
                          stats->n_ios++;
                          stats->region_logical_graph_ios++;
                          stats->region_cache_hits++;
                          stats->region_duplicate_load_attempts++;
                        }
                      }
                      handled_by_region_path = true;
                    }
                  }
                  if (enable_region_prefetch_) {
                    region_lookup_us_accum += static_cast<uint64_t>(region_io_now_us() - lookup_begin_us);
                  }
                  if (!satisfied_from_region_cache && !handled_by_region_path && page_visited.insert(pid).second) {
                    num_seen++;
                    auto fn = std::make_shared<FrontierNode>(id, pid, gc_index_fid);
                    assign_logical_request_seq(fn);
                    frontier.push_back(fn);
                  } else if (!satisfied_from_region_cache && !handled_by_region_path) {
                    deterministic_logical_page_duplicate_request_count++;
                  }
                  retset[marker].flag = false;
                }
              }
              marker++;
            }
            if (stats != nullptr) stats->dispatch_us += (double) part_timer.elapsed();

            // read nhoods of frontier ids
            if (ftr_id < frontier.size()) {
              part_timer.reset();
              const double region_io_construct_begin_us = collect_region_io_trace ? region_io_now_us() : 0.0;
              const size_t region_io_batch_begin = ftr_id;
              if (stats != nullptr) stats->n_hops++;
              const size_t ready_count = frontier.size() - ftr_id;
              size_t submit_limit = ready_count;
              if (enable_global_qd_control) {
                const double wait_begin_us = region_io_now_us();
                std::unique_lock<std::mutex> lock(global_qd_control_mutex);
                qd_control_wait_events++;
                if (ready_count <= static_cast<size_t>(global_qd_cap)) {
                  global_qd_control_cv.wait(lock, [&]() {
                    return global_qd_control_inflight + static_cast<int64_t>(ready_count) <= static_cast<int64_t>(global_qd_cap);
                  });
                  submit_limit = ready_count;
                } else {
                  global_qd_control_cv.wait(lock, [&]() { return global_qd_control_inflight < static_cast<int64_t>(global_qd_cap); });
                  const size_t available = static_cast<size_t>(static_cast<int64_t>(global_qd_cap) - global_qd_control_inflight);
                  submit_limit = std::max<size_t>(1, std::min(ready_count, available));
                  qd_control_split_submit_groups++;
                }
                global_qd_control_inflight += static_cast<int64_t>(submit_limit);
                lock.unlock();
                qd_control_wait_us_total += region_io_now_us() - wait_begin_us;
              }
              while(ftr_id < frontier.size() && frontier_read_reqs.size() < submit_limit) {
                const unsigned owner_id = frontier[ftr_id]->id;
                const unsigned pid = frontier[ftr_id]->pid;
                const unsigned rid = enable_region_prefetch_ ? prefetch_region_id(owner_id) : INF;
                if (enable_region_prefetch_ && rid != INF && region_prefetch_mapping_valid_) {
                  auto existing_region_it = region_id_to_cache_idx.find(rid);
                  if (existing_region_it != region_id_to_cache_idx.end() || pending_region_ids.find(rid) != pending_region_ids.end()) {
                    if (stats != nullptr) stats->region_duplicate_load_attempts++;
                    const bool region_complete = existing_region_it != region_id_to_cache_idx.end() &&
                        existing_region_it->second < query_region_cache.size() && query_region_cache[existing_region_it->second].complete;
                    if (region_complete) {
                      auto buf_it = region_physical_page_to_buf.find(pid);
                      if (buf_it == region_physical_page_to_buf.end()) {
                        std::cerr << "Completed Region cache missing requested page: query_id=" << task_id
                                  << " owner=" << owner_id << " pid=" << pid << " region=" << rid << std::endl;
                        std::exit(3);
                      }
                      frontier[ftr_id]->sector_buf = buf_it->second;
                      queue_page_for_processing(frontier[ftr_id]->sector_buf, frontier[ftr_id]);
                      region_cache_record_use(pid);
                      region_requested_page_parse_count++;
                      deterministic_logical_to_physical_coalescing_count++;
                      if (stats != nullptr) {
                        stats->n_ios++;
                        stats->region_logical_graph_ios++;
                        stats->region_cache_hits++;
                      }
                    } else {
                      region_same_region_pending_frontier_count++;
                      register_pending_region_waiter(rid, frontier[ftr_id]);
                      deterministic_logical_to_physical_coalescing_count++;
                      if (stats != nullptr) {
                        stats->n_ios++;
                        stats->region_logical_graph_ios++;
                        stats->region_cache_hits++;
                      }
                    }
                    num_ios++;
                    ftr_id++;
                    continue;
                  }
                  const double manage_begin_us = region_io_now_us();
                  const uint64_t read_bytes = static_cast<uint64_t>(region_prefetch_size_) * GR_SECTOR_LEN;
                  if (region_cache_bytes + read_bytes > region_cache_limit_bytes_per_query_) {
                    region_prefetch_overflow = true;
                    if (stats != nullptr) stats->region_prefetch_overflow = 1;
                    std::cerr << "Region cache overflow: query_id=" << task_id
                              << " current_regions=" << query_region_cache.size()
                              << " current_cache_bytes=" << region_cache_bytes
                              << " request_bytes=" << read_bytes
                              << " limit_bytes=" << region_cache_limit_bytes_per_query_ << std::endl;
                    std::exit(3);
                  }
                  char *region_buf = nullptr;
                  diskann::alloc_aligned((void **) &region_buf, read_bytes, GR_SECTOR_LEN);
                  QueryRegionCacheEntry entry;
                  entry.region_id = rid;
                  entry.trigger_pid = pid;
                  entry.physical_base = region_prefetch_physical_base_[rid];
                  entry.region_buf = region_buf;
                  const unsigned trigger_slot = region_cache_slot_for_physical(rid, pid);
                  if (trigger_slot == INF || trigger_slot >= region_prefetch_size_) {
                    region_slice_byte_mismatch_count++;
                    std::cerr << "Region trigger page is outside contiguous physical range: query_id=" << task_id
                              << " owner=" << owner_id << " pid=" << pid << " region=" << rid << std::endl;
                    std::exit(3);
                  }
                  entry.trigger[trigger_slot] = true;
                  entry.used[trigger_slot] = true;
                  query_region_cache.push_back(entry);
                  const size_t cache_idx = query_region_cache.size() - 1;
                  region_id_to_cache_idx[rid] = cache_idx;
                  pending_region_ids.insert(rid);
                  region_read_buf_to_cache_idx[region_buf] = cache_idx;
                  region_cache_bytes += read_bytes;
                  region_cache_peak_bytes = std::max(region_cache_peak_bytes, region_cache_bytes);
                  region_cache_allocations++;
                  region_cache_manage_us_accum += static_cast<uint64_t>(region_io_now_us() - manage_begin_us);

                  auto offset = (static_cast<_u64>(entry.physical_base)) * GR_SECTOR_LEN;
                  offset += GR_SECTOR_LEN;
                  if (frontier[ftr_id]->logical_request_seq == UINT64_MAX) assign_logical_request_seq(frontier[ftr_id]);
                  frontier[ftr_id]->sector_buf = region_buf;
                  sec_buf2ftr.insert({region_buf, frontier[ftr_id]});
                  frontier_read_reqs.push_back(AlignedRead(offset, read_bytes, region_buf));
                  deterministic_physical_io_submission_count++;
                  if (stats != nullptr) {
                    stats->n_ios++;
                    stats->region_logical_graph_ios++;
                    stats->region_device_read_submits++;
                    stats->region_device_read_bytes += read_bytes;
                    stats->region_device_16kb_reads++;
                    stats->region_first_triggers++;
                  }
                } else {
                  auto sector_buf = sector_scratch + sector_scratch_idx * GR_SECTOR_LEN;
                  sector_scratch_idx = (sector_scratch_idx + 1) % MAX_N_SECTOR_READS;
                  auto offset = (static_cast<_u64>(pid)) * GR_SECTOR_LEN;
                  offset += GR_SECTOR_LEN; // one page for metadata
                  if (frontier[ftr_id]->logical_request_seq == UINT64_MAX) assign_logical_request_seq(frontier[ftr_id]);
                  frontier[ftr_id]->sector_buf = sector_buf;
                  sec_buf2ftr.insert({sector_buf, frontier[ftr_id]});
                  frontier_read_reqs.push_back(AlignedRead(offset, GR_SECTOR_LEN, sector_buf));
                  deterministic_physical_io_submission_count++;
                  if (stats != nullptr) {
                    stats->n_ios++;
                    stats->region_logical_graph_ios++;
                    stats->region_device_read_submits++;
                    stats->region_device_read_bytes += GR_SECTOR_LEN;
                    stats->region_device_4kb_reads++;
                  }
                }
                num_ios++;
                ftr_id++;
              }
              const size_t region_io_batch_end = ftr_id;
              double region_io_prep_us = 0.0;
              double region_io_submit_us = 0.0;
              const double region_io_construct_us = collect_region_io_trace ? region_io_now_us() - region_io_construct_begin_us : 0.0;
              if (frontier_read_reqs.empty()) {
                if (stats != nullptr) stats->read_disk_us += (double) part_timer.elapsed();
                continue;
              }
              if (collect_region_io_trace) {
                const uint64_t submit_before_ns = region_io_now_ns();
                const int64_t per_context_before = region_io_context_inflight;
                const int64_t global_before = region_io_global_inflight.load(std::memory_order_relaxed);
                const int submitted_count = io_manager->submit_read_reqs(frontier_read_reqs, gc_index_fid, ctx, &region_io_prep_us, &region_io_submit_us);
                n_io_in_q += submitted_count;
                const uint64_t submit_after_ns = region_io_now_ns();
                region_io_context_inflight += submitted_count;
                const int64_t global_after = region_io_global_inflight.fetch_add(submitted_count, std::memory_order_relaxed) + submitted_count;
                region_io_qd_events.push_back({submit_after_ns, static_cast<unsigned>(tid), submitted_count,
                                               global_after, region_io_context_inflight, "submit"});
                append_region_io_batch(frontier, region_io_batch_begin, region_io_batch_end,
                                       region_io_construct_us, region_io_prep_us, region_io_submit_us, region_io_now_us(),
                                       submit_before_ns, submit_after_ns, per_context_before, region_io_context_inflight,
                                       global_before, global_after);
              } else {
                const int submitted_count = io_manager->submit_read_reqs(frontier_read_reqs, gc_index_fid, ctx);
                n_io_in_q += submitted_count;
                (void) submitted_count;
              }
              if (stats != nullptr) stats->read_disk_us += (double) part_timer.elapsed();
            }
            if (n_io_in_q == 0 && n_proc_in_q == 0 && n_cached_in_q == 0) break;
          }
        }
        part_timer.reset();

        // deperated here!
        frontier.clear();

        // done traversal, start read exact embedding.
        _u32 l_idx = 0;
        _u32 embedding_search_L = (_u32)(cur_list_size * emb_search_ratio);
        if (embedding_search_L < k_search) embedding_search_L = k_search;
        while (l_idx < embedding_search_L) {
          frontier_read_reqs.clear();
          cached_id_bufs.clear();
          // page visited don't need to be clear.
          tsl::robin_map<char*, unsigned> sec_buf2pid;
          for (_u32 ord_idx = l_idx; l_idx - ord_idx < MAX_N_SECTOR_READS && l_idx < embedding_search_L; l_idx++) {
            auto pid = graph_page_id(retset[l_idx].id);
            if (page_visited.find(pid) == page_visited.end()) {
              char* cached_emb_buf = get_mem_emb_addr(retset[l_idx].id);
              if (cached_emb_buf != nullptr) {
                cached_id_bufs.push_back(std::make_pair(retset[l_idx].id, cached_emb_buf));
              } else {
                auto sector_buf = sector_scratch + sector_scratch_idx * GR_SECTOR_LEN;
                sector_scratch_idx = (sector_scratch_idx + 1) % MAX_N_SECTOR_READS;
                auto offset = (static_cast<_u64>(pid + 1)) * GR_SECTOR_LEN; // one page for metadata
                frontier_read_reqs.push_back(AlignedRead(offset, GR_SECTOR_LEN, sector_buf));
                page_visited.insert(pid);
                sec_buf2pid.insert({sector_buf, pid});
              }
            }
          }
          int n_ops = 0;
          if (frontier_read_reqs.size() != 0) {
            n_ops = io_manager->submit_read_reqs(frontier_read_reqs, gc_index_fid, ctx);
            if (stats != nullptr) {
              stats-> n_emb_ios += n_ops;
            }
          }
          // pipeline disk read and calculate cached node.
          if (cached_id_bufs.size() != 0) {
            for (_u64 i = 0; i < cached_id_bufs.size(); i++) {
              _mm_prefetch((char *) cached_id_bufs[i].second, _MM_HINT_T0);
              compute_exact_dists_and_push(cached_id_bufs[i].second, cached_id_bufs[i].first);
            }
          }
          while (n_ops > 0) {
            int n_read_blks = io_manager->get_events(ctx, 1, n_ops, tmp_bufs);
            n_ops -= n_read_blks;
            for (int i = 0; i < n_read_blks; i++) {
              auto sector_buf = tmp_bufs[i];
              auto pid = sec_buf2pid[sector_buf];
              unsigned exact_id = enable_region_layout_ && !gp_layout_[pid].empty() ? gp_layout_[pid][0] : pid;
              compute_exact_dists_and_push(sector_buf, exact_id);
            }
          }
        }

        // clear the data.
        frontier_read_reqs.clear();
        visited.clear();
        page_visited.clear();

        // re-sort by distance
        std::sort(full_retset.begin(), full_retset.end(),
                  [](const Neighbor &left, const Neighbor &right) {
                    return left.distance < right.distance;
                  });

        // copy k_search values
        _u64 t = 0;
        for (_u64 i = 0; i < full_retset.size() && t < k_search; i++) {
          if (i > 0 && full_retset[i].id == full_retset[i - 1].id) {
            continue;
          }
          indices[t] = full_retset[i].id;
          if (distances != nullptr) {
            distances[t] = full_retset[i].distance;
            if (metric == diskann::Metric::INNER_PRODUCT) {
              // flip the sign to convert min to max
              distances[t] = (-distances[t]);
              // rescale to revert back to original norms (cancelling the effect of
              // base and query pre-processing)
              if (max_base_norm != 0)
                distances[t] *= (max_base_norm * query_norm);
            }
          }
          t++;
        }

        if (t < k_search) {
          diskann::cerr << "The number of unique ids is less than topk" << std::endl;
          exit(1);
        }

        region_cache_finalize_query();

        if (stats != nullptr) {
          stats->deterministic_logical_page_processing_enabled = deterministic_logical_page_processing_ ? 1 : 0;
          stats->deterministic_logical_page_request_count = deterministic_logical_page_request_count;
          stats->deterministic_logical_request_seq_assigned_count = deterministic_logical_request_seq_assigned_count;
          stats->deterministic_logical_page_ready_count = deterministic_logical_page_ready_count;
          stats->deterministic_logical_page_processed_count = deterministic_logical_page_processed_count;
          stats->deterministic_logical_page_duplicate_request_count = deterministic_logical_page_duplicate_request_count;
          stats->deterministic_logical_page_duplicate_process_count = deterministic_logical_page_duplicate_process_count;
          stats->deterministic_queue_insert_count = deterministic_queue_insert_count;
          stats->deterministic_queue_max_depth = deterministic_queue_max_depth;
          stats->deterministic_queue_leftover_at_query_end = static_cast<uint64_t>(deterministic_ready_pages.size());
          stats->deterministic_requested_but_not_processed_count = deterministic_logical_page_request_count >= deterministic_logical_page_processed_count ? deterministic_logical_page_request_count - deterministic_logical_page_processed_count : 0;
          stats->deterministic_processed_without_request_count = deterministic_processed_without_request_count;
          stats->deterministic_physical_io_submission_count = deterministic_physical_io_submission_count;
          stats->deterministic_logical_to_physical_coalescing_count = deterministic_logical_to_physical_coalescing_count;
          stats->total_us = (double) query_timer.elapsed();
          stats->postprocess_us = (double) part_timer.elapsed();
#ifdef ENABLE_REPLICA_REDUNDANCY_STATS
          const auto &replica_counters = replica_redundancy_tracker.counters();
          stats->replica_graph_page_ios = replica_counters.graph_page_ios;
          stats->replica_first_read_pages = replica_counters.first_read_pages;
          stats->replica_repeated_page_reads = replica_counters.repeated_page_reads;
          stats->replica_attempts = replica_counters.replica_attempts;
          stats->replica_attempts_on_first_read_pages = replica_counters.replica_attempts_on_first_read_pages;
          stats->unique_replica_adjacencies = replica_counters.unique_replica_adjacencies;
          stats->all_replica_duplicates = replica_counters.all_replica_duplicates;
          stats->cross_page_replica_duplicates = replica_counters.cross_page_replica_duplicates;
          stats->replica_to_replica_duplicates = replica_counters.replica_to_replica_duplicates;
          stats->owner_to_replica_duplicates = replica_counters.owner_to_replica_duplicates;
          stats->pages_with_any_replica_duplicate = replica_counters.pages_with_any_replica_duplicate;
          stats->fully_redundant_replica_pages = replica_counters.fully_redundant_replica_pages;
          stats->replica_redundancy = replica_counters.replica_attempts_on_first_read_pages == 0
                                         ? 0.0
                                         : static_cast<double>(replica_counters.replica_to_replica_duplicates) /
                                               static_cast<double>(replica_counters.replica_attempts_on_first_read_pages);
          stats->duplicate_replicas_per_graph_io = replica_counters.graph_page_ios == 0
                                                   ? 0.0
                                                   : static_cast<double>(replica_counters.replica_to_replica_duplicates) /
                                                         static_cast<double>(replica_counters.graph_page_ios);
#endif
        }
        if (collect_region_io_trace) {
          const double query_read_disk_us = stats != nullptr ? stats->read_disk_us : 0.0;
          const double query_total_us = stats != nullptr ? stats->total_us : 0.0;
          for (auto &batch : region_io_batches) {
            batch.query_read_disk_us = query_read_disk_us;
            batch.query_total_us = query_total_us;
          }
          const size_t tix = static_cast<size_t>(tid) < trace_v2_thread_count ? static_cast<size_t>(tid) : 0;
          region_io_thread_batches[tix].insert(region_io_thread_batches[tix].end(),
                                               region_io_batches.begin(), region_io_batches.end());
          region_io_thread_requests[tix].insert(region_io_thread_requests[tix].end(),
                                                region_io_requests.begin(), region_io_requests.end());
          region_io_thread_queries[tix].push_back({task_id, static_cast<unsigned>(tid),
                                                   static_cast<unsigned>(region_io_batches.size()),
                                                   region_io_request_count, region_io_submit_us_total,
                                                   region_io_wait_us_total, region_io_page_process_us_total,
                                                   query_read_disk_us, query_total_us, qd_control_wait_us_total,
                                                   qd_control_wait_events, qd_control_split_submit_groups});
          region_io_thread_ts_batches[tix].insert(region_io_thread_ts_batches[tix].end(),
                                                  region_io_ts_batches.begin(), region_io_ts_batches.end());
          region_io_thread_ts_requests[tix].insert(region_io_thread_ts_requests[tix].end(),
                                                   region_io_ts_requests.begin(), region_io_ts_requests.end());
          region_io_thread_qd_events[tix].insert(region_io_thread_qd_events[tix].end(),
                                                 region_io_qd_events.begin(), region_io_qd_events.end());
        }
        if (collect_trace && !trace_rows.empty()) {
          std::lock_guard<std::mutex> lock(transition_trace_mutex_);
          std::ofstream writer(transition_trace_file_, std::ios::app);
          for (const auto &row : trace_rows) {
            const bool later_expanded = trace_expanded.find(row.neighbor) != trace_expanded.end();
            writer << row.query_id << "," << row.expand_order << "," << row.parent << ","
                   << row.current << "," << row.neighbor << "," << (row.accepted ? 1 : 0)
                   << "," << row.pq_dist << "," << (row.is_replica ? 1 : 0) << ","
                   << (later_expanded ? 1 : 0) << "," << row.depth_from_entry << ","
                   << (row.graph_io ? 1 : 0) << "\n";
          }
        }
        if (trace_v2_postwrite && (!trace_v2_expansion_rows.empty() || !trace_v2_edge_rows.empty())) {
          if (!trace_v2_expansion_rows.empty()) {
            trace_v2_expansion_rows.back().query_finished_flag = true;
          }
          const size_t tix = static_cast<size_t>(tid) < trace_v2_thread_count ? static_cast<size_t>(tid) : 0;
          if (trace_v2_thread_expansions[tix].size() + trace_v2_expansion_rows.size() > trace_v2_thread_expansions[tix].capacity() ||
              trace_v2_thread_edges[tix].size() + trace_v2_edge_rows.size() > trace_v2_thread_edges[tix].capacity()) {
            trace_v2_thread_overflow[tix] = 1;
          } else {
            trace_v2_thread_expansions[tix].insert(trace_v2_thread_expansions[tix].end(),
                                                  trace_v2_expansion_rows.begin(), trace_v2_expansion_rows.end());
            trace_v2_thread_edges[tix].insert(trace_v2_thread_edges[tix].end(),
                                             trace_v2_edge_rows.begin(), trace_v2_edge_rows.end());
          }
        }
      }
    });

    if (collect_region_io_trace) {
      auto write_region_id = [](std::ofstream &writer, unsigned id) {
        if (id == INF) writer << -1;
        else writer << id;
      };
      {
        std::lock_guard<std::mutex> lock(region_io_trace_mutex_);
        std::ofstream writer(region_io_trace_dir_ + "/graph_io_batches.csv", std::ios::app);
        for (const auto &buf : region_io_thread_batches) {
          for (const auto &row : buf) {
            writer << row.query_id << "," << row.thread_id << "," << row.batch_id << ","
                   << row.batch_size << "," << row.logical_owner_key << "," << row.owner_min << ","
                   << row.owner_max << "," << row.physical_min << "," << row.physical_max << ","
                   << row.address_span << "," << row.adjacent_gap_mean << ","
                   << row.adjacent_gap_median << "," << row.adjacent_gap_p95 << ","
                   << row.adjacent_gap_p99 << "," << row.contiguous_adjacent_pairs << ","
                   << row.le1_pair_ratio << "," << row.le4_pair_ratio << ","
                   << row.le16_pair_ratio << "," << row.le64_pair_ratio << ","
                   << row.same_region_pair_ratio << "," << row.construct_us << ","
                   << row.prep_us << "," << row.submit_us << ","
                   << row.submit_to_first_completion_us << "," << row.submit_to_all_completion_us << ","
                   << row.getevents_wait_us << "," << row.page_process_us << ","
                   << row.query_read_disk_us << "," << row.query_total_us << "\n";
          }
        }
      }
      {
        std::lock_guard<std::mutex> lock(region_io_trace_mutex_);
        std::ofstream writer(region_io_trace_dir_ + "/graph_io_requests.csv", std::ios::app);
        for (const auto &buf : region_io_thread_requests) {
          for (const auto &row : buf) {
            writer << row.query_id << "," << row.thread_id << "," << row.batch_id << ","
                   << row.request_index << "," << row.owner_node_id << ","
                   << row.physical_page_id << "," << row.file_offset << ",";
            write_region_id(writer, row.region_id);
            writer << "\n";
          }
        }
      }
      {
        std::lock_guard<std::mutex> lock(region_io_trace_mutex_);
        std::ofstream writer(region_io_trace_dir_ + "/query_io_summary.csv", std::ios::app);
        for (const auto &buf : region_io_thread_queries) {
          for (const auto &row : buf) {
            writer << row.query_id << "," << row.thread_id << "," << row.graph_io_batches << ","
                   << row.graph_io_requests << "," << row.submit_us << ","
                   << row.completion_wait_us << "," << row.page_process_us << ","
                   << row.read_disk_us << "," << row.total_us << "," << row.qd_control_wait_us << ","
                   << row.qd_control_wait_events << "," << row.qd_control_split_submit_groups << "\n";
          }
        }
      }
      {
        std::lock_guard<std::mutex> lock(region_io_trace_mutex_);
        std::ofstream writer(region_io_trace_dir_ + "/stage_g_batch_timeline.csv", std::ios::app);
        for (const auto &buf : region_io_thread_ts_batches) {
          for (const auto &row : buf) {
            writer << row.query_id << "," << row.thread_id << "," << row.logical_batch_id << ","
                   << row.submit_group_id << "," << row.batch_size << "," << row.submit_before_ns << ","
                   << row.submit_after_ns << "," << row.submitted_count << ","
                   << row.per_context_inflight_before << "," << row.per_context_inflight_after_submit << ","
                   << row.global_inflight_before << "," << row.global_inflight_after_submit << ","
                   << row.first_completion_ns << "," << row.last_completion_ns << ","
                   << row.first_getevents_begin_ns << "," << row.last_getevents_end_ns << ","
                   << row.logical_owner_key << "," << row.physical_page_key << "," << row.request_order_key << "\n";
          }
        }
      }
      {
        std::lock_guard<std::mutex> lock(region_io_trace_mutex_);
        std::ofstream writer(region_io_trace_dir_ + "/stage_g_request_timeline.csv", std::ios::app);
        for (const auto &buf : region_io_thread_ts_requests) {
          for (const auto &row : buf) {
            writer << row.request_id << "," << row.query_id << "," << row.thread_id << ","
                   << row.submit_group_id << "," << row.request_index << "," << row.owner_node_id << ","
                   << row.physical_page_id << "," << row.submit_ns << "," << row.completion_ns << ","
                   << row.completion_order << "," << row.result_code << "\n";
          }
        }
      }
      {
        std::lock_guard<std::mutex> lock(region_io_trace_mutex_);
        std::ofstream writer(region_io_trace_dir_ + "/stage_g_qd_events.csv", std::ios::app);
        for (const auto &buf : region_io_thread_qd_events) {
          for (const auto &row : buf) {
            writer << row.timestamp_ns << "," << row.thread_id << "," << row.delta << ","
                   << row.global_inflight_after << "," << row.per_context_inflight_after << ","
                   << row.event << "\n";
          }
        }
      }
    }

    if (trace_v2_batch_postwrite) {
      auto write_id = [](std::ofstream &writer, unsigned id) {
        if (id == INF) writer << -1;
        else writer << id;
      };
      {
        std::ofstream writer(trace_v2_dir_ + "/expansion_trace_v2.csv", std::ios::app);
        for (const auto &buf : trace_v2_thread_expansions) {
          for (const auto &row : buf) {
            writer << row.query_id << "," << row.expansion_event_id << "," << row.expansion_order << ","
                   << row.current << ",";
            write_id(writer, row.arrival_parent);
            writer << "," << trace_v2_kind_name(row.arrival_kind) << ",";
            write_id(writer, row.current_first_insert_event);
            writer << ",";
            write_id(writer, row.current_first_insert_order);
            writer << "," << row.current_distance << "," << trace_v2_kind_name(row.expansion_kind) << ",";
            write_id(writer, row.triggering_page_owner);
            writer << "," << (row.graph_io ? 1 : 0) << ",";
            write_id(writer, row.physical_page_id);
            writer << ",";
            write_id(writer, row.current_adjacency_source);
            writer << "," << (row.current_adjacency_from_replica ? 1 : 0) << ",";
            write_id(writer, row.retset_position);
            writer << "," << (row.query_finished_flag ? 1 : 0) << "\n";
          }
        }
      }
      {
        std::ofstream writer(trace_v2_dir_ + "/edge_scan_trace_v2.csv", std::ios::app);
        for (const auto &buf : trace_v2_thread_edges) {
          for (const auto &row : buf) {
            writer << row.query_id << ",";
            write_id(writer, row.expansion_event_id);
            writer << "," << row.adjacency_scan_id << ",";
            write_id(writer, row.trigger_current);
            writer << ",";
            write_id(writer, row.adjacency_source);
            writer << ",";
            write_id(writer, row.page_owner);
            writer << "," << trace_v2_kind_name(row.scan_kind) << ",";
            write_id(writer, row.neighbor);
            writer << "," << (row.accepted ? 1 : 0) << "," << (row.first_accepted ? 1 : 0) << ",";
            write_id(writer, row.neighbor_arrival_parent_after_scan);
            writer << "," << row.neighbor_distance << "," << (row.graph_io ? 1 : 0) << ","
                   << (row.is_replica ? 1 : 0) << ",";
            write_id(writer, row.physical_page_id);
            writer << "\n";
          }
        }
      }
      {
        std::ofstream writer(trace_v2_dir_ + "/trace_v2_buffer_usage.tsv", std::ios::out);
        writer << "thread_id\texpansion_capacity\texpansion_used\tedge_capacity\tedge_used\toverflow\n";
        for (size_t i = 0; i < trace_v2_thread_count; i++) {
          writer << i << "\t" << trace_v2_thread_expansions[i].capacity() << "\t"
                 << trace_v2_thread_expansions[i].size() << "\t"
                 << trace_v2_thread_edges[i].capacity() << "\t"
                 << trace_v2_thread_edges[i].size() << "\t"
                 << trace_v2_thread_overflow[i] << "\n";
        }
      }
    }
  }

  template class DecoIndex<_u8>;
  template class DecoIndex<_s8>;
  template class DecoIndex<float>;
} // namespace diskann
