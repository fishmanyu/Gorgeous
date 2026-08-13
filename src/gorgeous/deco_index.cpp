// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "logger.h"
#include "deco_index.h"
#include <malloc.h>
#include <fstream>
#include <sys/stat.h>
#include "percentile_stats.h"

#include <omp.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <thread>
#include "distance.h"
#include "exceptions.h"
#include "parameters.h"
#include "search_utils.h"
#include "timer.h"
#include "utils.h"

#include "cosine_similarity.h"
#include "tsl/robin_set.h"

#define GRAPH_METADATA_ONLY 0
#define GRAPH_ONLY 1
#define STARLING_INDEX 2
#define GRAPH_CACHE_INDEX 3

namespace diskann {
  template<typename T>
  DecoIndex<T>::DecoIndex(std::shared_ptr<FileIOManager> &IOManager,
                                diskann::Metric m, bool use_graph_rep_index,
                                _u64 gr_sector_len, bool enable_region_layout)
      : io_manager(IOManager), metric(m), use_graph_rep_index_(use_graph_rep_index),
        enable_region_layout_(enable_region_layout) {
    if (m == diskann::Metric::COSINE || m == diskann::Metric::INNER_PRODUCT) {
      if (std::is_floating_point<T>::value) {
        diskann::cout << "Cosine metric chosen for (normalized) float data."
                         "Changing distance to L2 to boost accuracy."
                      << std::endl;
        m = diskann::Metric::L2;
      } else {
        diskann::cerr << "WARNING: Cannot normalize integral data types."
                      << " This may result in erroneous results or poor recall."
                      << " Consider using L2 distance with integral data types."
                      << std::endl;
      }
    }

    this->dist_cmp.reset(diskann::get_distance_function<T>(m));
    this->dist_cmp_float.reset(diskann::get_distance_function<float>(m));
    this->GR_SECTOR_LEN = gr_sector_len;
  }

  template<typename T>
  DecoIndex<T>::~DecoIndex() {
    if (data != nullptr) {
      delete[] data;
    }

    if (centroid_data != nullptr) {
      aligned_free(centroid_data);
      centroid_data = nullptr;
    }

    delete[] medoids;
    medoids = nullptr;

    delete cached_popular;
    cached_popular = nullptr;

    if (load_flag) {
      this->destroy_thread_data();
      io_manager->close();
    }

    mem_index_.reset();
  }


  template<typename T>
  void DecoIndex<T>::setup_thread_data(_u64 nthreads) {
    pool = std::make_shared<ThreadPool>(nthreads);
    ctxs.resize(nthreads);
    scratchs.resize(nthreads);
    // parallel for
    pool->runTask([&, this](int tid) {
      this->io_manager->register_thread();
      ctxs[tid] = this->io_manager->get_ctx();
      // alloc space for the thread
      DataScratch<T> scratch;
      diskann::alloc_aligned((void **) &scratch.sector_scratch,
                              (_u64) MAX_N_SECTOR_READS * (_u64) GR_SECTOR_LEN,
                              GR_SECTOR_LEN);
      diskann::alloc_aligned(
          (void **) &scratch.aligned_pq_coord_scratch,
          (_u64) MAX_GRAPH_DEGREE * (_u64) MAX_PQ_CHUNKS * sizeof(_u8), 256);
      diskann::alloc_aligned((void **) &scratch.aligned_pqtable_dist_scratch,
                              256 * (_u64) MAX_PQ_CHUNKS * sizeof(float), 256);
      diskann::alloc_aligned((void **) &scratch.aligned_dist_scratch,
                              (_u64) MAX_GRAPH_DEGREE * sizeof(float), 256);
      diskann::alloc_aligned((void **) &scratch.aligned_query_T,
                              this->aligned_dim * sizeof(T), 8 * sizeof(T));
      diskann::alloc_aligned((void **) &scratch.aligned_query_float,
                              this->aligned_dim * sizeof(float),
                              8 * sizeof(float));
      scratch.visited = new tsl::robin_set<_u32>(4096);
      scratch.pq_calculated = new tsl::robin_map<_u32, float>(4096);
      scratch.page_visited = new tsl::robin_set<_u32>(1024);
      scratch.exact_visited = new tsl::robin_map<_u32, bool>(1024);

      memset(scratch.aligned_query_T, 0, this->aligned_dim * sizeof(T));
      memset(scratch.aligned_query_float, 0,
              this->aligned_dim * sizeof(float));
      scratchs[tid] = scratch;
    });
    load_flag = true;
  }

  template<typename T>
  void DecoIndex<T>::destroy_thread_data() {
    diskann::cout << "Clearing scratch" << std::endl;
    assert(this->scratchs.size() == this->max_nthreads);
    for (_u64 tid = 0; tid < this->max_nthreads; tid++) {
      auto &scratch = scratchs[tid];
      diskann::aligned_free((void *) scratch.sector_scratch);
      diskann::aligned_free((void *) scratch.aligned_pq_coord_scratch);
      diskann::aligned_free((void *) scratch.aligned_pqtable_dist_scratch);
      diskann::aligned_free((void *) scratch.aligned_dist_scratch);
      diskann::aligned_free((void *) scratch.aligned_query_float);
      diskann::aligned_free((void *) scratch.aligned_query_T);

      delete scratch.visited;
      delete scratch.pq_calculated;
      delete scratch.page_visited;
      delete scratch.exact_visited;
      scratch.visited = nullptr;
      scratch.pq_calculated = nullptr;
      scratch.page_visited = nullptr;
      scratch.exact_visited = nullptr;
    }
    this->io_manager->deregister_all_threads();
  }

  template<typename T>
  void DecoIndex<T>::enable_transition_trace(bool enabled, const std::string &trace_file) {
    collect_transition_trace_ = enabled;
    transition_trace_file_ = trace_file;
    if (!collect_transition_trace_) {
      return;
    }
    mkdir("logs", 0755);
    std::lock_guard<std::mutex> lock(transition_trace_mutex_);
    std::ofstream writer(transition_trace_file_, std::ios::out);
    writer << "query_id,expand_order,parent,current,neighbor,accepted,pq_dist,is_replica,later_expanded,depth_from_entry,graph_io\n";
  }

  template<typename T>
  void DecoIndex<T>::enable_trace_v2(unsigned mode, const std::string &trace_dir) {
    trace_v2_mode_ = mode;
    trace_v2_dir_ = trace_dir;
    if (trace_v2_mode_ == 0 || trace_v2_mode_ == 1 || trace_v2_mode_ == 2 || trace_v2_mode_ == 3) {
      return;
    }
    mkdir(trace_v2_dir_.c_str(), 0755);
    std::lock_guard<std::mutex> lock(trace_v2_mutex_);
    {
      std::ofstream writer(trace_v2_dir_ + "/expansion_trace_v2.csv", std::ios::out);
      writer << "query_id,expansion_event_id,expansion_order,current,arrival_parent,arrival_kind,"
             << "current_first_insert_event,current_first_insert_order,current_distance,expansion_kind,"
             << "triggering_page_owner,graph_io,physical_page_id,current_adjacency_source,"
             << "current_adjacency_from_replica,retset_position_or_pop_position,query_finished_flag\n";
    }
    {
      std::ofstream writer(trace_v2_dir_ + "/edge_scan_trace_v2.csv", std::ios::out);
      writer << "query_id,expansion_event_id,adjacency_scan_id,trigger_current,adjacency_source,"
             << "page_owner,scan_kind,neighbor,accepted,first_accepted,"
             << "neighbor_arrival_parent_after_scan,neighbor_distance,graph_io,is_replica,physical_page_id\n";
    }
  }

  template<typename T>
  void DecoIndex<T>::configure_region_prefetch(bool enabled,
                                               const std::string &region_file,
                                               unsigned region_size,
                                               uint64_t cache_limit_bytes_per_query,
                                               const std::string &buffer_policy,
                                               const std::string &overflow_policy) {
    enable_region_prefetch_ = enabled;
    region_prefetch_size_ = region_size;
    region_cache_limit_bytes_per_query_ = cache_limit_bytes_per_query;
    region_buffer_policy_ = buffer_policy;
    region_cache_overflow_policy_ = overflow_policy;
    region_prefetch_owner_to_region_.clear();
    region_prefetch_owner_to_position_.clear();
    region_prefetch_regions_.clear();
    region_prefetch_physical_base_.clear();
    region_prefetch_mapping_valid_ = false;

    std::cout << "region_prefetch: " << (enable_region_prefetch_ ? "enabled" : "disabled")
              << ", size=" << region_prefetch_size_
              << ", buffer_policy=" << region_buffer_policy_
              << ", cache_limit_bytes_per_query=" << region_cache_limit_bytes_per_query_
              << ", overflow_policy=" << region_cache_overflow_policy_
              << ", region_file=" << region_file << std::endl;

    if (!enable_region_prefetch_) {
      return;
    }
    if (!enable_region_layout_) {
      std::cerr << "Region prefetch requires --enable_region_physical_reorder 1 so regions map to contiguous physical pages." << std::endl;
      std::exit(2);
    }
    if (region_prefetch_size_ != 4) {
      std::cerr << "Task 3A supports only region_prefetch_size=4." << std::endl;
      std::exit(2);
    }
    if (region_buffer_policy_ != "query_lifetime") {
      std::cerr << "Task 3A supports only region_buffer_policy=query_lifetime." << std::endl;
      std::exit(2);
    }
    if (region_cache_overflow_policy_ != "abort") {
      std::cerr << "Task 3A supports only region_cache_overflow_policy=abort." << std::endl;
      std::exit(2);
    }
    if (region_file.empty()) {
      std::cerr << "Region prefetch requires --region_file <regions.tsv>." << std::endl;
      std::exit(2);
    }

    std::ifstream reader(region_file);
    if (!reader) {
      std::cerr << "Failed to open region file for prefetch: " << region_file << std::endl;
      std::exit(2);
    }
    std::string line;
    uint64_t records = 0;
    while (std::getline(reader, line)) {
      if (line.empty() || line.rfind("region_id", 0) == 0) continue;
      std::istringstream iss(line);
      unsigned region_id = INF, position = INF, page_id = INF;
      if (!(iss >> region_id >> position >> page_id)) continue;
      if (position >= region_prefetch_size_) {
        std::cerr << "Region file position exceeds region_prefetch_size: region=" << region_id
                  << " position=" << position << std::endl;
        std::exit(2);
      }
      if (region_id >= region_prefetch_regions_.size()) {
        region_prefetch_regions_.resize(static_cast<size_t>(region_id) + 1);
      }
      if (region_prefetch_regions_[region_id].empty()) {
        region_prefetch_regions_[region_id].assign(region_prefetch_size_, INF);
      }
      if (region_prefetch_regions_[region_id][position] != INF) {
        std::cerr << "Duplicate region position in region file: region=" << region_id
                  << " position=" << position << std::endl;
        std::exit(2);
      }
      region_prefetch_regions_[region_id][position] = page_id;
      if (page_id >= region_prefetch_owner_to_region_.size()) {
        region_prefetch_owner_to_region_.resize(static_cast<size_t>(page_id) + 1, INF);
        region_prefetch_owner_to_position_.resize(static_cast<size_t>(page_id) + 1, INF);
      }
      if (region_prefetch_owner_to_region_[page_id] != INF) {
        std::cerr << "Page appears in multiple regions for prefetch: page=" << page_id << std::endl;
        std::exit(2);
      }
      region_prefetch_owner_to_region_[page_id] = region_id;
      region_prefetch_owner_to_position_[page_id] = position;
      records++;
    }

    region_prefetch_physical_base_.assign(region_prefetch_regions_.size(), INF);
    uint64_t complete_regions = 0;
    for (unsigned rid = 0; rid < region_prefetch_regions_.size(); rid++) {
      const auto &pages = region_prefetch_regions_[rid];
      if (pages.empty()) continue;
      bool complete = true;
      for (unsigned pos = 0; pos < region_prefetch_size_; pos++) {
        if (pages[pos] == INF) {
          complete = false;
          continue;
        }
        if (pages[pos] >= id2page_.size()) {
          std::cerr << "Region page id exceeds id2page mapping: page=" << pages[pos] << std::endl;
          std::exit(2);
        }
      }
      if (!complete) {
        for (unsigned pos = 0; pos < region_prefetch_size_; pos++) {
          if (pages[pos] != INF && pages[pos] < region_prefetch_owner_to_region_.size()) {
            region_prefetch_owner_to_region_[pages[pos]] = INF;
            region_prefetch_owner_to_position_[pages[pos]] = INF;
          }
        }
        continue;
      }
      std::vector<unsigned> physicals;
      physicals.reserve(region_prefetch_size_);
      for (unsigned pos = 0; pos < region_prefetch_size_; pos++) {
        physicals.push_back(id2page_[pages[pos]]);
      }
      std::vector<unsigned> sorted = physicals;
      std::sort(sorted.begin(), sorted.end());
      for (unsigned pos = 1; pos < region_prefetch_size_; pos++) {
        if (sorted[pos] != sorted[0] + pos) {
          std::cerr << "Region is not physically contiguous; aborting online prefetch. region=" << rid
                    << " physicals=";
          for (auto p : physicals) std::cerr << p << " ";
          std::cerr << std::endl;
          std::exit(2);
        }
      }
      if (sorted[0] + region_prefetch_size_ > num_points) {
        std::cerr << "Region read would pass graph file tail: region=" << rid
                  << " base=" << sorted[0] << " size=" << region_prefetch_size_ << std::endl;
        std::exit(2);
      }
      region_prefetch_physical_base_[rid] = sorted[0];
      complete_regions++;
    }
    region_prefetch_mapping_valid_ = true;
    std::cout << "region_prefetch loaded: records=" << records
              << ", regions=" << region_prefetch_regions_.size()
              << ", complete_regions=" << complete_regions << std::endl;
  }

  template<typename T>
  void DecoIndex<T>::configure_global_qd_control(bool enabled, unsigned cap, bool trace_enabled) {
    enable_global_qd_control_ = enabled;
    global_qd_cap_ = cap;
    qd_control_trace_ = trace_enabled;
    std::cout << "global_qd_control: " << (enable_global_qd_control_ ? "enabled" : "disabled")
              << ", cap=" << global_qd_cap_
              << ", trace=" << (qd_control_trace_ ? "enabled" : "disabled") << std::endl;
  }

  template<typename T>
  void DecoIndex<T>::configure_deterministic_logical_page_processing(bool enabled) {
    deterministic_logical_page_processing_ = enabled;
    std::cout << "deterministic_logical_page_processing: "
              << (deterministic_logical_page_processing_ ? "enabled" : "disabled") << std::endl;
  }

  template<typename T>
  void DecoIndex<T>::enable_region_io_trace(bool enabled, const std::string &trace_dir,
                                            const std::string &region_file) {
    collect_region_io_trace_ = enabled;
    region_io_trace_dir_ = trace_dir;
    region_io_owner_to_region_.clear();
    if (!collect_region_io_trace_) {
      return;
    }
    mkdir(region_io_trace_dir_.c_str(), 0755);
    if (!region_file.empty()) {
      std::ifstream reader(region_file);
      std::string line;
      while (std::getline(reader, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        unsigned region_id = INF, position = INF, page_id = INF;
        if (!(iss >> region_id >> position >> page_id)) continue;
        if (page_id >= region_io_owner_to_region_.size()) {
          region_io_owner_to_region_.resize(static_cast<size_t>(page_id) + 1, INF);
        }
        region_io_owner_to_region_[page_id] = region_id;
      }
    }
    std::lock_guard<std::mutex> lock(region_io_trace_mutex_);
    {
      std::ofstream writer(region_io_trace_dir_ + "/graph_io_batches.csv", std::ios::out);
      writer << "query_id,thread_id,batch_id,batch_size,logical_owner_key,owner_min,owner_max,"
             << "physical_min,physical_max,address_span,adjacent_gap_mean,adjacent_gap_median,"
             << "adjacent_gap_p95,adjacent_gap_p99,contiguous_adjacent_pairs,le1_pair_ratio,"
             << "le4_pair_ratio,le16_pair_ratio,le64_pair_ratio,same_region_pair_ratio,"
             << "construct_us,prep_us,submit_us,submit_to_first_completion_us,"
             << "submit_to_all_completion_us,getevents_wait_us,page_process_us,query_read_disk_us,"
             << "query_total_us\n";
    }
    {
      std::ofstream writer(region_io_trace_dir_ + "/graph_io_requests.csv", std::ios::out);
      writer << "query_id,thread_id,batch_id,request_index,owner_node_id,physical_page_id,file_offset,region_id\n";
    }
    {
      std::ofstream writer(region_io_trace_dir_ + "/query_io_summary.csv", std::ios::out);
      writer << "query_id,thread_id,graph_io_batches,graph_io_requests,submit_us,completion_wait_us,"
             << "page_process_us,read_disk_us,total_us,qd_control_wait_us,qd_control_wait_events,"
             << "qd_control_split_submit_groups\n";
    }
    {
      std::ofstream writer(region_io_trace_dir_ + "/stage_g_batch_timeline.csv", std::ios::out);
      writer << "query_id,thread_id,logical_batch_id,submit_group_id,batch_size,submit_before_ns,"
             << "submit_after_ns,submitted_count,per_context_inflight_before,"
             << "per_context_inflight_after_submit,global_inflight_before,global_inflight_after_submit,"
             << "first_completion_ns,last_completion_ns,first_getevents_begin_ns,last_getevents_end_ns,"
             << "logical_owner_key,physical_page_key,request_order_key\n";
    }
    {
      std::ofstream writer(region_io_trace_dir_ + "/stage_g_request_timeline.csv", std::ios::out);
      writer << "request_id,query_id,thread_id,submit_group_id,request_index,owner_node_id,"
             << "physical_page_id,submit_ns,completion_ns,completion_order,result_code\n";
    }
    {
      std::ofstream writer(region_io_trace_dir_ + "/stage_g_qd_events.csv", std::ios::out);
      writer << "timestamp_ns,thread_id,delta,global_inflight_after,per_context_inflight_after,event\n";
    }
  }

  template<typename T>
  void DecoIndex<T>::load_mem_index(Metric metric, const size_t query_dim, 
      const std::string& mem_index_path, const _u32 num_threads,
      const _u32 mem_L) {
      if (mem_index_path.empty()) {
        diskann::cerr << "mem_index_path is needed" << std::endl;
        exit(1);
      }
      mem_index_ = std::make_unique<diskann::Index<T, uint32_t>>(metric, query_dim, 0, false, true);
      mem_index_->load(mem_index_path.c_str(), num_threads, mem_L);
  }

  template<typename T>
  void DecoIndex<T>::load_sampled_tags(const std::string& tags_path,
                                         std::vector<unsigned>& tags) {
    diskann::cout << "Opening bin file " << tags_path << " ... " << std::endl;
    std::ifstream reader(tags_path, std::ios::binary);
    unsigned tags_size, const_one;
    reader.read((char *) &tags_size, sizeof(uint32_t));
    reader.read((char *) &const_one, sizeof(uint32_t));
    tags.resize(tags_size);
    reader.read((char*)tags.data(), tags_size * sizeof(_u32));
    reader.close();
    std::cout << "loaded tags: " << tags.size() << std::endl;
  }

  template<typename T>
  void DecoIndex<T>::load_mem_emb(std::vector<unsigned>& tags, float mem_emb_use_ratio) {
    if (mem_emb_use_ratio == 0) {
      return;
    }
    use_mem_emb = true;
    unsigned n_cached_nodes = num_points;
    if (mem_emb_use_ratio < 1) {
      n_cached_nodes = (int) (num_points * mem_emb_use_ratio);
    }

    _u64 coord_cache_buf_len = n_cached_nodes * aligned_dim;
    diskann::alloc_aligned((void **) &coord_cache_buf,
                           coord_cache_buf_len * sizeof(T), 8 * sizeof(T));
    memset(coord_cache_buf, 0, coord_cache_buf_len * sizeof(T));
    max_cached_emb_id = n_cached_nodes;

    IOContext& ctx = ctxs[0]; // borrow from ctxs
    auto query_scratch = &(scratchs[0]);
    // sector scratch
    _u64 &sector_scratch_idx = query_scratch->sector_idx;
    char *sector_scratch = query_scratch->sector_scratch;
    // vector for read requests.
    std::vector<char*> tmp_bufs(MAX_N_SECTOR_READS);
    std::vector<AlignedRead> frontier_read_reqs(MAX_N_SECTOR_READS);
    int fid = index_fid;  // graph cache fid or index fid.
    if (use_graph_rep_index_) {
      fid = gc_index_fid;
    }

    _u32 cur_id = 0;
    while (cur_id < n_cached_nodes) {
      frontier_read_reqs.clear();
      _u32 st = cur_id;
      _u32 ed = cur_id + MAX_N_SECTOR_READS;
      if (ed > n_cached_nodes) ed = n_cached_nodes;
      cur_id = ed;
      unsigned pid;
      tsl::robin_map<char*, unsigned> sec_buf2id;
      for (_u32 id = st; id < ed; id++) {
        if (use_graph_rep_index_) {
          pid = enable_region_layout_ ? id2page_[id] : id;
        } else {
          pid = id2page_[id];
        }
        auto sector_buf = sector_scratch + sector_scratch_idx * GR_SECTOR_LEN;
        sector_scratch_idx = (sector_scratch_idx + 1) % MAX_N_SECTOR_READS;
        auto offset = static_cast<_u64>(pid + 1) * GR_SECTOR_LEN; // one page for metadata
        frontier_read_reqs.push_back(AlignedRead(offset, GR_SECTOR_LEN, sector_buf));
        sec_buf2id.insert({sector_buf, id});
      }
      if (frontier_read_reqs.size() != 0) {
        auto n_ops = io_manager->submit_read_reqs(frontier_read_reqs, fid, ctx);
        while (n_ops > 0) {
          int n_read_blks = io_manager->get_events(ctx, 1, n_ops, tmp_bufs);
          n_ops -= n_read_blks;
          for (int i = 0; i < n_read_blks; i++) {
            char* buf_ptr = nullptr;
            auto sector_buf = tmp_bufs[i];
            auto id = sec_buf2id[sector_buf];
            if (use_graph_rep_index_) {
              buf_ptr = sector_buf; // cache layout, emb at the beginning.
            } else {
              pid = id2page_[id];
              for (unsigned j = 0; j < gp_layout_[pid].size(); ++j) {
                if (gp_layout_[pid][j] == id) {
                  buf_ptr = sector_buf + j * max_node_len;
                  break;
                }
              }
            }
            char* cached_coords = coord_cache_buf + id * aligned_dim * sizeof(T);
            memcpy(cached_coords, buf_ptr, emb_node_len);
          }
        }
      }
    }
    std::cout << "loaded memory emb done. load " << cur_id << " nodes" << std::endl;
  }

  template<typename T>  // graph only contain one page
  void DecoIndex<T>::load_mem_graph(const std::string& disk_graph_prefix,
                                       std::vector<unsigned>& tags,
                                       float mem_graph_use_ratio) {
    if (mem_graph_use_ratio == 0) {
      return;
    }
    // For fast loading, we use the compact storage for graph.
    use_mem_graph = true;
    this->load_partition_data(disk_graph_prefix, num_points, GRAPH_METADATA_ONLY);
    _u32 graph_header_offset = sizeof(unsigned) * (1 + this->n_graph_node_per_sector);
    std::string graph_fname(disk_graph_file);
    this->graph_fid = io_manager->open(graph_fname, O_DIRECT | O_RDONLY | O_LARGEFILE);
    _u32 n_cached_id = num_points;
    _u32 n_pages = (num_points + n_graph_node_per_sector - 1) / n_graph_node_per_sector;
    if (mem_graph_use_ratio < 1) {
      n_cached_id = (int) (num_points * mem_graph_use_ratio);
      // determine node to cache.
      _u32 n_cache_popular_id = std::min((unsigned)tags.size(), n_cached_id);
      max_cached_id = n_cached_id - n_cache_popular_id;
      if (n_cache_popular_id > 0) {
        for (_u32 i = 0; i < n_cache_popular_id; i++) {
          if (tags[i] >= max_cached_id) {
            cached_popular->insert({tags[i], INF});
          }
        }
      }
      mem_graph_.resize(n_cached_id);
    } else {
      max_cached_id = num_points;
      mem_graph_.resize(num_points);
    }
    // load full graph
    IOContext& ctx = ctxs[0]; // borrow from ctxs
    auto query_scratch = &(scratchs[0]);
    // sector scratch
    _u64 &sector_scratch_idx = query_scratch->sector_idx;
    char *sector_scratch = query_scratch->sector_scratch;
    // vector for read requests.
    std::vector<char*> tmp_bufs(MAX_N_SECTOR_READS);
    std::vector<AlignedRead> frontier_read_reqs(MAX_N_SECTOR_READS);
    _u32 cur_page = 0;
    _u32 cur_popolar_sid = max_cached_id; // start from here

    while (cur_page < n_pages) {
      frontier_read_reqs.clear();
      _u32 st = cur_page;
      _u32 ed = cur_page + MAX_N_SECTOR_READS;
      if (ed > n_pages) ed = n_pages;
      cur_page = ed;
      for (_u32 pid = st; pid < ed; pid++) {
        auto sector_buf = sector_scratch + sector_scratch_idx * 4096;
        sector_scratch_idx = (sector_scratch_idx + 1) % MAX_N_SECTOR_READS;
        auto offset = (static_cast<_u64>(pid + 1)) * 4096; // one page for metadata
        frontier_read_reqs.push_back(AlignedRead(offset, 4096, sector_buf));
      }
      auto n_ops = io_manager->submit_read_reqs(frontier_read_reqs, graph_fid, ctx);
      // read graph
      while (n_ops > 0) {
        int n_read_blks = io_manager->get_events(ctx, 1, n_ops, tmp_bufs);
        n_ops -= n_read_blks;
        for (int i = 0; i < n_read_blks; i++) {
          auto sector_buf = tmp_bufs[i];
          // read graph
          unsigned* p_layout = (unsigned*) sector_buf;
          unsigned p_size = *(p_layout++);
          for (unsigned j = 0; j < p_size; ++j) {
            unsigned id = p_layout[j];
            if (id < max_cached_id) {
              unsigned* node_nbrs = (unsigned*) (sector_buf + graph_header_offset + j * graph_node_len);
              unsigned nnbrs = *(node_nbrs++);
              mem_graph_[id].resize(nnbrs);
              for (unsigned m = 0; m < nnbrs; ++m) {
                mem_graph_[id][m] = node_nbrs[m];
              }
            } else if (cached_popular->find(id) != cached_popular->end()) {
              unsigned* node_nbrs = (unsigned*) (sector_buf + graph_header_offset + j * graph_node_len);
              unsigned nnbrs = *(node_nbrs++);
              mem_graph_[cur_popolar_sid].resize(nnbrs);
              for (unsigned m = 0; m < nnbrs; ++m) {
                mem_graph_[cur_popolar_sid][m] = node_nbrs[m];
              }
              cached_popular->insert_or_assign(id, cur_popolar_sid++);
            }
          }
        }
      }
    }
    std::cout << "loaded memory graph done. load " << cur_page << " pages." << std::endl;
  }

  template<typename T>
  int DecoIndex<T>::load(uint32_t num_threads, const char *index_prefix, const char *pq_prefix,
                            const std::string& disk_index_prefix,
                            const std::string& graph_rep_index_prefix,
                            const std::string& disk_graph_prefix) {
    std::string pq_table_bin = std::string(pq_prefix) + "_pq_pivots.bin";
    std::string pq_compressed_vectors =
        std::string(pq_prefix) + "_pq_compressed.bin";
    std::string disk_index_file = std::string(index_prefix) + "_disk.index";
    std::string disk_graph_rep_index_file = graph_rep_index_prefix + "_graph_rep.index";
    std::string disk_graph_file = disk_graph_prefix + "_disk_graph.index";

    size_t pq_file_dim, pq_file_num_centroids;
    get_bin_metadata(pq_table_bin, pq_file_num_centroids, pq_file_dim,
                     METADATA_SIZE);

    this->disk_index_file = disk_index_file;
    this->disk_graph_rep_index_file = disk_graph_rep_index_file;
    this->disk_graph_file = disk_graph_file;

    if (pq_file_num_centroids != 256) {
      diskann::cout << "Error. Number of PQ centroids is not 256. Exitting."
                    << std::endl;
      return -1;
    }

    this->data_dim = pq_file_dim;
    // will change later if we use PQ on disk or if we are using
    // inner product without PQ
    this->disk_bytes_per_point = this->data_dim * sizeof(T);
    this->aligned_dim = ROUND_UP(pq_file_dim, 8);

    size_t npts_u64, nchunks_u64;
    diskann::load_bin<_u8>(pq_compressed_vectors, this->data, npts_u64,
                           nchunks_u64);

    this->num_points = npts_u64;
    this->n_chunks = nchunks_u64;

    pq_table.load_pq_centroid_bin(pq_table_bin.c_str(), nchunks_u64);

    diskann::cout
        << "Loaded PQ centroids and in-memory compressed vectors. #points: "
        << num_points << " #dim: " << data_dim
        << " #aligned_dim: " << aligned_dim << " #chunks: " << n_chunks
        << std::endl;

    if (n_chunks > MAX_PQ_CHUNKS) {
      std::stringstream stream;
      stream << "Error loading index. Ensure that max PQ bytes for in-memory "
                "PQ data does not exceed "
             << MAX_PQ_CHUNKS << std::endl;
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    // read index metadata
    std::ifstream index_metadata(disk_index_file, std::ios::binary);
    _u32 nr, nc;  // metadata itself is stored as bin format (nr is number of
                  // metadata, nc should be 1)
    READ_U32(index_metadata, nr);
    READ_U32(index_metadata, nc);

    _u64 disk_nnodes;
    _u64 disk_ndims;  // can be disk PQ dim if disk_PQ is set to true
    READ_U64(index_metadata, disk_nnodes);
    READ_U64(index_metadata, disk_ndims);

    std::cout << nr << " " << nc << " " << disk_nnodes << " " << disk_ndims << std::endl;
    if (disk_nnodes != num_points) {
      diskann::cout << "Mismatch in #points for compressed data file and disk "
                       "index file: "
                    << disk_nnodes << " vs " << num_points << std::endl;
      return -1;
    }

    size_t medoid_id_on_file;
    READ_U64(index_metadata, medoid_id_on_file);
    READ_U64(index_metadata, max_node_len);
    READ_U64(index_metadata, nnodes_per_sector);
    max_degree = ((max_node_len - disk_bytes_per_point) / sizeof(unsigned)) - 1;

    this->graph_node_len = max_node_len - disk_bytes_per_point;
    this->emb_node_len = disk_bytes_per_point;

    std::cout << "max node len "<<max_node_len <<" disk bytes "<<disk_bytes_per_point << std::endl;
    if (max_degree > MAX_GRAPH_DEGREE) {
      std::stringstream stream;
      stream << "Error loading index. Ensure that max graph degree (R) does "
                "not exceed "
             << MAX_GRAPH_DEGREE << std::endl;
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    // Deperated in our codebase
    // setting up concept of frozen points in disk index for streaming-DiskANN
    READ_U64(index_metadata, this->num_frozen_points);
    _u64 file_frozen_id;
    READ_U64(index_metadata, file_frozen_id);
    if (this->num_frozen_points == 1)
      this->frozen_location = file_frozen_id;
    if (this->num_frozen_points == 1) {
      diskann::cout << " Detected frozen point in index at location "
                    << this->frozen_location
                    << ". Will not output it at search time." << std::endl;
    }

    // Deperated in our codebase
    READ_U64(index_metadata, this->reorder_data_exists);
    if (this->reorder_data_exists) {
      if (this->use_disk_index_pq == false) {
        throw ANNException(
            "Reordering is designed for used with disk PQ compression option",
            -1, __FUNCSIG__, __FILE__, __LINE__);
      }
      READ_U64(index_metadata, this->reorder_data_start_sector);
      READ_U64(index_metadata, this->ndims_reorder_vecs);
      READ_U64(index_metadata, this->nvecs_per_sector);
    }

    diskann::cout << "Disk-Index File Meta-data: ";
    diskann::cout << "# nodes per sector: " << nnodes_per_sector;
    diskann::cout << ", max node len (bytes): " << max_node_len;
    diskann::cout << ", max node degree: " << max_degree << std::endl;

    index_metadata.close();

    if (use_graph_rep_index_) {
      std::string graph_cache_fname(disk_graph_rep_index_file);
      this->load_partition_data(graph_rep_index_prefix, num_points, GRAPH_CACHE_INDEX);
      this->gc_index_fid = io_manager->open(graph_cache_fname, O_DIRECT | O_RDONLY | O_LARGEFILE);
      std::cout << "Graph-replicated mode." << std::endl;
    } else {
      this->load_partition_data(index_prefix, num_points, STARLING_INDEX);
      std::string index_fname(disk_index_file);
      this->index_fid = io_manager->open(index_fname, O_DIRECT | O_RDONLY | O_LARGEFILE);
    }

    std::cout << "use_graph_rep_index: " << use_graph_rep_index_ << std::endl;
    std::cout << "enable_region_layout: " << enable_region_layout_ << std::endl;

    this->max_nthreads = num_threads;
    this->setup_thread_data(num_threads);
    cached_popular = new tsl::robin_map<_u32, _u32>(0);

    num_medoids = 1;
    medoids = new uint32_t[1];
    medoids[0] = (_u32)(medoid_id_on_file);

    std::string norm_file = std::string(disk_graph_file) + "_max_base_norm.bin";

    if (file_exists(norm_file) && metric == diskann::Metric::INNER_PRODUCT) {
      _u64   dumr, dumc;
      float *norm_val;
      diskann::load_bin<float>(norm_file, norm_val, dumr, dumc);
      this->max_base_norm = norm_val[0];
      std::cout << "Setting re-scaling factor of base vectors to "
                << this->max_base_norm << std::endl;
      delete[] norm_val;
    }

    diskann::cout << "done.." << std::endl;
    return 0;
  }

  template<typename T>
  void DecoIndex<T>::load_partition_data(const std::string &index_prefix,
      const _u64 num_points, int mode) {
    std::string partition_file = index_prefix + "_partition.bin";
    std::cout << partition_file << std::endl;
    std::ifstream part(partition_file);
    _u64          C, partition_nums, nd;
    part.read((char *) &C, sizeof(_u64));
    part.read((char *) &partition_nums, sizeof(_u64));
    part.read((char *) &nd, sizeof(_u64));
    diskann::cout << "Partition meta: C: " << C << " partition_nums: " << partition_nums
              << " nd: " << nd << std::endl;
    if (num_points && (nd != num_points)) {
      diskann::cerr << "partition information not correct." << std::endl;
      exit(-1);
    }
    if (mode == STARLING_INDEX) {
      this->gp_layout_.resize(partition_nums);
      for (unsigned i = 0; i < partition_nums; i++) {
        unsigned s;
        part.read((char *) &s, sizeof(unsigned));
        this->gp_layout_[i].resize(s);
        part.read((char *) gp_layout_[i].data(), sizeof(unsigned) * s);
      }
      this->id2page_.resize(nd);
      part.read((char *) id2page_.data(), sizeof(unsigned) * nd);
    } else if (mode == GRAPH_CACHE_INDEX) {
      this->n_gc_node_per_sector = C;
      if (enable_region_layout_) {
        this->gp_layout_.resize(partition_nums);
        for (unsigned i = 0; i < partition_nums; i++) {
          unsigned s;
          part.read((char *) &s, sizeof(unsigned));
          this->gp_layout_[i].resize(s);
          part.read((char *) gp_layout_[i].data(), sizeof(unsigned) * s);
        }
        this->id2page_.resize(nd);
        part.read((char *) id2page_.data(), sizeof(unsigned) * nd);
      }
    } else if (mode == GRAPH_METADATA_ONLY) {
      this->n_graph_node_per_sector = C;
    } else if (mode == GRAPH_ONLY) {
      this->n_graph_node_per_sector = C;
      for (unsigned i = 0; i < partition_nums; i++) {
        unsigned s;
        part.read((char *) &s, sizeof(unsigned));
        // don't load to memory
        std::vector<unsigned> tmp_layout;
        tmp_layout.resize(s);
        part.read((char *) tmp_layout.data(), sizeof(unsigned) * s);
      }
    }
    diskann::cout << "Load partition data done." << std::endl;
  }

  // instantiations
  template class DecoIndex<_u8>;
  template class DecoIndex<_s8>;
  template class DecoIndex<float>;

}  // namespace diskann
