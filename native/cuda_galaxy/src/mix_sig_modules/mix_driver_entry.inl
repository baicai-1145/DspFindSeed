namespace
{
struct MixPoseFastBufferEntry
{
    int device_id = -2;
    dsp_vec3d_t* ptr = nullptr;
    size_t cap = 0;
    bool in_use = false;
};

struct MixPoseFastBufferLease
{
    int entry_idx = -1;
    dsp_vec3d_t* ptr = nullptr;
};

struct MixPoseFastBufferTiming
{
    double acquire_total_ms = 0.0;
    double acquire_lock_wait_ms = 0.0;
    double acquire_lock_hold_ms = 0.0;
    double acquire_set_device_ms = 0.0;
    double acquire_free_ms = 0.0;
    double acquire_malloc_ms = 0.0;
    double release_total_ms = 0.0;
    double release_lock_wait_ms = 0.0;
    double release_lock_hold_ms = 0.0;
};

inline double MixPoseFastBufferNowMs()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

struct MixPoseFastBufferStore
{
    std::mutex mu;
    std::condition_variable cv;
    bool prewarm_in_progress = false;
    std::vector<MixPoseFastBufferEntry> entries;

    ~MixPoseFastBufferStore()
    {
        for (size_t i = 0; i < entries.size(); ++i)
        {
            MixPoseFastBufferEntry& e = entries[i];
            if (e.ptr != nullptr)
            {
                if (e.device_id >= 0)
                    cudaSetDevice(e.device_id);
                cudaFree(e.ptr);
                e.ptr = nullptr;
            }
        }
    }
};

inline MixPoseFastBufferStore& GetMixPoseFastBufferStore()
{
    static MixPoseFastBufferStore store;
    return store;
}

inline int ResolveFastBufferPoolSlots()
{
    static std::atomic<int> cached_slots{ -1 };
    int cached = cached_slots.load(std::memory_order_relaxed);
    if (cached >= 0)
        return cached;
    int slots = 0;
    if (const char* s = std::getenv("DSP_NATIVE_SIG_FASTBUF_POOL_SLOTS"))
        slots = std::atoi(s);
    if (slots < 0)
        slots = 0;
    cached_slots.store(slots, std::memory_order_relaxed);
    return slots;
}

inline bool SetCudaDeviceIfNeeded(int device_id, MixPoseFastBufferTiming* timing)
{
    if (device_id < 0)
        return true;
    thread_local int tls_device_id = -2147483647;
    if (tls_device_id == device_id)
        return true;
    const bool profile = timing != nullptr;
    const double t0 = profile ? MixPoseFastBufferNowMs() : 0.0;
    const cudaError_t rc = cudaSetDevice(device_id);
    if (profile)
        timing->acquire_set_device_ms += (MixPoseFastBufferNowMs() - t0);
    if (rc != cudaSuccess)
        return false;
    tls_device_id = device_id;
    return true;
}

inline void EnsureMixPoseFastBufferPool(int device_id, size_t count, int target_slots, MixPoseFastBufferTiming* timing)
{
    if (target_slots <= 0)
        return;
    MixPoseFastBufferStore& store = GetMixPoseFastBufferStore();

    std::unique_lock<std::mutex> lk(store.mu);
    if (store.prewarm_in_progress)
        return;

    int ready = 0;
    for (size_t i = 0; i < store.entries.size(); ++i)
    {
        const MixPoseFastBufferEntry& e = store.entries[i];
        if (e.device_id == device_id && e.ptr != nullptr && e.cap >= count)
            ready++;
    }
    if (ready >= target_slots)
        return;

    store.prewarm_in_progress = true;
    lk.unlock();

    int need = target_slots - ready;
    std::vector<MixPoseFastBufferEntry> prepared;
    prepared.reserve(static_cast<size_t>(need));
    for (int i = 0; i < need; ++i)
    {
        MixPoseFastBufferEntry e{};
        e.device_id = device_id;
        if (!SetCudaDeviceIfNeeded(device_id, timing))
            break;
        const bool profile = timing != nullptr;
        const double tm = profile ? MixPoseFastBufferNowMs() : 0.0;
        if (cudaMalloc(reinterpret_cast<void**>(&e.ptr), count * sizeof(dsp_vec3d_t)) != cudaSuccess)
            break;
        if (profile)
            timing->acquire_malloc_ms += (MixPoseFastBufferNowMs() - tm);
        e.cap = count;
        e.in_use = false;
        prepared.push_back(e);
    }

    lk.lock();
    for (size_t i = 0; i < prepared.size(); ++i)
        store.entries.push_back(prepared[i]);
    store.prewarm_in_progress = false;
}

inline bool AcquireMixPoseFastBuffer(int device_id, size_t count, MixPoseFastBufferLease* out_lease, MixPoseFastBufferTiming* timing)
{
    if (out_lease == nullptr)
        return false;
    if (count == 0)
        count = 1;

    MixPoseFastBufferStore& store = GetMixPoseFastBufferStore();
    const bool profile = timing != nullptr;
    const double t_total_begin = profile ? MixPoseFastBufferNowMs() : 0.0;
    const int prewarm_slots = ResolveFastBufferPoolSlots();
    if (prewarm_slots > 0)
        EnsureMixPoseFastBufferPool(device_id, count, prewarm_slots, timing);
    auto add_lock_wait = [&](double begin, double end) {
        if (profile)
            timing->acquire_lock_wait_ms += (end - begin);
    };
    auto add_lock_hold = [&](double begin, double end) {
        if (profile)
            timing->acquire_lock_hold_ms += (end - begin);
    };
    auto finish = [&](bool ok) -> bool {
        if (profile)
            timing->acquire_total_ms += (MixPoseFastBufferNowMs() - t_total_begin);
        return ok;
    };

    int pick_idx = -1;
    int old_device = -2;
    dsp_vec3d_t* old_ptr = nullptr;
    size_t old_cap = 0;
    bool need_resize = false;

    {
        const double t_lock_wait_begin = profile ? MixPoseFastBufferNowMs() : 0.0;
        std::unique_lock<std::mutex> lock(store.mu);
        const double t_lock_hold_begin = profile ? MixPoseFastBufferNowMs() : 0.0;
        add_lock_wait(t_lock_wait_begin, t_lock_hold_begin);

        for (size_t i = 0; i < store.entries.size(); ++i)
        {
            MixPoseFastBufferEntry& e = store.entries[i];
            if (e.in_use || e.device_id != device_id)
                continue;
            if (e.cap >= count)
            {
                pick_idx = static_cast<int>(i);
                break;
            }
            if (pick_idx < 0)
                pick_idx = static_cast<int>(i);
        }
        if (pick_idx < 0)
        {
            for (size_t i = 0; i < store.entries.size(); ++i)
            {
                MixPoseFastBufferEntry& e = store.entries[i];
                if (!e.in_use)
                {
                    pick_idx = static_cast<int>(i);
                    break;
                }
            }
        }
        if (pick_idx < 0)
        {
            store.entries.push_back(MixPoseFastBufferEntry{});
            pick_idx = static_cast<int>(store.entries.size() - 1);
        }

        MixPoseFastBufferEntry& entry = store.entries[static_cast<size_t>(pick_idx)];
        entry.in_use = true;
        old_device = entry.device_id;
        old_ptr = entry.ptr;
        old_cap = entry.cap;
        need_resize = (entry.device_id != device_id) || (entry.cap < count) || (entry.ptr == nullptr);

        add_lock_hold(t_lock_hold_begin, profile ? MixPoseFastBufferNowMs() : t_lock_hold_begin);
    }

    if (!need_resize)
    {
        out_lease->entry_idx = pick_idx;
        out_lease->ptr = old_ptr;
        return finish(true);
    }

    dsp_vec3d_t* new_ptr = old_ptr;
    size_t new_cap = old_cap;
    int new_device = old_device;
    bool malloc_ok = true;

    if (old_device != device_id || old_cap < count || old_ptr == nullptr)
    {
        if (!SetCudaDeviceIfNeeded(device_id, timing))
            return finish(false);
        dsp_vec3d_t* alloc_ptr = nullptr;
        const double t_malloc_begin = profile ? MixPoseFastBufferNowMs() : 0.0;
        const cudaError_t rc_malloc = cudaMalloc(reinterpret_cast<void**>(&alloc_ptr), count * sizeof(dsp_vec3d_t));
        if (profile)
            timing->acquire_malloc_ms += (MixPoseFastBufferNowMs() - t_malloc_begin);
        if (rc_malloc != cudaSuccess)
            malloc_ok = false;
        if (malloc_ok)
        {
            if (old_ptr != nullptr)
            {
                if (old_device >= 0 && old_device != device_id)
                {
                    if (!SetCudaDeviceIfNeeded(old_device, timing))
                    {
                        malloc_ok = false;
                    }
                }
                if (malloc_ok)
                {
                    const double t_free_begin = profile ? MixPoseFastBufferNowMs() : 0.0;
                    cudaFree(old_ptr);
                    if (profile)
                        timing->acquire_free_ms += (MixPoseFastBufferNowMs() - t_free_begin);
                }
            }
            if (malloc_ok)
            {
                new_ptr = alloc_ptr;
                new_cap = count;
                new_device = device_id;
            }
            else
            {
                SetCudaDeviceIfNeeded(device_id, timing);
                cudaFree(alloc_ptr);
            }
        }
    }

    {
        const double t_lock_wait_begin = profile ? MixPoseFastBufferNowMs() : 0.0;
        std::unique_lock<std::mutex> lock(store.mu);
        const double t_lock_hold_begin = profile ? MixPoseFastBufferNowMs() : 0.0;
        add_lock_wait(t_lock_wait_begin, t_lock_hold_begin);

        MixPoseFastBufferEntry& entry = store.entries[static_cast<size_t>(pick_idx)];
        if (!malloc_ok)
        {
            entry.in_use = false;
            add_lock_hold(t_lock_hold_begin, profile ? MixPoseFastBufferNowMs() : t_lock_hold_begin);
            return finish(false);
        }
        entry.device_id = new_device;
        entry.ptr = new_ptr;
        entry.cap = new_cap;
        out_lease->entry_idx = pick_idx;
        out_lease->ptr = entry.ptr;

        add_lock_hold(t_lock_hold_begin, profile ? MixPoseFastBufferNowMs() : t_lock_hold_begin);
    }
    return finish(true);
}

inline void ReleaseMixPoseFastBuffer(MixPoseFastBufferLease* lease, MixPoseFastBufferTiming* timing)
{
    if (lease == nullptr || lease->entry_idx < 0)
        return;
    const bool profile = timing != nullptr;
    const double t_total_begin = profile ? MixPoseFastBufferNowMs() : 0.0;
    MixPoseFastBufferStore& store = GetMixPoseFastBufferStore();
    const double t_lock_wait_begin = profile ? MixPoseFastBufferNowMs() : 0.0;
    std::unique_lock<std::mutex> lock(store.mu);
    const double t_lock_hold_begin = profile ? MixPoseFastBufferNowMs() : 0.0;
    if (profile)
        timing->release_lock_wait_ms += (t_lock_hold_begin - t_lock_wait_begin);
    const int idx = lease->entry_idx;
    if (idx >= 0 && static_cast<size_t>(idx) < store.entries.size())
        store.entries[static_cast<size_t>(idx)].in_use = false;
    lease->entry_idx = -1;
    lease->ptr = nullptr;
    if (profile)
    {
        const double t_end = MixPoseFastBufferNowMs();
        timing->release_lock_hold_ms += (t_end - t_lock_hold_begin);
        timing->release_total_ms += (t_end - t_total_begin);
    }
}
} // namespace

extern "C" int dsp_cuda_mix_signatures_from_seeds_f32(
    const int* galaxy_seeds,
    int seed_count,
    int star_count,
    int collision_fp64,
    int use_fp32_prob_compare,
    int vein_len,
    int device_id,
    int star_type_main_seq,
    int star_type_giant,
    int star_type_white_dwarf,
    int star_type_neutron_star,
    int star_type_black_hole,
    int spectr_m,
    int spectr_k,
    int spectr_g,
    int spectr_f,
    int spectr_a,
    int spectr_b,
    int spectr_o,
    int spectr_x,
    int planet_type_gas,
    int planet_type_ocean,
    int planet_type_vocano,
    int planet_type_desert,
    int planet_type_ice,
    int theme_distribute_default,
    int theme_distribute_birth,
    int theme_distribute_interstellar,
    const int* theme_ids,
    const int* theme_planet_types,
    const float* theme_temperatures,
    const int* theme_distributes,
    const int* theme_water_item_ids,
    const int* theme_vein_spot_offsets,
    const int* theme_vein_spot_values,
    const int* theme_rare_vein_offsets,
    const int* theme_rare_vein_values,
    const int* theme_rare_settings_offsets,
    const float* theme_rare_settings_values,
    int theme_count,
    unsigned long long* out_galaxy_sigs,
    unsigned long long* out_planet_sigs,
    unsigned long long* out_vein_sigs,
    unsigned long long* out_pipeline_sigs)
{
    bool debug_enter = false;
    if (const char* de = std::getenv("DSP_NATIVE_SIG_DEBUG_ENTER"))
        debug_enter = std::atoi(de) != 0;
    if (debug_enter)
    {
        std::fprintf(stderr, "[native-sig-enter] seed_count=%d star_count=%d vein_len=%d theme_count=%d device_id=%d\n",
            seed_count, star_count, vein_len, theme_count, device_id);
        std::fflush(stderr);
    }
    if (galaxy_seeds == nullptr || seed_count <= 0 || star_count <= 0 ||
        out_galaxy_sigs == nullptr || out_planet_sigs == nullptr || out_vein_sigs == nullptr || out_pipeline_sigs == nullptr)
    {
        if (const char* st = std::getenv("DSP_NATIVE_SIG_STAGE_TIMING"))
        {
            if (std::atoi(st) != 0)
                std::fprintf(stderr, "[native-sig-error] invalid-args: seeds=%p seed_count=%d star_count=%d out=%p/%p/%p/%p\n",
                    static_cast<const void*>(galaxy_seeds), seed_count, star_count,
                    static_cast<void*>(out_galaxy_sigs), static_cast<void*>(out_planet_sigs), static_cast<void*>(out_vein_sigs), static_cast<void*>(out_pipeline_sigs));
                std::fflush(stderr);
        }
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    }
    if (theme_count <= 0 || theme_ids == nullptr || theme_planet_types == nullptr || theme_temperatures == nullptr ||
        theme_distributes == nullptr || theme_water_item_ids == nullptr ||
        theme_vein_spot_offsets == nullptr || theme_rare_vein_offsets == nullptr || theme_rare_settings_offsets == nullptr)
    {
        if (const char* st = std::getenv("DSP_NATIVE_SIG_STAGE_TIMING"))
        {
            if (std::atoi(st) != 0)
                std::fprintf(stderr, "[native-sig-error] invalid-theme-args: theme_count=%d ids=%p ptypes=%p temps=%p dist=%p water=%p voff=%p roff=%p rsoff=%p\n",
                    theme_count,
                    static_cast<const void*>(theme_ids),
                    static_cast<const void*>(theme_planet_types),
                    static_cast<const void*>(theme_temperatures),
                    static_cast<const void*>(theme_distributes),
                    static_cast<const void*>(theme_water_item_ids),
                    static_cast<const void*>(theme_vein_spot_offsets),
                    static_cast<const void*>(theme_rare_vein_offsets),
                    static_cast<const void*>(theme_rare_settings_offsets));
                std::fflush(stderr);
        }
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    }
    if (vein_len <= 1 || vein_len > 32)
    {
        if (const char* st = std::getenv("DSP_NATIVE_SIG_STAGE_TIMING"))
        {
            if (std::atoi(st) != 0)
                std::fprintf(stderr, "[native-sig-error] invalid-vein-len=%d\n", vein_len);
                std::fflush(stderr);
        }
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    }
    const int cuda_device_id = device_id >= 0 ? device_id : 0;

    EnumMap em{
        star_type_main_seq, star_type_giant, star_type_white_dwarf, star_type_neutron_star, star_type_black_hole,
        spectr_m, spectr_k, spectr_g, spectr_f, spectr_a, spectr_b, spectr_o, spectr_x,
        planet_type_gas, planet_type_ocean, planet_type_vocano, planet_type_desert, planet_type_ice,
        theme_distribute_default, theme_distribute_birth, theme_distribute_interstellar};

    std::vector<ThemeLite> themes;
    themes.resize(theme_count);
    int max_planet_type = 0;
    for (int i = 0; i < theme_count; ++i)
        max_planet_type = std::max(max_planet_type, theme_planet_types[i]);
    max_planet_type = std::max(max_planet_type, em.planet_type_gas);
    max_planet_type = std::max(max_planet_type, em.planet_type_ocean);
    max_planet_type = std::max(max_planet_type, em.planet_type_vocano);
    max_planet_type = std::max(max_planet_type, em.planet_type_desert);
    max_planet_type = std::max(max_planet_type, em.planet_type_ice);
    std::vector<std::vector<int>> themes_by_planet_type(static_cast<size_t>(max_planet_type + 1));
    for (int i = 0; i < theme_count; ++i)
    {
        ThemeLite t{};
        t.id = theme_ids[i];
        t.planet_type = theme_planet_types[i];
        t.temperature = theme_temperatures[i];
        t.distribute = theme_distributes[i];
        t.water_item_id = theme_water_item_ids[i];
        t.vein_begin = theme_vein_spot_offsets[i];
        t.vein_end = theme_vein_spot_offsets[i + 1];
        t.rare_begin = theme_rare_vein_offsets[i];
        t.rare_end = theme_rare_vein_offsets[i + 1];
        t.rare_settings_begin = theme_rare_settings_offsets[i];
        t.rare_settings_end = theme_rare_settings_offsets[i + 1];
        themes[i] = t;
        if (t.planet_type >= 0 && t.planet_type <= max_planet_type)
            themes_by_planet_type[t.planet_type].push_back(i);
    }
    std::vector<int> type_theme_offsets(static_cast<size_t>(max_planet_type + 2), 0);
    std::vector<int> type_theme_values;
    for (int pt = 0; pt <= max_planet_type; ++pt)
    {
        type_theme_offsets[pt] = static_cast<int>(type_theme_values.size());
        const auto& bucket = themes_by_planet_type[pt];
        type_theme_values.insert(type_theme_values.end(), bucket.begin(), bucket.end());
    }
    type_theme_offsets[max_planet_type + 1] = static_cast<int>(type_theme_values.size());

    const int iter_count = 4;
    const int max_pose_count = star_count * iter_count;
    const int pose_copy_count = star_count;
    const int group_cap = ResolveGroupSize(seed_count, star_count, iter_count);
    bool debug_dump = false;
    if (const char* dbg = std::getenv("DSP_NATIVE_SIG_DEBUG"))
        debug_dump = std::atoi(dbg) != 0;
    bool stage_timing = false;
    if (const char* st = std::getenv("DSP_NATIVE_SIG_STAGE_TIMING"))
        stage_timing = std::atoi(st) != 0;
    enum
    {
        kStageTimingModeFull = 0,
        kStageTimingModePose = 1,
        kStageTimingModePoseGenK = 2,
        kStageTimingModeResidual = 3
    };
    int stage_timing_mode = kStageTimingModeFull;
    if (stage_timing)
    {
        std::string timing_mode = "full";
        if (const char* tm = std::getenv("DSP_NATIVE_SIG_TIMING_MODE"))
        {
            if (tm[0] != '\0')
                timing_mode = tm;
        }
        if (timing_mode == "pose")
            stage_timing_mode = kStageTimingModePose;
        else if (timing_mode == "posegenk")
            stage_timing_mode = kStageTimingModePoseGenK;
        else if (timing_mode == "residual")
            stage_timing_mode = kStageTimingModeResidual;
    }
    auto now_ms = []() -> double {
        using clock = std::chrono::steady_clock;
        return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
    };
    double stage_pose_ms = 0.0;
    double stage_pose_h2d_ms = 0.0;
    double stage_pose_gen_kernel_ms = 0.0;
    double stage_pose_d2h_counts_ms = 0.0;
    double stage_pose_gather_kernel_ms = 0.0;
    double stage_pose_d2h_head_ms = 0.0;
    double stage_pose_gen_phase1_ms = 0.0;
    double stage_pose_gen_phase2_ms = 0.0;
    double stage_pose_gen_seed_p50_ms = 0.0;
    double stage_pose_gen_seed_p95_ms = 0.0;
    double stage_pose_gen_seed_max_ms = 0.0;
    double stage_pose_gen_seed_profile_weight = 0.0;
    double stage_pose_gen_attempts_total = 0.0;
    double stage_pose_gen_collision_total = 0.0;
    double stage_pose_gen_sphere_total = 0.0;
    double stage_pose_gen_gate_total = 0.0;
    double stage_pose_d2h_head_submit_ms = 0.0;
    double stage_pose_d2h_head_sync_wait_ms = 0.0;
    double stage_pose_d2h_head_bytes_mb = 0.0;
    double stage_pose_api_pre_ms = 0.0;
    double stage_pose_api_post_ms = 0.0;
    double stage_pose_api_total_host_ms = 0.0;
    double stage_pose_set_device_ms = 0.0;
    double stage_pose_ensure_buffers_ms = 0.0;
    double stage_pose_event_setup_ms = 0.0;
    double stage_pose_fast_buffer_ms = 0.0;
    double stage_pose_fast_buffer_release_ms = 0.0;
    double stage_pose_fast_buffer_acquire_lock_wait_ms = 0.0;
    double stage_pose_fast_buffer_acquire_lock_hold_ms = 0.0;
    double stage_pose_fast_buffer_acquire_set_device_ms = 0.0;
    double stage_pose_fast_buffer_acquire_free_ms = 0.0;
    double stage_pose_fast_buffer_acquire_malloc_ms = 0.0;
    double stage_pose_fast_buffer_release_lock_wait_ms = 0.0;
    double stage_pose_fast_buffer_release_lock_hold_ms = 0.0;
    double stage_seed_build_ms = 0.0;
    double stage_seed_build_host_ctx_ms = 0.0;
    double stage_seed_build_host_merge_ms = 0.0;
    double stage_seed_build_gpu_plan_call_ms = 0.0;
    double stage_seed_build_fallback_host_ms = 0.0;
    double stage_seed_build_h2d_ms = 0.0;
    double stage_seed_build_plan_kernel_ms = 0.0;
    double stage_seed_build_core_kernel_ms = 0.0;
    double stage_seed_build_d2h_ms = 0.0;
    double stage_seed_build_host_pack_ms = 0.0;
    double stage_seed_build_alloc_ms = 0.0;
    double stage_seed_build_scatter_ms = 0.0;
    double stage_core_pack_ms = 0.0;
    double stage_core_kernel_ms = 0.0;
    double stage_core_unpack_ms = 0.0;
    double stage_theme_ms = 0.0;
    double stage_vein_pack_ms = 0.0;
    double stage_vein_kernel_ms = 0.0;
    double stage_hash_ms = 0.0;
    double stage_total_begin_ms = stage_timing ? now_ms() : 0.0;
    int host_threads = 1;
    if (const char* hs = std::getenv("DSP_NATIVE_SIG_HOST_THREADS"))
    {
        host_threads = std::atoi(hs);
        if (host_threads < 1)
            host_threads = 1;
    }
    else
    {
        host_threads = 8;
    }
    bool use_gpu_seedbuild_plan = true;
    if (const char* gp = std::getenv("DSP_NATIVE_SEEDBUILD_GPU_PLAN"))
        use_gpu_seedbuild_plan = std::atoi(gp) != 0;
    bool use_gpu_theme_vein_hash = false;
    if (const char* gh = std::getenv("DSP_NATIVE_SIG_GPU_THEME_VEIN_HASH_EXPERIMENTAL"))
        use_gpu_theme_vein_hash = std::atoi(gh) != 0;
    bool trust_gpu_theme_vein_hash = false;
    if (const char* gh = std::getenv("DSP_NATIVE_SIG_GPU_THEME_VEIN_HASH_TRUST"))
        trust_gpu_theme_vein_hash = std::atoi(gh) != 0;
    bool unsafe_gpu_theme_vein_hash = false;
    if (const char* gh = std::getenv("DSP_NATIVE_SIG_GPU_THEME_VEIN_HASH_UNSAFE"))
        unsafe_gpu_theme_vein_hash = std::atoi(gh) != 0;
    if (trust_gpu_theme_vein_hash && !unsafe_gpu_theme_vein_hash)
        trust_gpu_theme_vein_hash = false;
    bool use_gpu_pose_direct = false;
    if (const char* gp = std::getenv("DSP_NATIVE_SIG_GPU_POSE_DIRECT"))
        use_gpu_pose_direct = std::atoi(gp) != 0;
    bool speed_only = false;
    if (const char* so = std::getenv("DSP_NATIVE_SIG_SPEED_ONLY"))
        speed_only = std::atoi(so) != 0;
    bool debug_theme_compare = false;
    if (const char* dc = std::getenv("DSP_NATIVE_SIG_DEBUG_THEME_COMPARE"))
        debug_theme_compare = std::atoi(dc) != 0;
    for (int group_base = 0; group_base < seed_count; group_base += group_cap)
    {
        int group_count = std::min(group_cap, seed_count - group_base);
        std::vector<int> pose_seeds(group_count);
        std::vector<int> pose_raw_counts(group_count, 0);
        const bool gpu_sig_direct_pose_path =
            use_gpu_pose_direct && use_gpu_seedbuild_plan && use_gpu_theme_vein_hash && trust_gpu_theme_vein_hash && !debug_theme_compare;
        bool use_pose_device_fast = gpu_sig_direct_pose_path && speed_only;
        std::vector<dsp_vec3d_t> pose_raw;
        if (!use_pose_device_fast)
            pose_raw.resize(static_cast<size_t>(group_count) * static_cast<size_t>(pose_copy_count));

        for (int gi = 0; gi < group_count; ++gi)
        {
            DotNet35RandomHost rng(galaxy_seeds[group_base + gi]);
            pose_seeds[gi] = rng.Next();
        }

        dsp_vec3d_t* d_pose_raw_fast = nullptr;
        auto collect_pose_timing = [&]() {
            if (!stage_timing)
                return;
            dsp_cuda_pose_batch_head_timing_t pose_timing{};
            if (dsp_cuda_get_last_pose_batch_head_timing(&pose_timing) == DSP_CUDA_OK)
            {
                stage_pose_h2d_ms += pose_timing.h2d_seeds_ms;
                stage_pose_gen_kernel_ms += pose_timing.gen_kernel_ms;
                stage_pose_d2h_counts_ms += pose_timing.d2h_counts_ms;
                stage_pose_gather_kernel_ms += pose_timing.gather_kernel_ms;
                stage_pose_d2h_head_ms += pose_timing.d2h_head_ms;
                stage_pose_gen_phase1_ms += pose_timing.gen_phase1_ms * static_cast<double>(group_count);
                stage_pose_gen_phase2_ms += pose_timing.gen_phase2_ms * static_cast<double>(group_count);
                stage_pose_gen_seed_p50_ms += pose_timing.gen_seed_p50_ms * static_cast<double>(group_count);
                stage_pose_gen_seed_p95_ms = std::max(stage_pose_gen_seed_p95_ms, pose_timing.gen_seed_p95_ms);
                stage_pose_gen_seed_max_ms = std::max(stage_pose_gen_seed_max_ms, pose_timing.gen_seed_max_ms);
                stage_pose_gen_seed_profile_weight += static_cast<double>(group_count);
                stage_pose_gen_attempts_total += pose_timing.gen_attempts_total;
                stage_pose_gen_collision_total += pose_timing.gen_collision_rejects_total;
                stage_pose_gen_sphere_total += pose_timing.gen_sphere_rejects_total;
                stage_pose_gen_gate_total += pose_timing.gen_gate_skips_total;
                stage_pose_d2h_head_submit_ms += pose_timing.d2h_head_submit_ms;
                stage_pose_d2h_head_sync_wait_ms += pose_timing.d2h_head_sync_wait_ms;
                stage_pose_d2h_head_bytes_mb += pose_timing.d2h_head_bytes_mb;
                stage_pose_api_pre_ms += pose_timing.api_pre_ms;
                stage_pose_api_post_ms += pose_timing.api_post_ms;
                stage_pose_api_total_host_ms += pose_timing.api_total_host_ms;
                stage_pose_set_device_ms += pose_timing.set_device_ms;
                stage_pose_ensure_buffers_ms += pose_timing.ensure_buffers_ms;
                stage_pose_event_setup_ms += pose_timing.event_setup_ms;
            }
        };
        auto run_pose_host = [&]() -> int {
            if (pose_raw.size() != static_cast<size_t>(group_count) * static_cast<size_t>(pose_copy_count))
                pose_raw.resize(static_cast<size_t>(group_count) * static_cast<size_t>(pose_copy_count));
            return dsp_cuda_generate_temp_poses_params_fp64_batch_head(
                pose_seeds.data(),
                group_count,
                max_pose_count,
                pose_copy_count,
                iter_count,
                2.0,
                2.3,
                3.5,
                0.18,
                collision_fp64,
                cuda_device_id,
                pose_raw.data(),
                pose_copy_count,
                pose_raw_counts.data());
        };
        auto run_pose_device = [&]() -> int {
            const size_t pose_n = static_cast<size_t>(group_count) * static_cast<size_t>(pose_copy_count);
            MixPoseFastBufferLease pose_lease{};
            MixPoseFastBufferTiming fb_timing{};
            if (!AcquireMixPoseFastBuffer(cuda_device_id, pose_n, &pose_lease, stage_timing ? &fb_timing : nullptr))
                return DSP_CUDA_ERR_CUDA;
            if (stage_timing)
            {
                stage_pose_fast_buffer_ms += fb_timing.acquire_total_ms;
                stage_pose_fast_buffer_acquire_lock_wait_ms += fb_timing.acquire_lock_wait_ms;
                stage_pose_fast_buffer_acquire_lock_hold_ms += fb_timing.acquire_lock_hold_ms;
                stage_pose_fast_buffer_acquire_set_device_ms += fb_timing.acquire_set_device_ms;
                stage_pose_fast_buffer_acquire_free_ms += fb_timing.acquire_free_ms;
                stage_pose_fast_buffer_acquire_malloc_ms += fb_timing.acquire_malloc_ms;
            }
            d_pose_raw_fast = pose_lease.ptr;
            int pose_rc = dsp_cuda_generate_temp_poses_params_fp64_batch_head_device(
                pose_seeds.data(),
                group_count,
                max_pose_count,
                pose_copy_count,
                iter_count,
                2.0,
                2.3,
                3.5,
                0.18,
                collision_fp64,
                cuda_device_id,
                d_pose_raw_fast,
                pose_copy_count,
                pose_raw_counts.data());
            ReleaseMixPoseFastBuffer(&pose_lease, stage_timing ? &fb_timing : nullptr);
            if (stage_timing)
            {
                stage_pose_fast_buffer_release_ms += fb_timing.release_total_ms;
                stage_pose_fast_buffer_release_lock_wait_ms += fb_timing.release_lock_wait_ms;
                stage_pose_fast_buffer_release_lock_hold_ms += fb_timing.release_lock_hold_ms;
            }
            return pose_rc;
        };

        double stage0 = stage_timing ? now_ms() : 0.0;
        int rc = use_pose_device_fast ? run_pose_device() : run_pose_host();
        if (stage_timing)
            stage_pose_ms += now_ms() - stage0;
        collect_pose_timing();
        if (rc != DSP_CUDA_OK)
        {
            if (stage_timing || debug_dump || debug_enter)
            {
                std::fprintf(stderr, "[native-sig-error] pose-batch rc=%d group_count=%d max_pose_count=%d\n", rc, group_count, max_pose_count);
                std::fflush(stderr);
            }
            return rc;
        }

        if (gpu_sig_direct_pose_path)
        {
            double t_direct_begin = stage_timing ? now_ms() : 0.0;
            bool direct_ok = TryEvalThemeVeinHashGpuDirectFromPose(
                cuda_device_id,
                em,
                galaxy_seeds + group_base,
                group_count,
                star_count,
                iter_count,
                pose_raw_counts,
                pose_raw,
                use_pose_device_fast ? d_pose_raw_fast : nullptr,
                vein_len,
                use_fp32_prob_compare,
                star_type_white_dwarf,
                star_type_neutron_star,
                star_type_black_hole,
                theme_count,
                max_planet_type,
                theme_ids,
                theme_planet_types,
                theme_temperatures,
                theme_distributes,
                theme_water_item_ids,
                theme_vein_spot_offsets,
                theme_vein_spot_values,
                theme_rare_vein_offsets,
                theme_rare_vein_values,
                theme_rare_settings_offsets,
                theme_rare_settings_values,
                type_theme_offsets,
                type_theme_values,
                speed_only,
                out_galaxy_sigs + group_base,
                out_planet_sigs + group_base,
                out_vein_sigs + group_base,
                out_pipeline_sigs + group_base);
            if (stage_timing)
            {
                double dt = now_ms() - t_direct_begin;
                stage_seed_build_ms += dt;
                stage_seed_build_gpu_plan_call_ms += dt;
            }
            if (direct_ok)
                continue;
            if (use_pose_device_fast)
            {
                use_pose_device_fast = false;
                stage0 = stage_timing ? now_ms() : 0.0;
                rc = run_pose_host();
                if (stage_timing)
                    stage_pose_ms += now_ms() - stage0;
                collect_pose_timing();
                if (rc != DSP_CUDA_OK)
                {
                    if (stage_timing || debug_dump || debug_enter)
                    {
                        std::fprintf(stderr, "[native-sig-error] pose-batch-fallback rc=%d group_count=%d max_pose_count=%d\n", rc, group_count, max_pose_count);
                        std::fflush(stderr);
                    }
                    return rc;
                }
            }
            if (stage_timing || debug_dump || debug_enter)
            {
                std::fprintf(stderr, "[native-sig-info] direct-pose-gpu path fallback-to-host group_count=%d\n", group_count);
                std::fflush(stderr);
            }
        }

        stage0 = stage_timing ? now_ms() : 0.0;
        const double stage_seed_build_begin = stage0;
        std::vector<SeedCtx> seeds;
        seeds.resize(group_count);
        std::vector<PlanetRef> primary_refs;
        std::vector<PlanetRef> secondary_refs;
        primary_refs.reserve(static_cast<size_t>(group_count) * 200);
        secondary_refs.reserve(static_cast<size_t>(group_count) * 80);
        int worker_count = std::min(host_threads, group_count);
        if (worker_count < 1)
            worker_count = 1;
        std::vector<std::vector<PlanetRef>> primary_refs_tls(static_cast<size_t>(worker_count));
        std::vector<std::vector<PlanetRef>> secondary_refs_tls(static_cast<size_t>(worker_count));
        auto build_seed_range = [&](int begin, int end, int worker_idx) {
            auto& pri = primary_refs_tls[worker_idx];
            auto& sec = secondary_refs_tls[worker_idx];
            if (!use_gpu_seedbuild_plan)
            {
                pri.reserve(static_cast<size_t>(std::max(0, end - begin)) * 200);
                sec.reserve(static_cast<size_t>(std::max(0, end - begin)) * 80);
            }
            for (int gi = begin; gi < end; ++gi)
            {
                SeedCtx& seed_ctx = seeds[gi];
                seed_ctx.galaxy_seed = galaxy_seeds[group_base + gi];
                seed_ctx.birth_star_id = 0;
                seed_ctx.birth_planet_id = 0;

                const int raw_count = pose_raw_counts[gi];
                const dsp_vec3d_t* raw_base = pose_raw.data() + static_cast<size_t>(gi) * static_cast<size_t>(pose_copy_count);
                int pose_count = raw_count / iter_count;
                if (pose_count < 0)
                    pose_count = 0;
                if (pose_count > star_count)
                    pose_count = star_count;
                seed_ctx.star_count = pose_count;
                seed_ctx.stars.resize(seed_ctx.star_count);
                if (seed_ctx.star_count <= 0)
                    continue;

                DotNet35RandomHost galaxy_rng(seed_ctx.galaxy_seed);
                galaxy_rng.Next(); // consumed by pose seed

                float num1 = static_cast<float>(galaxy_rng.NextDouble());
                float num2 = static_cast<float>(galaxy_rng.NextDouble());
                float num3 = static_cast<float>(galaxy_rng.NextDouble());
                float num4 = static_cast<float>(galaxy_rng.NextDouble());

                int num5 = static_cast<int>(std::ceil(0.01 * seed_ctx.star_count + num1 * 0.300000011920929));
                int num6 = static_cast<int>(std::ceil(0.01 * seed_ctx.star_count + num2 * 0.300000011920929));
                int num7 = static_cast<int>(std::ceil(0.0160000007599592 * seed_ctx.star_count + num3 * 0.400000005960464));
                int num8 = static_cast<int>(std::ceil(0.0130000002682209 * seed_ctx.star_count + num4 * 1.39999997615814));
                int num9 = seed_ctx.star_count - num5;
                int num10 = num9 - num6;
                int num11 = num10 - num7;
                int num12 = (num11 - 1) / num8;
                int num13 = num12 / 2;

                for (int si = 0; si < seed_ctx.star_count; ++si)
                {
                    int star_seed = galaxy_rng.Next();
                    StarCtx& sc = seed_ctx.stars[si];
                    if (si == 0)
                    {
                        sc.star = CreateBirthStarLite(seed_ctx.star_count, star_seed, em);
                    }
                    else
                    {
                        int need_spectr = em.spectr_x;
                        if (si == 3) need_spectr = em.spectr_m;
                        else if (si == num11 - 1) need_spectr = em.spectr_o;

                        int need_type = em.star_type_main_seq;
                        if (num12 != 0 && si % num12 == num13)
                            need_type = em.star_type_giant;
                        if (si >= num9)
                            need_type = em.star_type_black_hole;
                        else if (si >= num10)
                            need_type = em.star_type_neutron_star;
                        else if (si >= num11)
                            need_type = em.star_type_white_dwarf;

                        sc.star = CreateStarLite(seed_ctx.star_count, raw_base[si], si + 1, star_seed, need_type, need_spectr, em);
                    }
                    if (!use_gpu_seedbuild_plan)
                    {
                        BuildStarPlanetPlans(sc, em);
                        for (size_t pi = 0; pi < sc.planets.size(); ++pi)
                        {
                            PlanetRef pr{gi, si, static_cast<int>(pi)};
                            if (sc.planets[pi].orbit_around == 0)
                                pri.push_back(pr);
                            else
                                sec.push_back(pr);
                        }
                    }
                }
            }
        };

        if (worker_count <= 1)
        {
            build_seed_range(0, group_count, 0);
        }
        else
        {
            std::vector<std::thread> workers;
            workers.reserve(static_cast<size_t>(worker_count - 1));
            int chunk_size = (group_count + worker_count - 1) / worker_count;
            for (int w = 1; w < worker_count; ++w)
            {
                int begin = w * chunk_size;
                int end = std::min(group_count, begin + chunk_size);
                if (begin >= end)
                    break;
                workers.emplace_back([&, begin, end, w]() { build_seed_range(begin, end, w); });
            }
            int begin0 = 0;
            int end0 = std::min(group_count, chunk_size);
            build_seed_range(begin0, end0, 0);
            for (auto& th : workers)
                th.join();
        }
        if (stage_timing)
        {
            const double after_build_ms = now_ms();
            stage_seed_build_host_ctx_ms += (after_build_ms - stage0);
            stage0 = after_build_ms;
        }
        for (int w = 0; w < worker_count; ++w)
        {
            if (!use_gpu_seedbuild_plan)
            {
                primary_refs.insert(primary_refs.end(), primary_refs_tls[w].begin(), primary_refs_tls[w].end());
                secondary_refs.insert(secondary_refs.end(), secondary_refs_tls[w].begin(), secondary_refs_tls[w].end());
            }
        }
        if (stage_timing)
        {
            const double after_merge_ms = now_ms();
            stage_seed_build_host_merge_ms += (after_merge_ms - stage0);
            stage0 = after_merge_ms;
        }
        const bool gpu_sig_direct_path = use_gpu_seedbuild_plan && use_gpu_theme_vein_hash && trust_gpu_theme_vein_hash;
        bool core_ready_from_plan = false;
        std::vector<CoreLite> out_core_flat;
        std::vector<int> gpu_plan_planet_counts;
        std::vector<int> gpu_plan_orbit_arounds;
        std::vector<int> gpu_plan_orbit_indexes;
        std::vector<int> gpu_plan_gas_giants;
        std::vector<int> gpu_plan_gen_seeds;
        if (use_gpu_seedbuild_plan)
        {
            SeedBuildCudaTiming seedbuild_timing{};

            bool gpu_plan_ok = TryBuildStarPlanetPlansCuda(
                cuda_device_id,
                em,
                seeds,
                star_type_white_dwarf,
                star_type_neutron_star,
                star_type_black_hole,
                false,
                false,
                worker_count,
                primary_refs,
                secondary_refs,
                out_core_flat,
                stage_timing ? &seedbuild_timing : nullptr,
                gpu_sig_direct_path ? &gpu_plan_planet_counts : nullptr,
                gpu_sig_direct_path ? &gpu_plan_orbit_arounds : nullptr,
                gpu_sig_direct_path ? &gpu_plan_orbit_indexes : nullptr,
                gpu_sig_direct_path ? &gpu_plan_gas_giants : nullptr,
                gpu_sig_direct_path ? &gpu_plan_gen_seeds : nullptr,
                gpu_sig_direct_path);
            if (stage_timing)
            {
                const double after_gpu_plan_ms = now_ms();
                stage_seed_build_gpu_plan_call_ms += (after_gpu_plan_ms - stage0);
                stage0 = after_gpu_plan_ms;
            }
            if (stage_timing)
            {
                stage_seed_build_h2d_ms += seedbuild_timing.h2d_ms;
                stage_seed_build_plan_kernel_ms += seedbuild_timing.plan_kernel_ms;
                stage_seed_build_core_kernel_ms += seedbuild_timing.core_kernel_ms;
                stage_seed_build_d2h_ms += seedbuild_timing.d2h_ms;
                stage_seed_build_host_pack_ms += seedbuild_timing.host_pack_ms;
                stage_seed_build_alloc_ms += seedbuild_timing.alloc_ms;
                stage_seed_build_scatter_ms += seedbuild_timing.scatter_ms;
            }
            core_ready_from_plan = gpu_plan_ok;
            if (!gpu_plan_ok)
            {
                if (stage_timing || debug_dump)
                {
                    std::printf("[native-sig-info] gpu-seedbuild-plan fallback-to-host group_count=%d\n", group_count);
                    std::fflush(stdout);
                }
                primary_refs.clear();
                secondary_refs.clear();
                primary_refs.reserve(static_cast<size_t>(group_count) * 200);
                secondary_refs.reserve(static_cast<size_t>(group_count) * 80);
                for (int gi = 0; gi < group_count; ++gi)
                {
                    SeedCtx& seed_ctx = seeds[gi];
                    for (size_t si = 0; si < seed_ctx.stars.size(); ++si)
                    {
                        StarCtx& sc = seed_ctx.stars[si];
                        sc.planets.clear();
                        BuildStarPlanetPlans(sc, em);
                        for (size_t pi = 0; pi < sc.planets.size(); ++pi)
                        {
                            PlanetRef pr{gi, static_cast<int>(si), static_cast<int>(pi)};
                            if (sc.planets[pi].orbit_around == 0)
                                primary_refs.push_back(pr);
                            else
                                secondary_refs.push_back(pr);
                        }
                    }
                }
                if (stage_timing)
                {
                    const double after_fallback_ms = now_ms();
                    stage_seed_build_fallback_host_ms += (after_fallback_ms - stage0);
                    stage0 = after_fallback_ms;
                }
            }
        }
        if (stage_timing)
            stage_seed_build_ms += now_ms() - stage_seed_build_begin;

        auto run_group_parallel = [&](const auto& fn) {
            if (worker_count <= 1 || group_count <= 1)
            {
                fn(0, group_count, 0);
                return;
            }
            int chunk_size = (group_count + worker_count - 1) / worker_count;
            std::vector<std::thread> workers;
            workers.reserve(static_cast<size_t>(worker_count - 1));
            for (int w = 1; w < worker_count; ++w)
            {
                int begin = w * chunk_size;
                int end = std::min(group_count, begin + chunk_size);
                if (begin >= end)
                    break;
                workers.emplace_back([&, begin, end, w]() { fn(begin, end, w); });
            }
            int begin0 = 0;
            int end0 = std::min(group_count, chunk_size);
            fn(begin0, end0, 0);
            for (auto& th : workers)
                th.join();
        };

        auto run_core_phase = [&](const std::vector<PlanetRef>& refs, bool secondary_phase) -> int {
            if (refs.empty())
                return DSP_CUDA_OK;

            const int n = static_cast<int>(refs.size());
            double t_pack = stage_timing ? now_ms() : 0.0;
            std::vector<int> info_seeds(n);
            std::vector<int> orbit_arounds(n);
            std::vector<int> orbit_indexes(n);
            std::vector<int> gas_giants(n);
            std::vector<int> star_indexes(n);
            std::vector<int> galaxy_star_counts(n);
            std::vector<int> galaxy_habitable_counts(n, 0);
            std::vector<int> boost_inclination_ns(n);
            std::vector<int> compact_type_cases(n);
            std::vector<float> star_orbit_scalers(n);
            std::vector<double> star_masses(n);
            std::vector<float> star_habitable_radiuses(n);
            std::vector<float> star_light_balance_radiuses(n);
            std::vector<float> around_real_radiuses(n, 0.0f);
            std::vector<float> around_orbit_radiuses(n, 0.0f);
            std::vector<double> around_orbital_periods(n, 0.0);
            std::vector<dsp_planet_core_f32_out_t> out(n);

            for (int i = 0; i < n; ++i)
            {
                const PlanetRef& ref = refs[i];
                SeedCtx& seed_ctx = seeds[ref.seed_idx];
                StarCtx& star_ctx = seed_ctx.stars[ref.star_idx];
                PlanetPlanLite& pp = star_ctx.planets[ref.planet_idx];

                info_seeds[i] = pp.info_seed;
                orbit_arounds[i] = pp.orbit_around;
                orbit_indexes[i] = pp.orbit_index;
                gas_giants[i] = pp.gas_giant ? 1 : 0;
                star_indexes[i] = star_ctx.star.index;
                galaxy_star_counts[i] = seed_ctx.star_count;
                boost_inclination_ns[i] = BoostInclinationNsByStarType(star_ctx.star.type, em) ? 1 : 0;
                compact_type_cases[i] = CompactTypeCaseByStarType(star_ctx.star.type, em);
                star_orbit_scalers[i] = star_ctx.star.orbit_scaler;
                star_masses[i] = star_ctx.star.mass;
                star_habitable_radiuses[i] = star_ctx.star.habitable_radius;
                star_light_balance_radiuses[i] = star_ctx.star.light_balance_radius;
                if (secondary_phase)
                {
                    if (pp.parent_planet_index < 0 || pp.parent_planet_index >= static_cast<int>(star_ctx.planets.size()))
                    {
                        if (stage_timing || debug_dump)
                        {
                            std::printf(
                                "[native-sig-error] secondary-parent-index invalid seedIdx=%d starIdx=%d planetIdx=%d parent=%d starPlanets=%d\n",
                                ref.seed_idx, ref.star_idx, ref.planet_idx, pp.parent_planet_index, static_cast<int>(star_ctx.planets.size()));
                            std::fflush(stdout);
                        }
                        return DSP_CUDA_ERR_INVALID_ARGUMENT;
                    }
                    const PlanetPlanLite& parent = star_ctx.planets[pp.parent_planet_index];
                    around_real_radiuses[i] = parent.core.radius * parent.core.scale;
                    around_orbit_radiuses[i] = parent.core.orbit_radius;
                    around_orbital_periods[i] = parent.core.orbital_period;
                }
            }
            if (stage_timing)
                stage_core_pack_ms += now_ms() - t_pack;

            double t_kernel = stage_timing ? now_ms() : 0.0;
            int rc = dsp_cuda_planet_eval_core_f32_batch(
                info_seeds.data(),
                orbit_arounds.data(),
                orbit_indexes.data(),
                gas_giants.data(),
                star_indexes.data(),
                galaxy_star_counts.data(),
                galaxy_habitable_counts.data(),
                boost_inclination_ns.data(),
                compact_type_cases.data(),
                star_orbit_scalers.data(),
                star_masses.data(),
                star_habitable_radiuses.data(),
                star_light_balance_radiuses.data(),
                around_real_radiuses.data(),
                around_orbit_radiuses.data(),
                around_orbital_periods.data(),
                n,
                cuda_device_id,
                out.data());
            if (stage_timing)
                stage_core_kernel_ms += now_ms() - t_kernel;
            if (rc != DSP_CUDA_OK)
            {
                if (stage_timing || debug_dump)
                {
                    std::printf("[native-sig-error] core-batch rc=%d secondary=%d n=%d\n", rc, secondary_phase ? 1 : 0, n);
                    std::fflush(stdout);
                }
                return rc;
            }

            double t_unpack = stage_timing ? now_ms() : 0.0;
            for (int i = 0; i < n; ++i)
            {
                const PlanetRef& ref = refs[i];
                PlanetPlanLite& pp = seeds[ref.seed_idx].stars[ref.star_idx].planets[ref.planet_idx];
                pp.core.orbit_radius = out[i].orbit_radius;
                pp.core.orbital_period = out[i].orbital_period;
                pp.core.scale = out[i].scale;
                pp.core.radius = out[i].radius;
                pp.core.habitable_bias = out[i].habitable_bias;
                pp.core.sun_distance = out[i].sun_distance;
                pp.core.temperature_bias = out[i].temperature_bias;
                pp.core.num13 = out[i].num13;
                pp.core.num14 = out[i].num14;
                pp.core.rand1 = out[i].rand1;
                pp.core.theme_seed = out[i].theme_seed;
            }
            if (stage_timing)
                stage_core_unpack_ms += now_ms() - t_unpack;
            return DSP_CUDA_OK;
        };

        if (!core_ready_from_plan)
        {
            rc = run_core_phase(primary_refs, false);
            if (rc != DSP_CUDA_OK)
            {
                if (stage_timing || debug_dump)
                {
                    std::printf("[native-sig-error] run_core_phase primary rc=%d refs=%zu\n", rc, primary_refs.size());
                    std::fflush(stdout);
                }
                return rc;
            }
            rc = run_core_phase(secondary_refs, true);
            if (rc != DSP_CUDA_OK)
            {
                if (stage_timing || debug_dump)
                {
                    std::printf("[native-sig-error] run_core_phase secondary rc=%d refs=%zu\n", rc, secondary_refs.size());
                    std::fflush(stdout);
                }
                return rc;
            }
        }

        bool gpu_theme_vein_hash_ready = false;
        std::vector<int> gpu_theme_idx_dbg;
        SigGpuFlatSoA gpu_flat_dbg;
        bool gpu_theme_dbg_ready = false;
        if (use_gpu_theme_vein_hash && (trust_gpu_theme_vein_hash || debug_theme_compare))
        {
            double t_gpu_sig = stage_timing ? now_ms() : 0.0;
            SigGpuFlatSoA flat{};
            std::vector<int> gpu_birth_star_dbg;
            std::vector<int> gpu_birth_planet_dbg;
            if (debug_theme_compare)
            {
                gpu_birth_star_dbg.resize(static_cast<size_t>(group_count), 0);
                gpu_birth_planet_dbg.resize(static_cast<size_t>(group_count), 0);
            }
            bool flat_ready = false;
            if (gpu_sig_direct_path && core_ready_from_plan)
            {
                flat_ready = BuildSigGpuFlatSoAFromPlan(
                    seeds,
                    gpu_plan_planet_counts,
                    gpu_plan_orbit_arounds,
                    gpu_plan_orbit_indexes,
                    gpu_plan_gas_giants,
                    gpu_plan_gen_seeds,
                    out_core_flat,
                    flat);
            }
            if (!flat_ready)
            {
                BuildSigGpuFlatSoA(seeds, flat);
                flat_ready = true;
            }
            if (debug_theme_compare)
            {
                const int total_planets_dbg = flat.star_planet_offsets.empty() ? 0 : flat.star_planet_offsets.back();
                gpu_theme_idx_dbg.assign(static_cast<size_t>(std::max(0, total_planets_dbg)), 0);
                gpu_flat_dbg = flat;
                gpu_theme_dbg_ready = true;
            }
            bool gpu_sig_ok = TryEvalThemeVeinHashGpu(
                cuda_device_id,
                group_count,
                vein_len,
                use_fp32_prob_compare,
                em,
                theme_count,
                max_planet_type,
                theme_ids,
                theme_planet_types,
                theme_temperatures,
                theme_distributes,
                theme_water_item_ids,
                theme_vein_spot_offsets,
                theme_vein_spot_values,
                theme_rare_vein_offsets,
                theme_rare_vein_values,
                theme_rare_settings_offsets,
                theme_rare_settings_values,
                type_theme_offsets,
                type_theme_values,
                flat,
                speed_only,
                out_galaxy_sigs + group_base,
                out_planet_sigs + group_base,
                out_vein_sigs + group_base,
                out_pipeline_sigs + group_base,
                debug_theme_compare ? gpu_birth_star_dbg.data() : nullptr,
                debug_theme_compare ? gpu_birth_planet_dbg.data() : nullptr,
                debug_theme_compare ? gpu_theme_idx_dbg.data() : nullptr);
            if (stage_timing)
                stage_theme_ms += now_ms() - t_gpu_sig;
            if (debug_theme_compare && !gpu_birth_star_dbg.empty() && !gpu_birth_planet_dbg.empty())
            {
                std::fprintf(stderr,
                    "[native-sig-theme-debug] groupBase=%d gpuBirth(seed0): star=%d planet=%d sig(g/p/v/pi)=0x%016llX/0x%016llX/0x%016llX/0x%016llX\n",
                    group_base,
                    gpu_birth_star_dbg[0],
                    gpu_birth_planet_dbg[0],
                    static_cast<unsigned long long>(out_galaxy_sigs[group_base + 0]),
                    static_cast<unsigned long long>(out_planet_sigs[group_base + 0]),
                    static_cast<unsigned long long>(out_vein_sigs[group_base + 0]),
                    static_cast<unsigned long long>(out_pipeline_sigs[group_base + 0]));
                std::fflush(stderr);
            }
            if (gpu_sig_ok && trust_gpu_theme_vein_hash)
            {
                gpu_theme_vein_hash_ready = true;
            }
            else if (!gpu_sig_ok && gpu_sig_direct_path)
            {
                if (stage_timing || debug_dump)
                {
                    std::printf("[native-sig-error] gpu-theme-vein-hash failed in direct path group_count=%d\n", group_count);
                    std::fflush(stdout);
                }
                return DSP_CUDA_ERR_CUDA;
            }
            else if (!gpu_sig_ok && (stage_timing || debug_dump))
            {
                std::printf("[native-sig-info] gpu-theme-vein-hash fallback-to-host group_count=%d\n", group_count);
                std::fflush(stdout);
            }
            else if (stage_timing || debug_dump || debug_theme_compare)
            {
                std::printf("[native-sig-info] gpu-theme-vein-hash trust-disabled fallback-to-host group_count=%d\n", group_count);
                std::fflush(stdout);
            }
        }
        if (gpu_theme_vein_hash_ready)
            continue;

        stage0 = stage_timing ? now_ms() : 0.0;
        std::atomic<int> theme_rc(DSP_CUDA_OK);
        run_group_parallel([&](int begin, int end, int) {
            if (theme_rc.load(std::memory_order_relaxed) != DSP_CUDA_OK)
                return;
            for (int gi = begin; gi < end; ++gi)
            {
                SeedCtx& seed_ctx = seeds[gi];
                seed_ctx.birth_planet_id = 0;
                seed_ctx.birth_star_id = 0;
                int habitable_count = 0;
                for (size_t si = 0; si < seed_ctx.stars.size(); ++si)
                {
                    StarCtx& star_ctx = seed_ctx.stars[si];
                    std::vector<int> assigned;
                    assigned.assign(star_ctx.star.planet_count > 0 ? star_ctx.star.planet_count : 0, 0);
                    for (size_t pi = 0; pi < star_ctx.planets.size(); ++pi)
                    {
                        PlanetPlanLite& pp = star_ctx.planets[pi];
                        int type_case = 0;
                        int habitable_delta = 0;
                        if (pp.gas_giant)
                        {
                            type_case = 0;
                        }
                        else
                        {
                            float num18 = std::ceil(seed_ctx.star_count * 0.29f);
                            if (num18 < 11.0f) num18 = 11.0f;
                            float num19 = num18 - habitable_count;
                            float num20 = seed_ctx.star_count - star_ctx.star.index;
                            float num24 = Clamp((num19 / std::max(1.0f, num20)) * 0.5f + 0.175f, 0.08f, 0.8f);
                            float num25 = std::pow(Clamp01(pp.core.habitable_bias / num24), num24 * 10.0f);
                            float f2 = 1000.0f;
                            if (star_ctx.star.habitable_radius > 0.0f && pp.core.sun_distance > 0.0f)
                                f2 = pp.core.sun_distance / star_ctx.star.habitable_radius;

                            if ((pp.core.num13 > num25 && star_ctx.star.index > 0) ||
                                (pp.orbit_around > 0 && pp.orbit_index == 1 && star_ctx.star.index == 0))
                            {
                                type_case = 1;
                                habitable_delta = 1;
                            }
                            else if (f2 < 0.833333f)
                            {
                                float num26 = std::max(0.15f, f2 * 2.5f - 0.85f);
                                type_case = pp.core.num14 >= num26 ? 2 : 3;
                            }
                            else if (f2 < 1.2f)
                            {
                                type_case = 3;
                            }
                            else
                            {
                                float num27 = 0.9f / f2 - 0.1f;
                                type_case = pp.core.num14 >= num27 ? 4 : 3;
                            }
                        }

                        int provisional_type = MapTypeCaseToPlanetType(type_case, em);
                        if (habitable_delta > 0)
                            ++habitable_count;

                        auto pick_theme_index = [&]() -> int {
                            thread_local std::vector<int> candidates;
                            candidates.clear();
                            if (candidates.capacity() < 64)
                                candidates.reserve(64);
                            auto append_unique = [&](int ti) {
                                int pidx = pp.index;
                                bool ok = true;
                                for (int j = 0; j < pidx; ++j)
                                {
                                    if (j < static_cast<int>(assigned.size()) && assigned[j] == themes[ti].id)
                                    {
                                        ok = false;
                                        break;
                                    }
                                }
                                if (ok)
                                    candidates.push_back(ti);
                            };

                            if (provisional_type >= 0 && provisional_type <= max_planet_type)
                            {
                                const auto& type_themes = themes_by_planet_type[provisional_type];
                                for (int ti : type_themes)
                                {
                                    const ThemeLite& t = themes[ti];
                                    bool ok = false;
                                    if (star_ctx.star.index == 0 && provisional_type == em.planet_type_ocean)
                                    {
                                        ok = t.distribute == em.theme_distribute_birth;
                                    }
                                    else
                                    {
                                        const double temp_gate = -0.10000000149011612;
                                        const double temp_gate_eps = 2e-8;
                                        double temp_prod = static_cast<double>(t.temperature) * static_cast<double>(pp.core.temperature_bias);
                                        bool temp_ok = temp_prod >= (temp_gate + temp_gate_eps);
                                        if (std::fabs(t.temperature) < 0.5f && t.planet_type == em.planet_type_desert)
                                            temp_ok = std::fabs(pp.core.temperature_bias) < std::fabs(t.temperature) + 0.1f;

                                        if (t.planet_type == provisional_type && temp_ok)
                                        {
                                            if (star_ctx.star.index == 0)
                                                ok = t.distribute == em.theme_distribute_default;
                                            else
                                                ok = t.distribute == em.theme_distribute_default || t.distribute == em.theme_distribute_interstellar;
                                        }
                                    }
                                    if (ok)
                                        append_unique(ti);
                                }
                            }

                            if (candidates.empty())
                            {
                                if (em.planet_type_desert >= 0 && em.planet_type_desert <= max_planet_type)
                                {
                                    const auto& desert_themes = themes_by_planet_type[em.planet_type_desert];
                                    for (int ti : desert_themes)
                                        append_unique(ti);
                                }
                            }
                            if (candidates.empty())
                            {
                                if (em.planet_type_desert >= 0 && em.planet_type_desert <= max_planet_type)
                                {
                                    const auto& desert_themes = themes_by_planet_type[em.planet_type_desert];
                                    for (int ti : desert_themes)
                                        candidates.push_back(ti);
                                }
                            }
                            if (candidates.empty())
                                return 0;

                            int pick = static_cast<int>(pp.core.rand1 * static_cast<double>(candidates.size()));
                            if (pick < 0) pick = 0;
                            pick %= static_cast<int>(candidates.size());
                            return candidates[pick];
                        };

                        int theme_idx = pick_theme_index();
                        if (theme_idx < 0 || theme_idx >= theme_count)
                        {
                            theme_rc.store(DSP_CUDA_ERR_INVALID_ARGUMENT, std::memory_order_relaxed);
                            return;
                        }
                        const ThemeLite& t = themes[theme_idx];
                        pp.theme = t.id;
                        pp.theme_index = theme_idx;
                        pp.water_item_id = t.water_item_id;
                        pp.type = t.planet_type;
                        pp.is_gas_final = (pp.type == em.planet_type_gas);
                        if (pp.index >= 0 && pp.index < static_cast<int>(assigned.size()))
                            assigned[pp.index] = pp.theme;

                        if (seed_ctx.birth_planet_id == 0 && star_ctx.star.index == 0 && t.distribute == em.theme_distribute_birth)
                        {
                            seed_ctx.birth_planet_id = pp.id;
                            seed_ctx.birth_star_id = star_ctx.star.id;
                        }
                    }
                }
            }
        });
        if (theme_rc.load(std::memory_order_relaxed) != DSP_CUDA_OK)
            return theme_rc.load(std::memory_order_relaxed);
        if (stage_timing)
            stage_theme_ms += now_ms() - stage0;
        if (debug_theme_compare && !seeds.empty())
        {
            const SeedCtx& s0 = seeds[0];
            std::fprintf(stderr,
                "[native-sig-theme-debug] groupBase=%d hostBirth(seed0): star=%d planet=%d\n",
                group_base,
                s0.birth_star_id,
                s0.birth_planet_id);
            std::fflush(stderr);
        }
        if (debug_theme_compare && gpu_theme_dbg_ready)
        {
            std::vector<int> host_theme_idx_dbg;
            const int total_planets_dbg = gpu_flat_dbg.star_planet_offsets.empty() ? 0 : gpu_flat_dbg.star_planet_offsets.back();
            host_theme_idx_dbg.reserve(static_cast<size_t>(std::max(0, total_planets_dbg)));
            for (const SeedCtx& seed_ctx : seeds)
            {
                for (const StarCtx& star_ctx : seed_ctx.stars)
                {
                    for (const PlanetPlanLite& pp : star_ctx.planets)
                        host_theme_idx_dbg.push_back(pp.theme_index);
                }
            }
            int cmp_n = std::min(static_cast<int>(host_theme_idx_dbg.size()), static_cast<int>(gpu_theme_idx_dbg.size()));
            int first_diff = -1;
            for (int i = 0; i < cmp_n; ++i)
            {
                if (host_theme_idx_dbg[i] != gpu_theme_idx_dbg[i])
                {
                    first_diff = i;
                    break;
                }
            }
            if (first_diff >= 0)
            {
                std::fprintf(stderr,
                    "[native-sig-theme-debug] groupBase=%d firstThemeDiff idx=%d host=%d gpu=%d total=%d\n",
                    group_base,
                    first_diff,
                    host_theme_idx_dbg[first_diff],
                    gpu_theme_idx_dbg[first_diff],
                    cmp_n);
                int star_flat_dbg = -1;
                for (int s = 0; s + 1 < static_cast<int>(gpu_flat_dbg.star_planet_offsets.size()); ++s)
                {
                    if (gpu_flat_dbg.star_planet_offsets[s] <= first_diff && first_diff < gpu_flat_dbg.star_planet_offsets[s + 1])
                    {
                        star_flat_dbg = s;
                        break;
                    }
                }
                if (star_flat_dbg >= 0)
                {
                    int seed_dbg = -1;
                    for (int si = 0; si + 1 < static_cast<int>(gpu_flat_dbg.seed_star_offsets.size()); ++si)
                    {
                        if (gpu_flat_dbg.seed_star_offsets[si] <= star_flat_dbg && star_flat_dbg < gpu_flat_dbg.seed_star_offsets[si + 1])
                        {
                            seed_dbg = si;
                            break;
                        }
                    }
                    int p_local_dbg = first_diff - gpu_flat_dbg.star_planet_offsets[star_flat_dbg];
                    std::fprintf(
                        stderr,
                        "[native-sig-theme-debug] diffPos seed=%d starFlat=%d starId=%d starType=%d starSpectr=%d starIndex=%d starHab=%.9g pLocal=%d orbitIdx=%d orbitAround=%d gas=%d tempBias=%.9g habBias=%.9g sunDist=%.9g num13=%.17g num14=%.17g rand1=%.17g\n",
                        seed_dbg,
                        star_flat_dbg,
                        gpu_flat_dbg.star_ids[star_flat_dbg],
                        gpu_flat_dbg.star_types[star_flat_dbg],
                        gpu_flat_dbg.star_spectrs[star_flat_dbg],
                        gpu_flat_dbg.star_indexes[star_flat_dbg],
                        static_cast<double>(gpu_flat_dbg.star_habitable_radiuses[star_flat_dbg]),
                        p_local_dbg,
                        gpu_flat_dbg.planet_orbit_indexes[first_diff],
                        gpu_flat_dbg.planet_orbit_arounds[first_diff],
                        gpu_flat_dbg.planet_gas_giants[first_diff],
                        static_cast<double>(gpu_flat_dbg.planet_core_temperature_bias[first_diff]),
                        static_cast<double>(gpu_flat_dbg.planet_core_habitable_bias[first_diff]),
                        static_cast<double>(gpu_flat_dbg.planet_core_sun_distance[first_diff]),
                        gpu_flat_dbg.planet_core_num13[first_diff],
                        gpu_flat_dbg.planet_core_num14[first_diff],
                        gpu_flat_dbg.planet_core_rand1[first_diff]);
                }
            }
            else
            {
                std::fprintf(stderr,
                    "[native-sig-theme-debug] groupBase=%d themeIdxAllEqual total=%d\n",
                    group_base,
                    cmp_n);
            }
            std::fflush(stderr);
        }

        struct SolidRef
        {
            int seed_idx;
            int star_idx;
            int planet_idx;
        };
        std::vector<SolidRef> solids;
        solids.reserve(static_cast<size_t>(group_count) * 240);
        std::vector<std::vector<SolidRef>> solids_tls(static_cast<size_t>(worker_count));
        run_group_parallel([&](int begin, int end, int worker_idx) {
            auto& local = solids_tls[worker_idx];
            for (int gi = begin; gi < end; ++gi)
            {
                const SeedCtx& seed_ctx = seeds[gi];
                for (size_t si = 0; si < seed_ctx.stars.size(); ++si)
                {
                    const StarCtx& star_ctx = seed_ctx.stars[si];
                    for (size_t pi = 0; pi < star_ctx.planets.size(); ++pi)
                    {
                        if (!star_ctx.planets[pi].is_gas_final)
                            local.push_back(SolidRef{gi, static_cast<int>(si), static_cast<int>(pi)});
                    }
                }
            }
        });
        for (int w = 0; w < worker_count; ++w)
            solids.insert(solids.end(), solids_tls[w].begin(), solids_tls[w].end());

        std::vector<int> vein_counts;
        if (!solids.empty())
        {
            stage0 = stage_timing ? now_ms() : 0.0;
            int solid_count = static_cast<int>(solids.size());
            std::vector<int> planet_seeds(solid_count);
            std::vector<float> p_values(solid_count);
            std::vector<int> bonus_cases(solid_count);
            std::vector<int> is_birth_stars(solid_count);
            std::vector<int> theme_indexes(solid_count);
            vein_counts.resize(static_cast<size_t>(solid_count) * static_cast<size_t>(vein_len), 0);

            for (int i = 0; i < solid_count; ++i)
            {
                const SolidRef& sr = solids[i];
                const SeedCtx& seed_ctx = seeds[sr.seed_idx];
                const StarCtx& star_ctx = seed_ctx.stars[sr.star_idx];
                const PlanetPlanLite& pp = star_ctx.planets[sr.planet_idx];

                planet_seeds[i] = pp.gen_seed;
                p_values[i] = CalcPAndBonusCase(star_ctx.star.type, star_ctx.star.spectr, em, bonus_cases[i]);
                is_birth_stars[i] = star_ctx.star.index == 0 ? 1 : 0;

                if (pp.theme_index < 0 || pp.theme_index >= theme_count)
                {
                    if (stage_timing || debug_dump)
                    {
                        std::printf(
                            "[native-sig-error] solid-theme-index invalid seedIdx=%d starIdx=%d planetIdx=%d theme_index=%d theme_count=%d\n",
                            sr.seed_idx, sr.star_idx, sr.planet_idx, pp.theme_index, theme_count);
                        std::fflush(stdout);
                    }
                    return DSP_CUDA_ERR_INVALID_ARGUMENT;
                }
                theme_indexes[i] = pp.theme_index;
            }
            if (stage_timing)
                stage_vein_pack_ms += now_ms() - stage0;

            double t_vein = stage_timing ? now_ms() : 0.0;
            rc = dsp_cuda_mix_chunk_eval_veins_by_theme_f32(
                planet_seeds.data(),
                p_values.data(),
                bonus_cases.data(),
                is_birth_stars.data(),
                theme_indexes.data(),
                solid_count,
                vein_len,
                use_fp32_prob_compare,
                cuda_device_id,
                theme_vein_spot_offsets,
                theme_vein_spot_values,
                theme_rare_vein_offsets,
                theme_rare_vein_values,
                theme_rare_settings_offsets,
                theme_rare_settings_values,
                theme_count,
                vein_counts.data());
            if (stage_timing)
                stage_vein_kernel_ms += now_ms() - t_vein;
            if (rc != DSP_CUDA_OK)
            {
                if (stage_timing || debug_dump)
                {
                    int tv_end = theme_vein_spot_offsets != nullptr ? theme_vein_spot_offsets[theme_count] : -1;
                    int tr_end = theme_rare_vein_offsets != nullptr ? theme_rare_vein_offsets[theme_count] : -1;
                    int ts_end = theme_rare_settings_offsets != nullptr ? theme_rare_settings_offsets[theme_count] : -1;
                    std::printf(
                        "[native-sig-error] vein_by_theme rc=%d solid=%d themeCount=%d veinTotal=%d rareTotal=%d settingsTotal=%d\n",
                        rc, solid_count, theme_count, tv_end, tr_end, ts_end);
                    std::fflush(stdout);
                }
                return rc;
            }
        }

        stage0 = stage_timing ? now_ms() : 0.0;
        int solid_cursor = 0;
        for (int gi = 0; gi < group_count; ++gi)
        {
            const SeedCtx& seed_ctx = seeds[gi];
            unsigned long long h_galaxy = kFnvOffset;
            unsigned long long h_planet = kFnvOffset;
            unsigned long long h_vein = kFnvOffset;
            unsigned long long h_pipeline = kFnvOffset;

            int birth_star_hash = seed_ctx.birth_star_id;
            int birth_planet_hash = seed_ctx.birth_planet_id;
            h_galaxy = MixHash(h_galaxy, seed_ctx.star_count);
            h_planet = MixHash(h_planet, seed_ctx.star_count);
            h_planet = MixHash(h_planet, birth_star_hash);
            h_planet = MixHash(h_planet, birth_planet_hash);
            h_vein = MixHash(h_vein, seed_ctx.star_count);
            h_pipeline = MixHash(h_pipeline, seed_ctx.star_count);
            h_pipeline = MixHash(h_pipeline, birth_star_hash);
            h_pipeline = MixHash(h_pipeline, birth_planet_hash);

            for (const StarCtx& star_ctx : seed_ctx.stars)
            {
                h_galaxy = MixHash(h_galaxy, star_ctx.star.id);
                h_galaxy = MixHash(h_galaxy, star_ctx.star.type);
                h_galaxy = MixHash(h_galaxy, star_ctx.star.spectr);
                h_galaxy = MixHash(h_galaxy, star_ctx.star.planet_count);
                h_galaxy = MixHash(h_galaxy, star_ctx.star.pos_qx);
                h_galaxy = MixHash(h_galaxy, star_ctx.star.pos_qy);
                h_galaxy = MixHash(h_galaxy, star_ctx.star.pos_qz);

                h_planet = MixHash(h_planet, star_ctx.star.id);
                h_planet = MixHash(h_planet, star_ctx.star.type);
                h_planet = MixHash(h_planet, star_ctx.star.spectr);
                h_planet = MixHash(h_planet, star_ctx.star.planet_count);
                h_planet = MixHash(h_planet, star_ctx.star.pos_qx);
                h_planet = MixHash(h_planet, star_ctx.star.pos_qy);
                h_planet = MixHash(h_planet, star_ctx.star.pos_qz);

                h_vein = MixHash(h_vein, star_ctx.star.id);

                h_pipeline = MixHash(h_pipeline, star_ctx.star.id);
                h_pipeline = MixHash(h_pipeline, star_ctx.star.type);
                h_pipeline = MixHash(h_pipeline, star_ctx.star.spectr);
                h_pipeline = MixHash(h_pipeline, star_ctx.star.planet_count);
                h_pipeline = MixHash(h_pipeline, star_ctx.star.pos_qx);
                h_pipeline = MixHash(h_pipeline, star_ctx.star.pos_qy);
                h_pipeline = MixHash(h_pipeline, star_ctx.star.pos_qz);

                for (const PlanetPlanLite& pp : star_ctx.planets)
                {
                    h_planet = MixHash(h_planet, pp.type);
                    h_planet = MixHash(h_planet, pp.theme);
                    h_planet = MixHash(h_planet, pp.water_item_id);
                    h_planet = MixHash(h_planet, pp.orbit_index);
                    h_planet = MixHash(h_planet, pp.orbit_around);

                    h_vein = MixHash(h_vein, pp.id);

                    h_pipeline = MixHash(h_pipeline, pp.type);
                    h_pipeline = MixHash(h_pipeline, pp.theme);
                    h_pipeline = MixHash(h_pipeline, pp.water_item_id);
                    h_pipeline = MixHash(h_pipeline, pp.orbit_index);
                    h_pipeline = MixHash(h_pipeline, pp.orbit_around);

                    if (pp.is_gas_final)
                    {
                        h_vein = MixHash(h_vein, 0);
                        continue;
                    }

                    int vmax = vein_len < 32 ? vein_len : 32;
                    for (int vid = 1; vid < vmax; ++vid)
                    {
                        int vv = vein_counts[static_cast<size_t>(solid_cursor) * static_cast<size_t>(vein_len) + static_cast<size_t>(vid)];
                        h_vein = MixHash(h_vein, vv);
                        h_pipeline = MixHash(h_pipeline, vv);
                    }
                    ++solid_cursor;
                }
            }

            out_galaxy_sigs[group_base + gi] = h_galaxy;
            out_planet_sigs[group_base + gi] = h_planet;
            out_vein_sigs[group_base + gi] = h_vein;
            out_pipeline_sigs[group_base + gi] = h_pipeline;
            if (debug_theme_compare && gi == 0)
            {
                std::fprintf(stderr,
                    "[native-sig-theme-debug] groupBase=%d hostSig(seed0): g/p/v/pi=0x%016llX/0x%016llX/0x%016llX/0x%016llX\n",
                    group_base,
                    static_cast<unsigned long long>(h_galaxy),
                    static_cast<unsigned long long>(h_planet),
                    static_cast<unsigned long long>(h_vein),
                    static_cast<unsigned long long>(h_pipeline));
                std::fflush(stderr);
            }

            if (debug_dump && (group_base + gi) == 0)
            {
                std::printf(
                    "[native-sig-debug] seed=%d stars=%d birthStar=%d birthPlanet=%d "
                    "gal=0x%016llX pl=0x%016llX ve=0x%016llX pipe=0x%016llX\n",
                    seed_ctx.galaxy_seed,
                    seed_ctx.star_count,
                    seed_ctx.birth_star_id,
                    seed_ctx.birth_planet_id,
                    static_cast<unsigned long long>(h_galaxy),
                    static_cast<unsigned long long>(h_planet),
                    static_cast<unsigned long long>(h_vein),
                    static_cast<unsigned long long>(h_pipeline));
                if (!seed_ctx.stars.empty())
                {
                    int sshow = std::min(4, static_cast<int>(seed_ctx.stars.size()));
                    for (int si = 0; si < sshow; ++si)
                    {
                        std::printf(
                            "[native-sig-debug] star%d id=%d seed=%d type=%d spectr=%d pc=%d posQ=(%d,%d,%d) coreInputs(mass=%.9g orbitScaler=%.9g hab=%.9g light=%.9g)\n",
                            si,
                            seed_ctx.stars[si].star.id,
                            seed_ctx.stars[si].star.seed,
                            seed_ctx.stars[si].star.type,
                            seed_ctx.stars[si].star.spectr,
                            seed_ctx.stars[si].star.planet_count,
                            seed_ctx.stars[si].star.pos_qx,
                            seed_ctx.stars[si].star.pos_qy,
                            seed_ctx.stars[si].star.pos_qz,
                            static_cast<double>(seed_ctx.stars[si].star.mass),
                            static_cast<double>(seed_ctx.stars[si].star.orbit_scaler),
                            static_cast<double>(seed_ctx.stars[si].star.habitable_radius),
                            static_cast<double>(seed_ctx.stars[si].star.light_balance_radius));
                        int pshow = std::min(2, static_cast<int>(seed_ctx.stars[si].planets.size()));
                        for (int pi = 0; pi < pshow; ++pi)
                        {
                            const PlanetPlanLite& p = seed_ctx.stars[si].planets[pi];
                            std::printf(
                                "[native-sig-debug] star%d.p%d id=%d infoSeed=%d genSeed=%d type=%d theme=%d water=%d orbit=(%d,%d) gas=%d core(orbit=%.9g scale=%.9g radius=%.9g temp=%.9g)\n",
                                si, pi, p.id, p.info_seed, p.gen_seed, p.type, p.theme, p.water_item_id, p.orbit_index, p.orbit_around, p.is_gas_final ? 1 : 0,
                                static_cast<double>(p.core.orbit_radius),
                                static_cast<double>(p.core.scale),
                                static_cast<double>(p.core.radius),
                                static_cast<double>(p.core.temperature_bias));
                        }
                    }
                    std::fflush(stdout);
                }
            }
        }
        if (stage_timing)
            stage_hash_ms += now_ms() - stage0;
    }

    if (stage_timing)
    {
        const double total_ms = now_ms() - stage_total_begin_ms;
        const double pose_known_ms = stage_pose_h2d_ms + stage_pose_gen_kernel_ms + stage_pose_d2h_counts_ms + stage_pose_gather_kernel_ms + stage_pose_d2h_head_ms;
        double pose_residual_ms = stage_pose_ms - pose_known_ms;
        if (pose_residual_ms < 0.0)
            pose_residual_ms = 0.0;
        const double pose_residual_pct = stage_pose_ms > 1e-9 ? (pose_residual_ms * 100.0 / stage_pose_ms) : 0.0;
        const double pose_residual_sync_wait_ms = stage_pose_d2h_head_sync_wait_ms;
        const double pose_residual_submit_ms = stage_pose_d2h_head_submit_ms;
        const double pose_residual_pre_ms = stage_pose_api_pre_ms;
        const double pose_residual_post_ms = stage_pose_api_post_ms;
        const double pose_residual_fast_buffer_ms = stage_pose_fast_buffer_ms;
        const double pose_residual_fast_buffer_release_ms = stage_pose_fast_buffer_release_ms;
        double pose_residual_fast_buffer_acquire_rest_ms =
            stage_pose_fast_buffer_ms
            - stage_pose_fast_buffer_acquire_lock_wait_ms
            - stage_pose_fast_buffer_acquire_lock_hold_ms
            - stage_pose_fast_buffer_acquire_set_device_ms
            - stage_pose_fast_buffer_acquire_free_ms
            - stage_pose_fast_buffer_acquire_malloc_ms;
        if (pose_residual_fast_buffer_acquire_rest_ms < 0.0)
            pose_residual_fast_buffer_acquire_rest_ms = 0.0;
        double pose_residual_fast_buffer_release_rest_ms =
            stage_pose_fast_buffer_release_ms
            - stage_pose_fast_buffer_release_lock_wait_ms
            - stage_pose_fast_buffer_release_lock_hold_ms;
        if (pose_residual_fast_buffer_release_rest_ms < 0.0)
            pose_residual_fast_buffer_release_rest_ms = 0.0;
        double pose_residual_accounted_ms =
            pose_residual_sync_wait_ms
            + pose_residual_submit_ms
            + pose_residual_pre_ms
            + pose_residual_post_ms
            + pose_residual_fast_buffer_ms;
        double pose_residual_other_ms = pose_residual_ms - pose_residual_accounted_ms;
        if (pose_residual_other_ms < 0.0)
            pose_residual_other_ms = 0.0;
        const double pose_residual_other_pct = pose_residual_ms > 1e-9 ? (pose_residual_other_ms * 100.0 / pose_residual_ms) : 0.0;
        double pose_api_residual_ms = stage_pose_api_total_host_ms - pose_known_ms;
        if (pose_api_residual_ms < 0.0)
            pose_api_residual_ms = 0.0;
        const double pose_api_residual_accounted_ms =
            pose_residual_pre_ms + pose_residual_submit_ms + pose_residual_sync_wait_ms + pose_residual_post_ms;
        double pose_residual_other_api_gap_candidate_ms = pose_api_residual_ms - pose_api_residual_accounted_ms;
        if (pose_residual_other_api_gap_candidate_ms < 0.0)
            pose_residual_other_api_gap_candidate_ms = 0.0;
        double pose_residual_other_outside_api_candidate_ms =
            stage_pose_ms - stage_pose_api_total_host_ms - pose_residual_fast_buffer_ms;
        if (pose_residual_other_outside_api_candidate_ms < 0.0)
            pose_residual_other_outside_api_candidate_ms = 0.0;
        double pose_residual_other_outside_api_release_candidate_ms = stage_pose_fast_buffer_release_ms;
        if (pose_residual_other_outside_api_release_candidate_ms < 0.0)
            pose_residual_other_outside_api_release_candidate_ms = 0.0;
        if (pose_residual_other_outside_api_release_candidate_ms > pose_residual_other_outside_api_candidate_ms)
            pose_residual_other_outside_api_release_candidate_ms = pose_residual_other_outside_api_candidate_ms;

        double pose_other_budget_ms = pose_residual_other_ms;
        double pose_residual_other_api_gap_ms = std::min(pose_other_budget_ms, pose_residual_other_api_gap_candidate_ms);
        pose_other_budget_ms -= pose_residual_other_api_gap_ms;
        double pose_residual_other_outside_api_ms = std::min(pose_other_budget_ms, pose_residual_other_outside_api_candidate_ms);
        pose_other_budget_ms -= pose_residual_other_outside_api_ms;
        double pose_residual_other_outside_api_release_ms =
            std::min(pose_residual_other_outside_api_ms, pose_residual_other_outside_api_release_candidate_ms);
        double pose_residual_other_outside_api_rest_ms =
            pose_residual_other_outside_api_ms - pose_residual_other_outside_api_release_ms;
        double pose_residual_other_unattributed_ms = pose_other_budget_ms;
        const double pose_residual_other_api_gap_pct =
            pose_residual_ms > 1e-9 ? (pose_residual_other_api_gap_ms * 100.0 / pose_residual_ms) : 0.0;
        const double pose_residual_other_outside_api_pct =
            pose_residual_ms > 1e-9 ? (pose_residual_other_outside_api_ms * 100.0 / pose_residual_ms) : 0.0;
        const double pose_residual_other_outside_api_release_pct =
            pose_residual_ms > 1e-9 ? (pose_residual_other_outside_api_release_ms * 100.0 / pose_residual_ms) : 0.0;
        const double pose_residual_other_outside_api_rest_pct =
            pose_residual_ms > 1e-9 ? (pose_residual_other_outside_api_rest_ms * 100.0 / pose_residual_ms) : 0.0;
        const double pose_residual_other_unattributed_pct =
            pose_residual_ms > 1e-9 ? (pose_residual_other_unattributed_ms * 100.0 / pose_residual_ms) : 0.0;
        const double pose_gen_seed_p50_ms =
            (stage_pose_gen_seed_profile_weight > 0.0)
                ? (stage_pose_gen_seed_p50_ms / stage_pose_gen_seed_profile_weight)
                : 0.0;
        const double pose_gen_phase1_avg_ms =
            (stage_pose_gen_seed_profile_weight > 0.0)
                ? (stage_pose_gen_phase1_ms / stage_pose_gen_seed_profile_weight)
                : 0.0;
        const double pose_gen_phase2_avg_ms =
            (stage_pose_gen_seed_profile_weight > 0.0)
                ? (stage_pose_gen_phase2_ms / stage_pose_gen_seed_profile_weight)
                : 0.0;
        const double pose_d2h_head_bw_gbps =
            (stage_pose_d2h_head_ms > 1e-9)
                ? (stage_pose_d2h_head_bytes_mb * 1024.0 * 1024.0) / (stage_pose_d2h_head_ms * 1.0e6)
                : 0.0;
        if (stage_timing_mode == kStageTimingModePose)
        {
            std::printf(
                "[native-sig-pose] seeds=%d stars=%d groupCap=%d totalMs=%.3f poseMs=%.3f "
                "poseH2D=%.3f poseGenK=%.3f poseD2HCounts=%.3f poseGatherK=%.3f poseD2HHead=%.3f "
                "poseKnown=%.3f poseResidual=%.3f poseResidualPct=%.3f "
                "poseResidualSyncWait=%.3f poseResidualApiPre=%.3f poseResidualApiPost=%.3f poseResidualFastBuffer=%.3f poseResidualOther=%.3f\n",
                seed_count, star_count, group_cap, total_ms, stage_pose_ms,
                stage_pose_h2d_ms, stage_pose_gen_kernel_ms, stage_pose_d2h_counts_ms, stage_pose_gather_kernel_ms, stage_pose_d2h_head_ms,
                pose_known_ms, pose_residual_ms, pose_residual_pct,
                pose_residual_sync_wait_ms, pose_residual_pre_ms, pose_residual_post_ms, pose_residual_fast_buffer_ms, pose_residual_other_ms);
        }
        else if (stage_timing_mode == kStageTimingModePoseGenK)
        {
            std::printf(
                "[native-sig-poseGenK] seeds=%d stars=%d groupCap=%d poseGenK=%.3f\n",
                seed_count, star_count, group_cap, stage_pose_gen_kernel_ms);
        }
        else if (stage_timing_mode == kStageTimingModeResidual)
        {
            std::printf(
                "[native-sig-residual] seeds=%d stars=%d groupCap=%d poseMs=%.3f poseKnown=%.3f poseResidual=%.3f poseResidualPct=%.3f "
                "syncWait=%.3f submit=%.3f apiPre=%.3f(setDevice=%.3f ensureBuffers=%.3f eventSetup=%.3f) "
                "apiPost=%.3f fastBuffer=%.3f(acqLockWait=%.3f acqLockHold=%.3f acqSetDevice=%.3f acqFree=%.3f acqMalloc=%.3f acqRest=%.3f rel=%.3f relLockWait=%.3f relLockHold=%.3f relRest=%.3f) apiTotalHost=%.3f "
                "other=%.3f(otherApiGap=%.3f otherOutsideApi=%.3f[release=%.3f rest=%.3f] otherUnattributed=%.3f) "
                "otherPct=%.3f(otherApiGapPct=%.3f otherOutsideApiPct=%.3f[releasePct=%.3f restPct=%.3f] otherUnattributedPct=%.3f)\n",
                seed_count, star_count, group_cap, stage_pose_ms, pose_known_ms, pose_residual_ms, pose_residual_pct,
                pose_residual_sync_wait_ms, pose_residual_submit_ms, pose_residual_pre_ms,
                stage_pose_set_device_ms, stage_pose_ensure_buffers_ms, stage_pose_event_setup_ms,
                pose_residual_post_ms, pose_residual_fast_buffer_ms,
                stage_pose_fast_buffer_acquire_lock_wait_ms, stage_pose_fast_buffer_acquire_lock_hold_ms,
                stage_pose_fast_buffer_acquire_set_device_ms, stage_pose_fast_buffer_acquire_free_ms, stage_pose_fast_buffer_acquire_malloc_ms, pose_residual_fast_buffer_acquire_rest_ms,
                pose_residual_fast_buffer_release_ms, stage_pose_fast_buffer_release_lock_wait_ms, stage_pose_fast_buffer_release_lock_hold_ms, pose_residual_fast_buffer_release_rest_ms,
                stage_pose_api_total_host_ms,
                pose_residual_other_ms, pose_residual_other_api_gap_ms, pose_residual_other_outside_api_ms,
                pose_residual_other_outside_api_release_ms, pose_residual_other_outside_api_rest_ms, pose_residual_other_unattributed_ms,
                pose_residual_other_pct, pose_residual_other_api_gap_pct, pose_residual_other_outside_api_pct,
                pose_residual_other_outside_api_release_pct, pose_residual_other_outside_api_rest_pct, pose_residual_other_unattributed_pct);
        }
        else
        {
            std::printf(
                "[native-sig-timing] seeds=%d stars=%d groupCap=%d totalMs=%.3f "
                "poseMs=%.3f seedBuildMs=%.3f corePackMs=%.3f coreKernelMs=%.3f coreUnpackMs=%.3f "
                "themeMs=%.3f veinPackMs=%.3f veinKernelMs=%.3f hashMs=%.3f "
                "poseH2D=%.3f poseGenK=%.3f poseD2HCounts=%.3f poseGatherK=%.3f poseD2HHead=%.3f "
                "poseGenPhase1Avg=%.3f poseGenPhase2Avg=%.3f poseGenSeedP50=%.3f poseGenSeedP95=%.3f poseGenSeedMax=%.3f "
                "poseGenAttempts=%.0f poseGenCollisionRejects=%.0f poseGenSphereRejects=%.0f poseGenGateSkips=%.0f "
                "poseD2HHeadSubmit=%.3f poseD2HHeadWait=%.3f poseD2HHeadMB=%.3f poseD2HHeadBW=%.3f "
                "seedBuildHostCtx=%.3f seedBuildHostMerge=%.3f seedBuildGpuCall=%.3f seedBuildFallbackHost=%.3f "
                "seedBuildH2D=%.3f seedBuildPlanK=%.3f seedBuildCoreK=%.3f seedBuildD2H=%.3f "
                "seedBuildHostPack=%.3f seedBuildAlloc=%.3f seedBuildScatter=%.3f\n",
                seed_count, star_count, group_cap, total_ms,
                stage_pose_ms, stage_seed_build_ms, stage_core_pack_ms, stage_core_kernel_ms, stage_core_unpack_ms,
                stage_theme_ms, stage_vein_pack_ms, stage_vein_kernel_ms, stage_hash_ms,
                stage_pose_h2d_ms, stage_pose_gen_kernel_ms, stage_pose_d2h_counts_ms, stage_pose_gather_kernel_ms, stage_pose_d2h_head_ms,
                pose_gen_phase1_avg_ms, pose_gen_phase2_avg_ms, pose_gen_seed_p50_ms, stage_pose_gen_seed_p95_ms, stage_pose_gen_seed_max_ms,
                stage_pose_gen_attempts_total, stage_pose_gen_collision_total, stage_pose_gen_sphere_total, stage_pose_gen_gate_total,
                stage_pose_d2h_head_submit_ms, stage_pose_d2h_head_sync_wait_ms, stage_pose_d2h_head_bytes_mb, pose_d2h_head_bw_gbps,
                stage_seed_build_host_ctx_ms, stage_seed_build_host_merge_ms, stage_seed_build_gpu_plan_call_ms, stage_seed_build_fallback_host_ms,
                stage_seed_build_h2d_ms, stage_seed_build_plan_kernel_ms, stage_seed_build_core_kernel_ms, stage_seed_build_d2h_ms,
                stage_seed_build_host_pack_ms, stage_seed_build_alloc_ms, stage_seed_build_scatter_ms);
        }
        std::fflush(stdout);
    }

    return DSP_CUDA_OK;
}
