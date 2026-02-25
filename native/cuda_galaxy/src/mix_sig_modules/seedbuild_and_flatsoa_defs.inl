struct SeedBuildCudaTiming
{
    double h2d_ms;
    double plan_kernel_ms;
    double core_kernel_ms;
    double d2h_ms;
    double host_pack_ms;
    double alloc_ms;
    double scatter_ms;
};

inline bool TryBuildStarPlanetPlansCuda(
    int device_id,
    const EnumMap& em,
    std::vector<SeedCtx>& seeds,
    int star_type_white_dwarf,
    int star_type_neutron_star,
    int star_type_black_hole,
    bool build_core_refs,
    bool need_info_seed,
    int scatter_threads,
    std::vector<PlanetRef>& primary_refs,
    std::vector<PlanetRef>& secondary_refs,
    std::vector<CoreLite>& out_core_flat,
    SeedBuildCudaTiming* timing_out,
    std::vector<int>* out_planet_counts_flat,
    std::vector<int>* out_orbit_arounds_flat,
    std::vector<int>* out_orbit_indexes_flat,
    std::vector<int>* out_gas_giants_flat,
    std::vector<int>* out_gen_seeds_flat,
    bool skip_scatter)
{
    if (timing_out != nullptr)
        *timing_out = SeedBuildCudaTiming{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    auto now_ms_local = []() -> double {
        using clock = std::chrono::steady_clock;
        return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
    };
    double t_host_pack_begin = timing_out != nullptr ? now_ms_local() : 0.0;
    size_t star_count_all = 0;
    for (const SeedCtx& seed_ctx : seeds)
        star_count_all += seed_ctx.stars.size();
    const int star_count = static_cast<int>(star_count_all);
    if (star_count <= 0)
    {
        primary_refs.clear();
        secondary_refs.clear();
        out_core_flat.clear();
        if (out_planet_counts_flat != nullptr) out_planet_counts_flat->clear();
        if (out_orbit_arounds_flat != nullptr) out_orbit_arounds_flat->clear();
        if (out_orbit_indexes_flat != nullptr) out_orbit_indexes_flat->clear();
        if (out_gas_giants_flat != nullptr) out_gas_giants_flat->clear();
        if (out_gen_seeds_flat != nullptr) out_gen_seeds_flat->clear();
        return true;
    }

    std::vector<int> map_seed_idx(star_count);
    std::vector<int> map_star_idx(star_count);
    std::vector<int> in_star_seeds(star_count);
    std::vector<int> in_star_types(star_count);
    std::vector<int> in_star_spectrs(star_count);
    std::vector<int> in_star_indexes(star_count);
    std::vector<int> in_star_counts_in_galaxy(star_count);
    std::vector<float> in_star_orbit_scalers(star_count);
    std::vector<double> in_star_masses(star_count);
    std::vector<float> in_star_habitable_radiuses(star_count);
    std::vector<float> in_star_light_balance_radiuses(star_count);
    int star_flat = 0;
    for (int seed_idx = 0; seed_idx < static_cast<int>(seeds.size()); ++seed_idx)
    {
        const SeedCtx& seed_ctx = seeds[seed_idx];
        for (int star_idx = 0; star_idx < static_cast<int>(seed_ctx.stars.size()); ++star_idx)
        {
            const StarCtx& sc = seed_ctx.stars[star_idx];
            map_seed_idx[star_flat] = seed_idx;
            map_star_idx[star_flat] = star_idx;
            in_star_seeds[star_flat] = sc.star.seed;
            in_star_types[star_flat] = sc.star.type;
            in_star_spectrs[star_flat] = sc.star.spectr;
            in_star_indexes[star_flat] = sc.star.index;
            in_star_counts_in_galaxy[star_flat] = seed_ctx.star_count;
            in_star_orbit_scalers[star_flat] = sc.star.orbit_scaler;
            in_star_masses[star_flat] = sc.star.mass;
            in_star_habitable_radiuses[star_flat] = sc.star.habitable_radius;
            in_star_light_balance_radiuses[star_flat] = sc.star.light_balance_radius;
            ++star_flat;
        }
    }
    if (timing_out != nullptr)
        timing_out->host_pack_ms += (now_ms_local() - t_host_pack_begin);

    const size_t n = static_cast<size_t>(star_count);
    const size_t plans_n = n * static_cast<size_t>(kStarPlanMaxPlanets);
    std::vector<int> out_planet_counts(star_count);
    std::vector<int> out_orbit_arounds(plans_n);
    std::vector<int> out_orbit_indexes(plans_n);
    std::vector<int> out_numbers(plans_n);
    std::vector<int> out_gas_giants(plans_n);
    std::vector<int> out_info_seeds;
    if (need_info_seed)
        out_info_seeds.resize(plans_n);
    std::vector<int> out_gen_seeds(plans_n);

    if (device_id >= 0)
    {
        cudaError_t rc_set = cudaSetDevice(device_id);
        if (rc_set != cudaSuccess)
            return false;
    }
    int* d_star_seeds = nullptr;
    int* d_star_types = nullptr;
    int* d_star_spectrs = nullptr;
    int* d_star_indexes = nullptr;
    int* d_out_planet_counts = nullptr;
    int* d_out_orbit_arounds = nullptr;
    int* d_out_orbit_indexes = nullptr;
    int* d_out_numbers = nullptr;
    int* d_out_gas_giants = nullptr;
    int* d_out_info_seeds = nullptr;
    int* d_out_gen_seeds = nullptr;
    int* d_star_counts_in_galaxy = nullptr;
    float* d_star_orbit_scalers = nullptr;
    double* d_star_masses = nullptr;
    float* d_star_habitable_radiuses = nullptr;
    float* d_star_light_balance_radiuses = nullptr;
    CoreLite* d_out_core = nullptr;
    cudaEvent_t ev_plan_begin = nullptr;
    cudaEvent_t ev_plan_end = nullptr;
    cudaEvent_t ev_core_end = nullptr;
    cudaStream_t stream = nullptr;

    auto cleanup = [&]() {
        if (stream != nullptr) cudaStreamDestroy(stream);
        if (ev_core_end != nullptr) cudaEventDestroy(ev_core_end);
        if (ev_plan_end != nullptr) cudaEventDestroy(ev_plan_end);
        if (ev_plan_begin != nullptr) cudaEventDestroy(ev_plan_begin);
        cudaFree(d_out_core);
        cudaFree(d_star_light_balance_radiuses);
        cudaFree(d_star_habitable_radiuses);
        cudaFree(d_star_masses);
        cudaFree(d_star_orbit_scalers);
        cudaFree(d_star_counts_in_galaxy);
        cudaFree(d_out_gen_seeds);
        cudaFree(d_out_info_seeds);
        cudaFree(d_out_gas_giants);
        cudaFree(d_out_numbers);
        cudaFree(d_out_orbit_indexes);
        cudaFree(d_out_orbit_arounds);
        cudaFree(d_out_planet_counts);
        cudaFree(d_star_indexes);
        cudaFree(d_star_spectrs);
        cudaFree(d_star_types);
        cudaFree(d_star_seeds);
    };
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess)
        return false;

    auto alloc = [&](int** p, size_t bytes) -> bool {
        return cudaMalloc(reinterpret_cast<void**>(p), bytes) == cudaSuccess;
    };
    auto allocf = [&](float** p, size_t bytes) -> bool {
        return cudaMalloc(reinterpret_cast<void**>(p), bytes) == cudaSuccess;
    };
    auto allocd = [&](double** p, size_t bytes) -> bool {
        return cudaMalloc(reinterpret_cast<void**>(p), bytes) == cudaSuccess;
    };
    auto alloc_core = [&](CoreLite** p, size_t bytes) -> bool {
        return cudaMalloc(reinterpret_cast<void**>(p), bytes) == cudaSuccess;
    };

    const size_t stars_bytes = n * sizeof(int);
    const size_t stars_f_bytes = n * sizeof(float);
    const size_t stars_d_bytes = n * sizeof(double);
    const size_t plans_bytes = plans_n * sizeof(int);
    const size_t core_bytes = plans_n * sizeof(CoreLite);
    double t_alloc_begin = timing_out != nullptr ? now_ms_local() : 0.0;
    if (!alloc(&d_star_seeds, stars_bytes) ||
        !alloc(&d_star_types, stars_bytes) ||
        !alloc(&d_star_spectrs, stars_bytes) ||
        !alloc(&d_star_indexes, stars_bytes) ||
        !alloc(&d_star_counts_in_galaxy, stars_bytes) ||
        !allocf(&d_star_orbit_scalers, stars_f_bytes) ||
        !allocd(&d_star_masses, stars_d_bytes) ||
        !allocf(&d_star_habitable_radiuses, stars_f_bytes) ||
        !allocf(&d_star_light_balance_radiuses, stars_f_bytes) ||
        !alloc(&d_out_planet_counts, stars_bytes) ||
        !alloc(&d_out_orbit_arounds, plans_bytes) ||
        !alloc(&d_out_orbit_indexes, plans_bytes) ||
        !alloc(&d_out_numbers, plans_bytes) ||
        !alloc(&d_out_gas_giants, plans_bytes) ||
        !alloc(&d_out_info_seeds, plans_bytes) ||
        !alloc(&d_out_gen_seeds, plans_bytes) ||
        !alloc_core(&d_out_core, core_bytes))
    {
        cleanup();
        return false;
    }
    if (timing_out != nullptr)
        timing_out->alloc_ms += (now_ms_local() - t_alloc_begin);

    auto h2d = [&](void* dst, const void* src, size_t bytes) -> bool {
        return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream) == cudaSuccess;
    };
    double t_h2d_begin = timing_out != nullptr ? now_ms_local() : 0.0;
    if (!h2d(d_star_seeds, in_star_seeds.data(), stars_bytes) ||
        !h2d(d_star_types, in_star_types.data(), stars_bytes) ||
        !h2d(d_star_spectrs, in_star_spectrs.data(), stars_bytes) ||
        !h2d(d_star_indexes, in_star_indexes.data(), stars_bytes) ||
        !h2d(d_star_counts_in_galaxy, in_star_counts_in_galaxy.data(), stars_bytes) ||
        cudaMemcpyAsync(d_star_orbit_scalers, in_star_orbit_scalers.data(), stars_f_bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(d_star_masses, in_star_masses.data(), stars_d_bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(d_star_habitable_radiuses, in_star_habitable_radiuses.data(), stars_f_bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(d_star_light_balance_radiuses, in_star_light_balance_radiuses.data(), stars_f_bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        cleanup();
        return false;
    }
    if (timing_out != nullptr)
        timing_out->h2d_ms += (now_ms_local() - t_h2d_begin);

    const int block = ResolveSigBlockSize();
    const int grid = (star_count + block - 1) / block;
    if (timing_out != nullptr)
    {
        if (cudaEventCreate(&ev_plan_begin) != cudaSuccess ||
            cudaEventCreate(&ev_plan_end) != cudaSuccess ||
            cudaEventCreate(&ev_core_end) != cudaSuccess)
        {
            cleanup();
            return false;
        }
        cudaEventRecord(ev_plan_begin, stream);
    }
    BuildStarPlanetPlansKernel<<<grid, block, 0, stream>>>(
        d_star_seeds,
        d_star_types,
        d_star_spectrs,
        d_star_indexes,
        star_count,
        em.star_type_main_seq,
        em.star_type_giant,
        em.star_type_white_dwarf,
        em.star_type_neutron_star,
        em.star_type_black_hole,
        em.spectr_m,
        em.spectr_k,
        em.spectr_g,
        em.spectr_f,
        em.spectr_a,
        em.spectr_b,
        em.spectr_o,
        d_out_planet_counts,
        d_out_orbit_arounds,
        d_out_orbit_indexes,
        d_out_numbers,
        d_out_gas_giants,
        d_out_info_seeds,
        d_out_gen_seeds);
    if (timing_out != nullptr)
        cudaEventRecord(ev_plan_end, stream);

    cudaError_t rc = cudaGetLastError();
    if (rc != cudaSuccess)
    {
        cleanup();
        return false;
    }

    EvalPlanetCoreFromPlansKernel<<<grid, block, 0, stream>>>(
        d_star_types,
        d_star_indexes,
        d_star_counts_in_galaxy,
        d_star_orbit_scalers,
        d_star_masses,
        d_star_habitable_radiuses,
        d_star_light_balance_radiuses,
        d_out_planet_counts,
        d_out_orbit_arounds,
        d_out_orbit_indexes,
        d_out_numbers,
        d_out_gas_giants,
        d_out_info_seeds,
        star_count,
        star_type_white_dwarf,
        star_type_neutron_star,
        star_type_black_hole,
        d_out_core);
    if (timing_out != nullptr)
        cudaEventRecord(ev_core_end, stream);

    rc = cudaGetLastError();
    if (rc != cudaSuccess)
    {
        cleanup();
        return false;
    }

    auto d2h = [&](void* dst, const void* src, size_t bytes) -> bool {
        return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream) == cudaSuccess;
    };
    double t_d2h_begin = timing_out != nullptr ? now_ms_local() : 0.0;
    if (!d2h(out_planet_counts.data(), d_out_planet_counts, stars_bytes) ||
        !d2h(out_orbit_arounds.data(), d_out_orbit_arounds, plans_bytes) ||
        !d2h(out_orbit_indexes.data(), d_out_orbit_indexes, plans_bytes) ||
        !d2h(out_numbers.data(), d_out_numbers, plans_bytes) ||
        !d2h(out_gas_giants.data(), d_out_gas_giants, plans_bytes) ||
        !d2h(out_gen_seeds.data(), d_out_gen_seeds, plans_bytes))
    {
        cleanup();
        return false;
    }
    if (need_info_seed && !d2h(out_info_seeds.data(), d_out_info_seeds, plans_bytes))
    {
        cleanup();
        return false;
    }
    out_core_flat.resize(plans_n);
    if (!d2h(out_core_flat.data(), d_out_core, core_bytes))
    {
        cleanup();
        return false;
    }
    rc = cudaStreamSynchronize(stream);
    if (rc != cudaSuccess)
    {
        cleanup();
        return false;
    }
    if (skip_scatter)
    {
        if (out_planet_counts_flat != nullptr)
            *out_planet_counts_flat = out_planet_counts;
        if (out_orbit_arounds_flat != nullptr)
            *out_orbit_arounds_flat = out_orbit_arounds;
        if (out_orbit_indexes_flat != nullptr)
            *out_orbit_indexes_flat = out_orbit_indexes;
        if (out_gas_giants_flat != nullptr)
            *out_gas_giants_flat = out_gas_giants;
        if (out_gen_seeds_flat != nullptr)
            *out_gen_seeds_flat = out_gen_seeds;
    }
    if (timing_out != nullptr)
    {
        timing_out->d2h_ms += (now_ms_local() - t_d2h_begin);
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, ev_plan_begin, ev_plan_end) == cudaSuccess)
            timing_out->plan_kernel_ms += static_cast<double>(ms);
        if (cudaEventElapsedTime(&ms, ev_plan_end, ev_core_end) == cudaSuccess)
            timing_out->core_kernel_ms += static_cast<double>(ms);
    }
    cleanup();

    if (skip_scatter)
    {
        primary_refs.clear();
        secondary_refs.clear();
        return true;
    }

    primary_refs.clear();
    secondary_refs.clear();
    if (build_core_refs)
    {
        primary_refs.reserve(static_cast<size_t>(star_count) * 3);
        secondary_refs.reserve(static_cast<size_t>(star_count) * 2);
    }
    auto scatter_one = [&](int i) -> bool {
        int seed_idx = map_seed_idx[i];
        int star_idx = map_star_idx[i];
        StarCtx& star_ctx = seeds[seed_idx].stars[star_idx];
        const int pc = out_planet_counts[i];
        if (pc < 0 || pc > kStarPlanMaxPlanets)
            return false;

        star_ctx.planets.resize(pc);
        const int base = i * kStarPlanMaxPlanets;
        int primary_number_to_index[kStarPlanMaxPlanets + 1];
        for (int t = 0; t <= kStarPlanMaxPlanets; ++t)
            primary_number_to_index[t] = -1;
        for (int pi = 0; pi < pc; ++pi)
        {
            PlanetPlanLite& p = star_ctx.planets[pi];
            const CoreLite& core = out_core_flat[base + pi];
            p.index = pi;
            p.orbit_around = out_orbit_arounds[base + pi];
            p.orbit_index = out_orbit_indexes[base + pi];
            p.number = out_numbers[base + pi];
            p.gas_giant = out_gas_giants[base + pi] != 0;
            p.info_seed = need_info_seed ? out_info_seeds[base + pi] : 0;
            p.gen_seed = out_gen_seeds[base + pi];
            p.id = star_ctx.star.id * 100 + pi + 1;
            p.core = core;
            p.parent_planet_index = -1;
            if (p.orbit_around == 0 && p.number >= 0 && p.number <= kStarPlanMaxPlanets)
                primary_number_to_index[p.number] = pi;
        }
        for (int pi = 0; pi < pc; ++pi)
        {
            PlanetPlanLite& p = star_ctx.planets[pi];
            if (p.orbit_around > 0 && p.orbit_around <= kStarPlanMaxPlanets)
                p.parent_planet_index = primary_number_to_index[p.orbit_around];
        }
        star_ctx.star.planet_count = static_cast<int>(star_ctx.planets.size());
        return true;
    };

    double t_scatter_begin = timing_out != nullptr ? now_ms_local() : 0.0;
    if (!build_core_refs && scatter_threads > 1 && star_count >= 1024)
    {
        int tcount = std::min(scatter_threads, star_count);
        if (tcount < 1)
            tcount = 1;
        std::atomic<int> scatter_ok(1);
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(tcount - 1));
        const int chunk_size = (star_count + tcount - 1) / tcount;
        for (int w = 1; w < tcount; ++w)
        {
            int begin = w * chunk_size;
            int end = std::min(star_count, begin + chunk_size);
            if (begin >= end)
                break;
            workers.emplace_back([&, begin, end]() {
                for (int i = begin; i < end; ++i)
                {
                    if (scatter_ok.load(std::memory_order_relaxed) == 0)
                        break;
                    if (!scatter_one(i))
                    {
                        scatter_ok.store(0, std::memory_order_relaxed);
                        break;
                    }
                }
            });
        }
        int begin0 = 0;
        int end0 = std::min(star_count, chunk_size);
        for (int i = begin0; i < end0; ++i)
        {
            if (!scatter_one(i))
            {
                scatter_ok.store(0, std::memory_order_relaxed);
                break;
            }
        }
        for (auto& th : workers)
            th.join();
        if (scatter_ok.load(std::memory_order_relaxed) == 0)
            return false;
    }
    else
    {
        for (int i = 0; i < star_count; ++i)
        {
            if (!scatter_one(i))
                return false;

            if (!build_core_refs)
                continue;

            int seed_idx = map_seed_idx[i];
            int star_idx = map_star_idx[i];
            const StarCtx& star_ctx = seeds[seed_idx].stars[star_idx];
            for (int pi = 0; pi < static_cast<int>(star_ctx.planets.size()); ++pi)
            {
                PlanetRef pr{seed_idx, star_idx, pi};
                if (star_ctx.planets[pi].orbit_around == 0)
                    primary_refs.push_back(pr);
                else
                    secondary_refs.push_back(pr);
            }
        }
    }
    if (timing_out != nullptr)
        timing_out->scatter_ms += (now_ms_local() - t_scatter_begin);

    return true;
}

struct SigGpuFlatSoA
{
    std::vector<int> seed_star_offsets;
    std::vector<int> star_planet_offsets;

    std::vector<int> star_ids;
    std::vector<int> star_types;
    std::vector<int> star_spectrs;
    std::vector<int> star_planet_counts;
    std::vector<int> star_pos_qx;
    std::vector<int> star_pos_qy;
    std::vector<int> star_pos_qz;
    std::vector<int> star_indexes;
    std::vector<float> star_habitable_radiuses;

    std::vector<int> planet_ids;
    std::vector<int> planet_indexes;
    std::vector<int> planet_orbit_indexes;
    std::vector<int> planet_orbit_arounds;
    std::vector<int> planet_gen_seeds;
    std::vector<int> planet_gas_giants;
    std::vector<float> planet_core_habitable_bias;
    std::vector<float> planet_core_sun_distance;
    std::vector<float> planet_core_temperature_bias;
    std::vector<double> planet_core_num13;
    std::vector<double> planet_core_num14;
    std::vector<double> planet_core_rand1;
};

inline void BuildSigGpuFlatSoA(const std::vector<SeedCtx>& seeds, SigGpuFlatSoA& out)
{
    const int seed_count = static_cast<int>(seeds.size());
    out.seed_star_offsets.assign(seed_count + 1, 0);

    int total_stars = 0;
    int total_planets = 0;
    for (int si = 0; si < seed_count; ++si)
    {
        out.seed_star_offsets[si] = total_stars;
        const SeedCtx& seed_ctx = seeds[si];
        total_stars += static_cast<int>(seed_ctx.stars.size());
        for (const StarCtx& star_ctx : seed_ctx.stars)
            total_planets += static_cast<int>(star_ctx.planets.size());
    }
    out.seed_star_offsets[seed_count] = total_stars;

    out.star_planet_offsets.assign(total_stars + 1, 0);
    out.star_ids.resize(total_stars);
    out.star_types.resize(total_stars);
    out.star_spectrs.resize(total_stars);
    out.star_planet_counts.resize(total_stars);
    out.star_pos_qx.resize(total_stars);
    out.star_pos_qy.resize(total_stars);
    out.star_pos_qz.resize(total_stars);
    out.star_indexes.resize(total_stars);
    out.star_habitable_radiuses.resize(total_stars);

    out.planet_ids.resize(total_planets);
    out.planet_indexes.resize(total_planets);
    out.planet_orbit_indexes.resize(total_planets);
    out.planet_orbit_arounds.resize(total_planets);
    out.planet_gen_seeds.resize(total_planets);
    out.planet_gas_giants.resize(total_planets);
    out.planet_core_habitable_bias.resize(total_planets);
    out.planet_core_sun_distance.resize(total_planets);
    out.planet_core_temperature_bias.resize(total_planets);
    out.planet_core_num13.resize(total_planets);
    out.planet_core_num14.resize(total_planets);
    out.planet_core_rand1.resize(total_planets);

    int star_cursor = 0;
    int planet_cursor = 0;
    for (const SeedCtx& seed_ctx : seeds)
    {
        for (const StarCtx& star_ctx : seed_ctx.stars)
        {
            const StarLite& st = star_ctx.star;
            out.star_ids[star_cursor] = st.id;
            out.star_types[star_cursor] = st.type;
            out.star_spectrs[star_cursor] = st.spectr;
            out.star_planet_counts[star_cursor] = st.planet_count;
            out.star_pos_qx[star_cursor] = st.pos_qx;
            out.star_pos_qy[star_cursor] = st.pos_qy;
            out.star_pos_qz[star_cursor] = st.pos_qz;
            out.star_indexes[star_cursor] = st.index;
            out.star_habitable_radiuses[star_cursor] = st.habitable_radius;
            out.star_planet_offsets[star_cursor] = planet_cursor;

            for (const PlanetPlanLite& pp : star_ctx.planets)
            {
                out.planet_ids[planet_cursor] = pp.id;
                out.planet_indexes[planet_cursor] = pp.index;
                out.planet_orbit_indexes[planet_cursor] = pp.orbit_index;
                out.planet_orbit_arounds[planet_cursor] = pp.orbit_around;
                out.planet_gen_seeds[planet_cursor] = pp.gen_seed;
                out.planet_gas_giants[planet_cursor] = pp.gas_giant ? 1 : 0;
                out.planet_core_habitable_bias[planet_cursor] = pp.core.habitable_bias;
                out.planet_core_sun_distance[planet_cursor] = pp.core.sun_distance;
                out.planet_core_temperature_bias[planet_cursor] = pp.core.temperature_bias;
                out.planet_core_num13[planet_cursor] = pp.core.num13;
                out.planet_core_num14[planet_cursor] = pp.core.num14;
                out.planet_core_rand1[planet_cursor] = pp.core.rand1;
                ++planet_cursor;
            }

            ++star_cursor;
        }
    }
    out.star_planet_offsets[total_stars] = total_planets;
}

inline bool BuildSigGpuFlatSoAFromPlan(
    const std::vector<SeedCtx>& seeds,
    const std::vector<int>& star_planet_counts,
    const std::vector<int>& plan_orbit_arounds,
    const std::vector<int>& plan_orbit_indexes,
    const std::vector<int>& plan_gas_giants,
    const std::vector<int>& plan_gen_seeds,
    const std::vector<CoreLite>& core_flat,
    SigGpuFlatSoA& out)
{
    const int seed_count = static_cast<int>(seeds.size());
    out.seed_star_offsets.assign(seed_count + 1, 0);
    int total_stars = 0;
    for (int si = 0; si < seed_count; ++si)
    {
        out.seed_star_offsets[si] = total_stars;
        total_stars += static_cast<int>(seeds[si].stars.size());
    }
    out.seed_star_offsets[seed_count] = total_stars;

    if (static_cast<int>(star_planet_counts.size()) != total_stars)
        return false;
    const size_t plans_n = static_cast<size_t>(total_stars) * static_cast<size_t>(kStarPlanMaxPlanets);
    if (plan_orbit_arounds.size() != plans_n ||
        plan_orbit_indexes.size() != plans_n ||
        plan_gas_giants.size() != plans_n ||
        plan_gen_seeds.size() != plans_n ||
        core_flat.size() != plans_n)
    {
        return false;
    }

    int total_planets = 0;
    for (int i = 0; i < total_stars; ++i)
    {
        int pc = star_planet_counts[i];
        if (pc < 0) pc = 0;
        if (pc > kStarPlanMaxPlanets) pc = kStarPlanMaxPlanets;
        total_planets += pc;
    }

    out.star_planet_offsets.assign(total_stars + 1, 0);
    out.star_ids.resize(total_stars);
    out.star_types.resize(total_stars);
    out.star_spectrs.resize(total_stars);
    out.star_planet_counts.resize(total_stars);
    out.star_pos_qx.resize(total_stars);
    out.star_pos_qy.resize(total_stars);
    out.star_pos_qz.resize(total_stars);
    out.star_indexes.resize(total_stars);
    out.star_habitable_radiuses.resize(total_stars);

    out.planet_ids.resize(total_planets);
    out.planet_indexes.resize(total_planets);
    out.planet_orbit_indexes.resize(total_planets);
    out.planet_orbit_arounds.resize(total_planets);
    out.planet_gen_seeds.resize(total_planets);
    out.planet_gas_giants.resize(total_planets);
    out.planet_core_habitable_bias.resize(total_planets);
    out.planet_core_sun_distance.resize(total_planets);
    out.planet_core_temperature_bias.resize(total_planets);
    out.planet_core_num13.resize(total_planets);
    out.planet_core_num14.resize(total_planets);
    out.planet_core_rand1.resize(total_planets);

    int star_cursor = 0;
    int planet_cursor = 0;
    for (const SeedCtx& seed_ctx : seeds)
    {
        for (const StarCtx& star_ctx : seed_ctx.stars)
        {
            const StarLite& st = star_ctx.star;
            out.star_ids[star_cursor] = st.id;
            out.star_types[star_cursor] = st.type;
            out.star_spectrs[star_cursor] = st.spectr;
            out.star_pos_qx[star_cursor] = st.pos_qx;
            out.star_pos_qy[star_cursor] = st.pos_qy;
            out.star_pos_qz[star_cursor] = st.pos_qz;
            out.star_indexes[star_cursor] = st.index;
            out.star_habitable_radiuses[star_cursor] = st.habitable_radius;
            out.star_planet_offsets[star_cursor] = planet_cursor;

            int pc = star_planet_counts[star_cursor];
            if (pc < 0) pc = 0;
            if (pc > kStarPlanMaxPlanets) pc = kStarPlanMaxPlanets;
            out.star_planet_counts[star_cursor] = pc;

            const int plan_base = star_cursor * kStarPlanMaxPlanets;
            for (int pi = 0; pi < pc; ++pi)
            {
                const int p = plan_base + pi;
                const CoreLite& core = core_flat[p];
                out.planet_ids[planet_cursor] = st.id * 100 + pi + 1;
                out.planet_indexes[planet_cursor] = pi;
                out.planet_orbit_indexes[planet_cursor] = plan_orbit_indexes[p];
                out.planet_orbit_arounds[planet_cursor] = plan_orbit_arounds[p];
                out.planet_gen_seeds[planet_cursor] = plan_gen_seeds[p];
                out.planet_gas_giants[planet_cursor] = plan_gas_giants[p];
                out.planet_core_habitable_bias[planet_cursor] = core.habitable_bias;
                out.planet_core_sun_distance[planet_cursor] = core.sun_distance;
                out.planet_core_temperature_bias[planet_cursor] = core.temperature_bias;
                out.planet_core_num13[planet_cursor] = core.num13;
                out.planet_core_num14[planet_cursor] = core.num14;
                out.planet_core_rand1[planet_cursor] = core.rand1;
                ++planet_cursor;
            }
            ++star_cursor;
        }
    }
    out.star_planet_offsets[total_stars] = planet_cursor;
    return true;
}

__device__ __forceinline__ unsigned long long MixHashDevice(unsigned long long h, int v)
{
    h ^= static_cast<unsigned int>(v);
    h *= kFnvPrime;
    return h;
}

__device__ __forceinline__ float DClamp01(float v)
{
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

__device__ __forceinline__ float DCalcPAndBonusCase(int star_type, int spectr, const EnumMap& m, int& bonus_case)
{
    bonus_case = 0;
    float p = 1.0f;
    if (star_type == m.star_type_main_seq)
    {
        if (spectr == m.spectr_m) p = 2.5f;
        else if (spectr == m.spectr_k) p = 1.0f;
        else if (spectr == m.spectr_g) p = 0.7f;
        else if (spectr == m.spectr_f) p = 0.6f;
        else if (spectr == m.spectr_a) p = 1.0f;
        else if (spectr == m.spectr_b) p = 0.4f;
        else if (spectr == m.spectr_o) p = 1.6f;
    }
    else if (star_type == m.star_type_giant)
    {
        p = 2.5f;
    }
    else if (star_type == m.star_type_white_dwarf)
    {
        p = 3.5f;
        bonus_case = 1;
    }
    else if (star_type == m.star_type_neutron_star)
    {
        p = 4.5f;
        bonus_case = 2;
    }
    else if (star_type == m.star_type_black_hole)
    {
        p = 5.0f;
        bonus_case = 2;
    }
    return p;
}

__device__ __forceinline__ bool ProbHitDevice(DotNet35RandomDeviceLite& rng, float threshold, bool use_fp32_prob_compare)
{
    double sample = rng.NextDouble();
    if (use_fp32_prob_compare)
    {
        float sample_f = static_cast<float>(sample);
        float threshold_f = static_cast<float>(threshold);
        return sample_f < threshold_f || (sample_f == threshold_f && sample < static_cast<double>(threshold_f));
    }
    return sample < static_cast<double>(threshold);
}

__device__ __forceinline__ int DMapTypeCaseToPlanetType(int type_case, const EnumMap& m)
{
    switch (type_case)
    {
    case 0: return m.planet_type_gas;
    case 1: return m.planet_type_ocean;
    case 2: return m.planet_type_vocano;
    case 4: return m.planet_type_ice;
    default: return m.planet_type_desert;
    }
}

