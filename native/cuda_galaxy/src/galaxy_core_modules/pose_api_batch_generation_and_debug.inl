extern "C" int dsp_cuda_generate_temp_poses_params_fp64_batch(
    const int* seeds,
    int seed_count,
    int max_count,
    double min_dist,
    double min_step_len,
    double max_step_len,
    double flatten,
    int collision_fp64,
    int device_id,
    dsp_vec3d_t* out_poses,
    int out_stride,
    int* out_counts)
{
    bool debug_enter = false;
    if (const char* de = std::getenv("DSP_NATIVE_SIG_DEBUG_ENTER"))
        debug_enter = std::atoi(de) != 0;
    if (seeds == nullptr || out_poses == nullptr || out_counts == nullptr)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    if (seed_count <= 0 || max_count <= 0 || out_stride < max_count)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;

    int resolved_device_id = device_id;
    cudaError_t set_device_rc = EnsurePoseApiDevice(device_id, &resolved_device_id);
    if (set_device_rc != cudaSuccess)
    {
        if (debug_enter)
            std::fprintf(stderr, "[cuda-galaxy-error] pose_batch cudaSetDevice rc=%d %s\n", static_cast<int>(set_device_rc), cudaGetErrorString(set_device_rc));
        return DSP_CUDA_ERR_CUDA;
    }

    const size_t seeds_bytes = static_cast<size_t>(seed_count) * sizeof(int);
    const size_t counts_bytes = static_cast<size_t>(seed_count) * sizeof(int);
    const size_t poses_bytes = static_cast<size_t>(seed_count) * static_cast<size_t>(out_stride) * sizeof(dsp_vec3d_t);

    int* d_seeds = nullptr;
    int* d_seed_cursor = nullptr;
    int* d_counts = nullptr;
    dsp_vec3d_t* d_poses = nullptr;
    dsp_vec3d_t* d_drunk = nullptr;
    cudaStream_t stream = nullptr;

    cudaError_t rc = EnsureBatchBuffers(
        device_id,
        seeds_bytes,
        counts_bytes,
        poses_bytes,
        0,
        0,
        &d_seeds,
        &d_counts,
        &d_poses,
        &d_drunk,
        nullptr,
        nullptr,
        &d_seed_cursor,
        &stream);
    if (rc != cudaSuccess)
    {
        if (debug_enter)
            std::fprintf(stderr, "[cuda-galaxy-error] pose_batch EnsureBatchBuffers rc=%d %s\n", static_cast<int>(rc), cudaGetErrorString(rc));
        return DSP_CUDA_ERR_CUDA;
    }

    rc = cudaMemcpyAsync(d_seeds, seeds, seeds_bytes, cudaMemcpyHostToDevice, stream);
    if (rc != cudaSuccess)
    {
        if (debug_enter)
            std::fprintf(stderr, "[cuda-galaxy-error] pose_batch H2D seeds rc=%d %s\n", static_cast<int>(rc), cudaGetErrorString(rc));
        return DSP_CUDA_ERR_CUDA;
    }

    const int use_spatial_hash = UsePoseSpatialHash() ? 1 : 0;
    const int use_fast_math = UsePoseFastMath() ? 1 : 0;
    const int use_two_stage = UsePoseTwoStagePipeline() ? 1 : 0;
    const int candidate_batch = ResolvePoseCandidateBatch();
    const int block_size = 128;
    if (ShouldUsePosePersistentForSeedCount(seed_count))
    {
        rc = cudaMemsetAsync(d_seed_cursor, 0, sizeof(int), stream);
        if (rc == cudaSuccess)
        {
            const int grid_size = ResolvePosePersistentBlocks(seed_count, resolved_device_id);
            GenerateTempPosesParamsFp64BatchKernelPersistent<<<grid_size, block_size, 0, stream>>>(
                d_seeds,
                seed_count,
                max_count,
                min_dist,
                min_step_len,
                max_step_len,
                flatten,
                collision_fp64,
                d_poses,
                d_drunk,
                out_stride,
                d_counts,
                nullptr,
                d_seed_cursor,
                use_spatial_hash,
                use_fast_math,
                use_two_stage,
                candidate_batch);
        }
    }
    else
    {
        const int grid_size = (seed_count + block_size - 1) / block_size;
        GenerateTempPosesParamsFp64BatchKernel<<<grid_size, block_size, 0, stream>>>(
            d_seeds,
            seed_count,
            max_count,
            min_dist,
            min_step_len,
            max_step_len,
            flatten,
            collision_fp64,
            d_poses,
            d_drunk,
            out_stride,
            d_counts,
            nullptr,
            use_spatial_hash,
            use_fast_math,
            use_two_stage,
            candidate_batch);
    }

    if (rc == cudaSuccess)
        rc = cudaGetLastError();
    if (rc == cudaSuccess)
        rc = cudaMemcpyAsync(out_counts, d_counts, counts_bytes, cudaMemcpyDeviceToHost, stream);
    if (rc == cudaSuccess)
        rc = cudaMemcpyAsync(out_poses, d_poses, poses_bytes, cudaMemcpyDeviceToHost, stream);
    if (rc == cudaSuccess)
        rc = cudaStreamSynchronize(stream);

    if (rc != cudaSuccess)
    {
        if (debug_enter)
            std::fprintf(stderr, "[cuda-galaxy-error] pose_batch post-kernel rc=%d %s\n", static_cast<int>(rc), cudaGetErrorString(rc));
        return DSP_CUDA_ERR_CUDA;
    }
    return DSP_CUDA_OK;
}

extern "C" int dsp_cuda_generate_temp_poses_params_fp64_batch_head(
    const int* seeds,
    int seed_count,
    int max_count,
    int head_count,
    int sample_step,
    double min_dist,
    double min_step_len,
    double max_step_len,
    double flatten,
    int collision_fp64,
    int device_id,
    dsp_vec3d_t* out_poses,
    int out_stride,
    int* out_counts)
{
    g_last_pose_batch_head_timing =
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    bool debug_enter = false;
    if (const char* de = std::getenv("DSP_NATIVE_SIG_DEBUG_ENTER"))
        debug_enter = std::atoi(de) != 0;
    if (seeds == nullptr || out_poses == nullptr || out_counts == nullptr)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    if (seed_count <= 0 || max_count <= 0 || head_count <= 0 || head_count > max_count || out_stride < head_count)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    if (sample_step <= 0)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    if ((head_count - 1) * sample_step >= max_count)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    bool op_timing = false;
    if (const char* st = std::getenv("DSP_NATIVE_SIG_STAGE_TIMING"))
        op_timing = std::atoi(st) != 0;
    if (const char* ot = std::getenv("DSP_NATIVE_OP_TIMING"))
        op_timing = op_timing || (std::atoi(ot) != 0);
    bool collect_profile_detail = false;
    if (const char* pd = std::getenv("DSP_NATIVE_SIG_POSE_PROFILE_DETAIL"))
        collect_profile_detail = std::atoi(pd) != 0;
    auto now_ms_host = []() -> double {
        using clock = std::chrono::steady_clock;
        return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
    };

    int resolved_device_id = device_id;
    cudaError_t set_device_rc = EnsurePoseApiDevice(device_id, &resolved_device_id);
    if (set_device_rc != cudaSuccess)
    {
        if (debug_enter)
            std::fprintf(stderr, "[cuda-galaxy-error] pose_batch_head cudaSetDevice rc=%d %s\n", static_cast<int>(set_device_rc), cudaGetErrorString(set_device_rc));
        return DSP_CUDA_ERR_CUDA;
    }

    const int device_stride = max_count;
    const size_t seeds_bytes = static_cast<size_t>(seed_count) * sizeof(int);
    const size_t counts_bytes = static_cast<size_t>(seed_count) * sizeof(int);
    const size_t poses_bytes = static_cast<size_t>(seed_count) * static_cast<size_t>(device_stride) * sizeof(dsp_vec3d_t);
    const size_t head_bytes = static_cast<size_t>(seed_count) * static_cast<size_t>(out_stride) * sizeof(dsp_vec3d_t);
    const size_t gen_profiles_bytes = collect_profile_detail
        ? static_cast<size_t>(seed_count) * sizeof(PoseGenSeedProfile)
        : 0;
    bool use_pinned_staging = false;
    if (const char* ps = std::getenv("DSP_CUDA_POSE_PINNED_STAGING"))
        use_pinned_staging = std::atoi(ps) != 0;

    int* d_seeds = nullptr;
    int* d_seed_cursor = nullptr;
    int* d_counts = nullptr;
    dsp_vec3d_t* d_poses = nullptr;
    dsp_vec3d_t* d_drunk = nullptr;
    dsp_vec3d_t* d_head_poses = nullptr;
    PoseGenSeedProfile* d_gen_profiles = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t ev0 = nullptr, ev1 = nullptr, ev2 = nullptr, ev3 = nullptr, ev4 = nullptr, ev5 = nullptr;
    auto destroy_events = [&]() {
        if (ev5 != nullptr) cudaEventDestroy(ev5);
        if (ev4 != nullptr) cudaEventDestroy(ev4);
        if (ev3 != nullptr) cudaEventDestroy(ev3);
        if (ev2 != nullptr) cudaEventDestroy(ev2);
        if (ev1 != nullptr) cudaEventDestroy(ev1);
        if (ev0 != nullptr) cudaEventDestroy(ev0);
    };

    cudaError_t rc = EnsureBatchBuffers(
        device_id,
        seeds_bytes,
        counts_bytes,
        poses_bytes,
        head_bytes,
        gen_profiles_bytes,
        &d_seeds,
        &d_counts,
        &d_poses,
        &d_drunk,
        &d_head_poses,
        &d_gen_profiles,
        &d_seed_cursor,
        &stream);
    if (rc != cudaSuccess)
    {
        if (debug_enter)
            std::fprintf(stderr, "[cuda-galaxy-error] pose_batch_head EnsureBatchBuffers rc=%d %s\n", static_cast<int>(rc), cudaGetErrorString(rc));
        return DSP_CUDA_ERR_CUDA;
    }
    int* h_counts_target = out_counts;
    dsp_vec3d_t* h_head_target = out_poses;
    if (use_pinned_staging)
    {
        BatchDeviceBuffers& host_buffers = g_batch_buffers;
        rc = EnsurePinnedHostBuffer(
            reinterpret_cast<void**>(&host_buffers.h_counts_pinned),
            &host_buffers.host_counts_capacity_bytes,
            counts_bytes);
        if (rc == cudaSuccess)
        {
            rc = EnsurePinnedHostBuffer(
                reinterpret_cast<void**>(&host_buffers.h_head_poses_pinned),
                &host_buffers.host_head_poses_capacity_bytes,
                head_bytes);
        }
        if (rc != cudaSuccess)
        {
            if (debug_enter)
                std::fprintf(stderr, "[cuda-galaxy-error] pose_batch_head EnsurePinnedHostBuffer rc=%d %s\n", static_cast<int>(rc), cudaGetErrorString(rc));
            return DSP_CUDA_ERR_CUDA;
        }
        h_counts_target = host_buffers.h_counts_pinned;
        h_head_target = host_buffers.h_head_poses_pinned;
    }
    if (op_timing)
    {
        rc = cudaEventCreate(&ev0);
        if (rc == cudaSuccess) rc = cudaEventCreate(&ev1);
        if (rc == cudaSuccess) rc = cudaEventCreate(&ev2);
        if (rc == cudaSuccess) rc = cudaEventCreate(&ev3);
        if (rc == cudaSuccess) rc = cudaEventCreate(&ev4);
        if (rc == cudaSuccess) rc = cudaEventCreate(&ev5);
        if (rc != cudaSuccess)
        {
            destroy_events();
            ev0 = ev1 = ev2 = ev3 = ev4 = ev5 = nullptr;
            op_timing = false;
        }
    }
    if (op_timing)
        cudaEventRecord(ev0, stream);

    rc = cudaMemcpyAsync(d_seeds, seeds, seeds_bytes, cudaMemcpyHostToDevice, stream);
    if (rc != cudaSuccess)
    {
        if (debug_enter)
            std::fprintf(stderr, "[cuda-galaxy-error] pose_batch_head H2D seeds rc=%d %s\n", static_cast<int>(rc), cudaGetErrorString(rc));
        return DSP_CUDA_ERR_CUDA;
    }
    if (op_timing)
        cudaEventRecord(ev1, stream);

    const int use_spatial_hash = UsePoseSpatialHash() ? 1 : 0;
    const int use_fast_math = UsePoseFastMath() ? 1 : 0;
    const int use_two_stage = UsePoseTwoStagePipeline() ? 1 : 0;
    const int candidate_batch = ResolvePoseCandidateBatch();
    const int block_size = 128;
    if (ShouldUsePosePersistentForSeedCount(seed_count))
    {
        rc = cudaMemsetAsync(d_seed_cursor, 0, sizeof(int), stream);
        if (rc == cudaSuccess)
        {
            const int grid_size = ResolvePosePersistentBlocks(seed_count, resolved_device_id);
            GenerateTempPosesParamsFp64BatchKernelPersistent<<<grid_size, block_size, 0, stream>>>(
                d_seeds,
                seed_count,
                max_count,
                min_dist,
                min_step_len,
                max_step_len,
                flatten,
                collision_fp64,
                d_poses,
                d_drunk,
                device_stride,
                d_counts,
                collect_profile_detail ? d_gen_profiles : nullptr,
                d_seed_cursor,
                use_spatial_hash,
                use_fast_math,
                use_two_stage,
                candidate_batch);
        }
    }
    else
    {
        int grid_size = (seed_count + block_size - 1) / block_size;
        GenerateTempPosesParamsFp64BatchKernel<<<grid_size, block_size, 0, stream>>>(
            d_seeds,
            seed_count,
            max_count,
            min_dist,
            min_step_len,
            max_step_len,
            flatten,
            collision_fp64,
            d_poses,
            d_drunk,
            device_stride,
            d_counts,
            collect_profile_detail ? d_gen_profiles : nullptr,
            use_spatial_hash,
            use_fast_math,
            use_two_stage,
            candidate_batch);
    }
    if (op_timing)
        cudaEventRecord(ev2, stream);

    if (rc == cudaSuccess)
        rc = cudaGetLastError();
    if (rc == cudaSuccess)
        rc = cudaMemcpyAsync(h_counts_target, d_counts, counts_bytes, cudaMemcpyDeviceToHost, stream);
    if (rc == cudaSuccess && op_timing)
        cudaEventRecord(ev3, stream);
    if (rc == cudaSuccess)
    {
        int block = 128;
        int grid = (seed_count * head_count + block - 1) / block;
        GatherTempPosesHeadKernel<<<grid, block, 0, stream>>>(
            d_poses,
            seed_count,
            device_stride,
            head_count,
            sample_step,
            d_head_poses,
            out_stride);
        if (op_timing)
            cudaEventRecord(ev4, stream);
        rc = cudaGetLastError();
        if (rc == cudaSuccess)
        {
            double t_submit_begin = op_timing ? now_ms_host() : 0.0;
            rc = cudaMemcpyAsync(h_head_target, d_head_poses, head_bytes, cudaMemcpyDeviceToHost, stream);
            if (rc == cudaSuccess && op_timing)
                g_last_pose_batch_head_timing.d2h_head_submit_ms += (now_ms_host() - t_submit_begin);
        }
        if (rc == cudaSuccess && op_timing)
            cudaEventRecord(ev5, stream);
        if (rc == cudaSuccess)
        {
            double t_sync_begin = op_timing ? now_ms_host() : 0.0;
            rc = cudaStreamSynchronize(stream);
            if (rc == cudaSuccess && op_timing)
                g_last_pose_batch_head_timing.d2h_head_sync_wait_ms += (now_ms_host() - t_sync_begin);
        }
        if (rc == cudaSuccess && use_pinned_staging)
        {
            std::memcpy(out_counts, h_counts_target, counts_bytes);
            std::memcpy(out_poses, h_head_target, head_bytes);
        }
    }
    if (rc == cudaSuccess && op_timing)
    {
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, ev0, ev1) == cudaSuccess) g_last_pose_batch_head_timing.h2d_seeds_ms = static_cast<double>(ms);
        if (cudaEventElapsedTime(&ms, ev1, ev2) == cudaSuccess) g_last_pose_batch_head_timing.gen_kernel_ms = static_cast<double>(ms);
        if (cudaEventElapsedTime(&ms, ev2, ev3) == cudaSuccess) g_last_pose_batch_head_timing.d2h_counts_ms = static_cast<double>(ms);
        if (cudaEventElapsedTime(&ms, ev3, ev4) == cudaSuccess) g_last_pose_batch_head_timing.gather_kernel_ms = static_cast<double>(ms);
        if (cudaEventElapsedTime(&ms, ev4, ev5) == cudaSuccess) g_last_pose_batch_head_timing.d2h_head_ms = static_cast<double>(ms);
        if (cudaEventElapsedTime(&ms, ev0, ev5) == cudaSuccess) g_last_pose_batch_head_timing.total_ms = static_cast<double>(ms);
        g_last_pose_batch_head_timing.d2h_head_bytes_mb = static_cast<double>(head_bytes) / (1024.0 * 1024.0);
        if (g_last_pose_batch_head_timing.d2h_head_ms > 1e-9)
            g_last_pose_batch_head_timing.d2h_head_bw_gbps =
                static_cast<double>(head_bytes) / (g_last_pose_batch_head_timing.d2h_head_ms * 1.0e6);
    }
    if (rc == cudaSuccess && op_timing && collect_profile_detail && d_gen_profiles != nullptr)
    {
        std::vector<PoseGenSeedProfile> host_profiles(seed_count);
        cudaError_t rc_prof = cudaMemcpy(
            host_profiles.data(),
            d_gen_profiles,
            gen_profiles_bytes,
            cudaMemcpyDeviceToHost);
        if (rc_prof == cudaSuccess)
        {
            int profile_device = device_id;
            if (profile_device < 0)
            {
                if (cudaGetDevice(&profile_device) != cudaSuccess)
                    profile_device = 0;
            }
            int clock_khz = 0;
            if (cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, profile_device) == cudaSuccess && clock_khz > 0)
            {
                std::vector<double> seed_ms;
                seed_ms.reserve(seed_count);
                double phase1_ms = 0.0;
                double phase2_ms = 0.0;
                double attempts_total = 0.0;
                double collision_total = 0.0;
                double sphere_total = 0.0;
                double gate_total = 0.0;
                for (int i = 0; i < seed_count; ++i)
                {
                    const PoseGenSeedProfile& p = host_profiles[i];
                    phase1_ms += static_cast<double>(p.phase1_cycles) / static_cast<double>(clock_khz);
                    phase2_ms += static_cast<double>(p.phase2_cycles) / static_cast<double>(clock_khz);
                    seed_ms.push_back(static_cast<double>(p.phase1_cycles + p.phase2_cycles) / static_cast<double>(clock_khz));
                    attempts_total += static_cast<double>(p.attempts);
                    collision_total += static_cast<double>(p.collision_rejects);
                    sphere_total += static_cast<double>(p.sphere_rejects);
                    gate_total += static_cast<double>(p.gate_skips);
                }
                std::sort(seed_ms.begin(), seed_ms.end());
                auto percentile = [&](double q) -> double {
                    if (seed_ms.empty())
                        return 0.0;
                    double pos = q * static_cast<double>(seed_ms.size() - 1);
                    size_t idx = static_cast<size_t>(pos);
                    size_t idx2 = std::min(idx + 1, seed_ms.size() - 1);
                    double frac = pos - static_cast<double>(idx);
                    return seed_ms[idx] * (1.0 - frac) + seed_ms[idx2] * frac;
                };
                g_last_pose_batch_head_timing.gen_phase1_ms = phase1_ms;
                g_last_pose_batch_head_timing.gen_phase2_ms = phase2_ms;
                g_last_pose_batch_head_timing.gen_seed_p50_ms = percentile(0.50);
                g_last_pose_batch_head_timing.gen_seed_p95_ms = percentile(0.95);
                g_last_pose_batch_head_timing.gen_seed_max_ms = seed_ms.empty() ? 0.0 : seed_ms.back();
                g_last_pose_batch_head_timing.gen_attempts_total = attempts_total;
                g_last_pose_batch_head_timing.gen_collision_rejects_total = collision_total;
                g_last_pose_batch_head_timing.gen_sphere_rejects_total = sphere_total;
                g_last_pose_batch_head_timing.gen_gate_skips_total = gate_total;
                if (seed_count > 0)
                {
                    g_last_pose_batch_head_timing.gen_phase1_ms /= static_cast<double>(seed_count);
                    g_last_pose_batch_head_timing.gen_phase2_ms /= static_cast<double>(seed_count);
                }
            }
        }
    }
    destroy_events();

    if (rc != cudaSuccess)
    {
        if (debug_enter)
            std::fprintf(stderr, "[cuda-galaxy-error] pose_batch_head post-kernel rc=%d %s\n", static_cast<int>(rc), cudaGetErrorString(rc));
        return DSP_CUDA_ERR_CUDA;
    }
    return DSP_CUDA_OK;
}

extern "C" int dsp_cuda_generate_temp_poses_params_fp64_batch_head_device(
    const int* seeds,
    int seed_count,
    int max_count,
    int head_count,
    int sample_step,
    double min_dist,
    double min_step_len,
    double max_step_len,
    double flatten,
    int collision_fp64,
    int device_id,
    dsp_vec3d_t* out_poses_device,
    int out_stride,
    int* out_counts)
{
    g_last_pose_batch_head_timing =
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    bool debug_enter = false;
    if (const char* de = std::getenv("DSP_NATIVE_SIG_DEBUG_ENTER"))
        debug_enter = std::atoi(de) != 0;
    if (seeds == nullptr || out_poses_device == nullptr || out_counts == nullptr)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    if (seed_count <= 0 || max_count <= 0 || head_count <= 0 || head_count > max_count || out_stride < head_count)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    if (sample_step <= 0)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    if ((head_count - 1) * sample_step >= max_count)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;

    bool op_timing = false;
    if (const char* st = std::getenv("DSP_NATIVE_SIG_STAGE_TIMING"))
        op_timing = std::atoi(st) != 0;
    if (const char* ot = std::getenv("DSP_NATIVE_OP_TIMING"))
        op_timing = op_timing || (std::atoi(ot) != 0);

    int resolved_device_id = device_id;
    cudaError_t set_device_rc = EnsurePoseApiDevice(device_id, &resolved_device_id);
    if (set_device_rc != cudaSuccess)
    {
        if (debug_enter)
            std::fprintf(stderr, "[cuda-galaxy-error] pose_batch_head_device cudaSetDevice rc=%d %s\n", static_cast<int>(set_device_rc), cudaGetErrorString(set_device_rc));
        return DSP_CUDA_ERR_CUDA;
    }

    const int device_stride = max_count;
    const size_t seeds_bytes = static_cast<size_t>(seed_count) * sizeof(int);
    const size_t counts_bytes = static_cast<size_t>(seed_count) * sizeof(int);
    const size_t poses_bytes = static_cast<size_t>(seed_count) * static_cast<size_t>(device_stride) * sizeof(dsp_vec3d_t);

    int* d_seeds = nullptr;
    int* d_seed_cursor = nullptr;
    int* d_counts = nullptr;
    dsp_vec3d_t* d_poses = nullptr;
    dsp_vec3d_t* d_drunk = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t ev0 = nullptr, ev1 = nullptr, ev2 = nullptr, ev3 = nullptr, ev4 = nullptr;
    auto destroy_events = [&]() {
        if (ev4 != nullptr) cudaEventDestroy(ev4);
        if (ev3 != nullptr) cudaEventDestroy(ev3);
        if (ev2 != nullptr) cudaEventDestroy(ev2);
        if (ev1 != nullptr) cudaEventDestroy(ev1);
        if (ev0 != nullptr) cudaEventDestroy(ev0);
    };

    cudaError_t rc = EnsureBatchBuffers(
        device_id,
        seeds_bytes,
        counts_bytes,
        poses_bytes,
        0,
        0,
        &d_seeds,
        &d_counts,
        &d_poses,
        &d_drunk,
        nullptr,
        nullptr,
        &d_seed_cursor,
        &stream);
    if (rc != cudaSuccess)
    {
        if (debug_enter)
            std::fprintf(stderr, "[cuda-galaxy-error] pose_batch_head_device EnsureBatchBuffers rc=%d %s\n", static_cast<int>(rc), cudaGetErrorString(rc));
        return DSP_CUDA_ERR_CUDA;
    }

    if (op_timing)
    {
        rc = cudaEventCreate(&ev0);
        if (rc == cudaSuccess) rc = cudaEventCreate(&ev1);
        if (rc == cudaSuccess) rc = cudaEventCreate(&ev2);
        if (rc == cudaSuccess) rc = cudaEventCreate(&ev3);
        if (rc == cudaSuccess) rc = cudaEventCreate(&ev4);
        if (rc != cudaSuccess)
        {
            destroy_events();
            ev0 = ev1 = ev2 = ev3 = ev4 = nullptr;
            op_timing = false;
        }
    }
    if (op_timing)
        cudaEventRecord(ev0, stream);

    rc = cudaMemcpyAsync(d_seeds, seeds, seeds_bytes, cudaMemcpyHostToDevice, stream);
    if (rc != cudaSuccess)
    {
        if (debug_enter)
            std::fprintf(stderr, "[cuda-galaxy-error] pose_batch_head_device H2D seeds rc=%d %s\n", static_cast<int>(rc), cudaGetErrorString(rc));
        destroy_events();
        return DSP_CUDA_ERR_CUDA;
    }
    if (op_timing)
        cudaEventRecord(ev1, stream);

    const int use_spatial_hash = UsePoseSpatialHash() ? 1 : 0;
    const int use_fast_math = UsePoseFastMath() ? 1 : 0;
    const int use_two_stage = UsePoseTwoStagePipeline() ? 1 : 0;
    const int candidate_batch = ResolvePoseCandidateBatch();
    const int block_size = 128;
    if (ShouldUsePosePersistentForSeedCount(seed_count))
    {
        rc = cudaMemsetAsync(d_seed_cursor, 0, sizeof(int), stream);
        if (rc == cudaSuccess)
        {
            const int grid_size = ResolvePosePersistentBlocks(seed_count, resolved_device_id);
            GenerateTempPosesParamsFp64BatchKernelPersistent<<<grid_size, block_size, 0, stream>>>(
                d_seeds,
                seed_count,
                max_count,
                min_dist,
                min_step_len,
                max_step_len,
                flatten,
                collision_fp64,
                d_poses,
                d_drunk,
                device_stride,
                d_counts,
                nullptr,
                d_seed_cursor,
                use_spatial_hash,
                use_fast_math,
                use_two_stage,
                candidate_batch);
        }
    }
    else
    {
        int grid_size = (seed_count + block_size - 1) / block_size;
        GenerateTempPosesParamsFp64BatchKernel<<<grid_size, block_size, 0, stream>>>(
            d_seeds,
            seed_count,
            max_count,
            min_dist,
            min_step_len,
            max_step_len,
            flatten,
            collision_fp64,
            d_poses,
            d_drunk,
            device_stride,
            d_counts,
            nullptr,
            use_spatial_hash,
            use_fast_math,
            use_two_stage,
            candidate_batch);
    }
    if (op_timing)
        cudaEventRecord(ev2, stream);

    if (rc == cudaSuccess)
        rc = cudaGetLastError();
    if (rc == cudaSuccess)
        rc = cudaMemcpyAsync(out_counts, d_counts, counts_bytes, cudaMemcpyDeviceToHost, stream);
    if (rc == cudaSuccess && op_timing)
        cudaEventRecord(ev3, stream);
    if (rc == cudaSuccess)
    {
        int block = 128;
        int grid = (seed_count * head_count + block - 1) / block;
        GatherTempPosesHeadKernel<<<grid, block, 0, stream>>>(
            d_poses,
            seed_count,
            device_stride,
            head_count,
            sample_step,
            out_poses_device,
            out_stride);
        rc = cudaGetLastError();
    }
    if (rc == cudaSuccess && op_timing)
        cudaEventRecord(ev4, stream);
    if (rc == cudaSuccess)
        rc = cudaStreamSynchronize(stream);

    if (rc == cudaSuccess && op_timing)
    {
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, ev0, ev1) == cudaSuccess) g_last_pose_batch_head_timing.h2d_seeds_ms = static_cast<double>(ms);
        if (cudaEventElapsedTime(&ms, ev1, ev2) == cudaSuccess) g_last_pose_batch_head_timing.gen_kernel_ms = static_cast<double>(ms);
        if (cudaEventElapsedTime(&ms, ev2, ev3) == cudaSuccess) g_last_pose_batch_head_timing.d2h_counts_ms = static_cast<double>(ms);
        if (cudaEventElapsedTime(&ms, ev3, ev4) == cudaSuccess) g_last_pose_batch_head_timing.gather_kernel_ms = static_cast<double>(ms);
        g_last_pose_batch_head_timing.d2h_head_ms = 0.0;
        g_last_pose_batch_head_timing.d2h_head_bytes_mb = 0.0;
        g_last_pose_batch_head_timing.d2h_head_bw_gbps = 0.0;
        if (cudaEventElapsedTime(&ms, ev0, ev4) == cudaSuccess) g_last_pose_batch_head_timing.total_ms = static_cast<double>(ms);
    }
    destroy_events();

    if (rc != cudaSuccess)
    {
        if (debug_enter)
            std::fprintf(stderr, "[cuda-galaxy-error] pose_batch_head_device post-kernel rc=%d %s\n", static_cast<int>(rc), cudaGetErrorString(rc));
        return DSP_CUDA_ERR_CUDA;
    }
    return DSP_CUDA_OK;
}

extern "C" int dsp_cuda_get_last_pose_batch_head_timing(
    dsp_cuda_pose_batch_head_timing_t* out_timing)
{
    if (out_timing == nullptr)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    *out_timing = g_last_pose_batch_head_timing;
    return DSP_CUDA_OK;
}

extern "C" int dsp_cuda_generate_temp_poses_params_fp64(
    int seed,
    int max_count,
    double min_dist,
    double min_step_len,
    double max_step_len,
    double flatten,
    int collision_fp64,
    int device_id,
    dsp_vec3d_t* out_poses,
    int out_capacity,
    int* out_count)
{
    if (out_capacity < max_count || out_count == nullptr)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;

    int seed_value = seed;
    int count_value = 0;
    int rc = dsp_cuda_generate_temp_poses_params_fp64_batch(
        &seed_value,
        1,
        max_count,
        min_dist,
        min_step_len,
        max_step_len,
        flatten,
        collision_fp64,
        device_id,
        out_poses,
        out_capacity,
        &count_value);
    *out_count = count_value;
    return rc;
}

extern "C" int dsp_cuda_debug_rng_nextdouble(
    int seed,
    int count,
    int device_id,
    double* out_values)
{
    if (count <= 0 || out_values == nullptr)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;

    if (device_id >= 0)
    {
        cudaError_t set_device_rc = cudaSetDevice(device_id);
        if (set_device_rc != cudaSuccess)
            return DSP_CUDA_ERR_CUDA;
    }

    double* d_values = nullptr;
    size_t values_bytes = static_cast<size_t>(count) * sizeof(double);
    cudaError_t rc = cudaMalloc(reinterpret_cast<void**>(&d_values), values_bytes);
    if (rc != cudaSuccess)
        return DSP_CUDA_ERR_CUDA;

    DebugRngNextDoubleKernel<<<1, 1>>>(seed, count, d_values);
    rc = cudaGetLastError();
    if (rc == cudaSuccess)
        rc = cudaDeviceSynchronize();
    if (rc == cudaSuccess)
        rc = cudaMemcpy(out_values, d_values, values_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_values);
    if (rc != cudaSuccess)
        return DSP_CUDA_ERR_CUDA;
    return DSP_CUDA_OK;
}

extern "C" int dsp_cuda_debug_rng_state_after_ctor(
    int seed,
    int device_id,
    int* out_seed_array_56,
    int* out_inext,
    int* out_inextp)
{
    if (out_seed_array_56 == nullptr || out_inext == nullptr || out_inextp == nullptr)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;

    if (device_id >= 0)
    {
        cudaError_t set_device_rc = cudaSetDevice(device_id);
        if (set_device_rc != cudaSuccess)
            return DSP_CUDA_ERR_CUDA;
    }

    int* d_seed_array = nullptr;
    int* d_inext = nullptr;
    int* d_inextp = nullptr;
    cudaError_t rc = cudaMalloc(reinterpret_cast<void**>(&d_seed_array), sizeof(int) * 56);
    if (rc != cudaSuccess)
        return DSP_CUDA_ERR_CUDA;
    rc = cudaMalloc(reinterpret_cast<void**>(&d_inext), sizeof(int));
    if (rc != cudaSuccess)
    {
        cudaFree(d_seed_array);
        return DSP_CUDA_ERR_CUDA;
    }
    rc = cudaMalloc(reinterpret_cast<void**>(&d_inextp), sizeof(int));
    if (rc != cudaSuccess)
    {
        cudaFree(d_inext);
        cudaFree(d_seed_array);
        return DSP_CUDA_ERR_CUDA;
    }

    DebugRngStateAfterCtorKernel<<<1, 1>>>(seed, d_seed_array, d_inext, d_inextp);
    rc = cudaGetLastError();
    if (rc == cudaSuccess)
        rc = cudaDeviceSynchronize();
    if (rc == cudaSuccess)
        rc = cudaMemcpy(out_seed_array_56, d_seed_array, sizeof(int) * 56, cudaMemcpyDeviceToHost);
    if (rc == cudaSuccess)
        rc = cudaMemcpy(out_inext, d_inext, sizeof(int), cudaMemcpyDeviceToHost);
    if (rc == cudaSuccess)
        rc = cudaMemcpy(out_inextp, d_inextp, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_inextp);
    cudaFree(d_inext);
    cudaFree(d_seed_array);

    if (rc != cudaSuccess)
        return DSP_CUDA_ERR_CUDA;
    return DSP_CUDA_OK;
}
