# native/cuda_galaxy 源码拆分报告（按完整语义边界）

## 本次目标
- 解决“`.inl` 被半途截断、阅读像半截代码”的问题。
- 重新按 **完整函数 / struct / kernel / extern API** 边界拆分。
- 保持行为不变（仅组织结构重构）。

## 关键原则
- 不再按固定行数硬切。
- 不在函数体中间断开文件。
- include 顺序保持与原始单文件一致，保证编译语义稳定。

## 当前文件结构

### `dsp_cuda_mix_seed_signatures.cu`
薄入口：`native/cuda_galaxy/src/dsp_cuda_mix_seed_signatures.cu`

包含模块：
- `native/cuda_galaxy/src/mix_sig_modules/common_theme_cache_workspace.inl`（413）
- `native/cuda_galaxy/src/mix_sig_modules/rng_star_models_and_host_math.inl`（869）
- `native/cuda_galaxy/src/mix_sig_modules/rng_star_device_and_build_kernels.inl`（548）
- `native/cuda_galaxy/src/mix_sig_modules/plan_core_planet_layout_kernels.inl`（995）
- `native/cuda_galaxy/src/mix_sig_modules/plan_core_finalize_and_compact_kernels.inl`（280）
- `native/cuda_galaxy/src/mix_sig_modules/seedbuild_and_flatsoa_defs.inl`（775）
- `native/cuda_galaxy/src/mix_sig_modules/flatsoa_build_and_theme_hash_kernel.inl`（662）
- `native/cuda_galaxy/src/mix_sig_modules/direct_pose_pipeline_direct_from_pose.inl`（1092）
- `native/cuda_galaxy/src/mix_sig_modules/direct_pose_pipeline_flat_eval.inl`（392）
- `native/cuda_galaxy/src/mix_sig_modules/mix_driver_entry.inl`（1531）

说明：
- `direct_pose_pipeline_direct_from_pose.inl` 与 `mix_driver_entry.inl` 仍超过 1000 行，
  但内部为单个完整大函数，未再做“函数中间切割”。
- 若要继续降行数，需要提取 helper 函数（会引入实质代码重构，不再是纯组织调整）。

### `dsp_cuda_galaxy.cu`
薄入口：`native/cuda_galaxy/src/dsp_cuda_galaxy.cu`

包含模块：
- `native/cuda_galaxy/src/galaxy_core_modules/pose_rng_collision_and_buffers.inl`（735）
- `native/cuda_galaxy/src/galaxy_core_modules/planet_core_vein_kernels_and_pose_kernels.inl`（741）
- `native/cuda_galaxy/src/galaxy_core_modules/pose_api_batch_generation_and_debug.inl`（729）
- `native/cuda_galaxy/src/galaxy_core_modules/planet_vein_api_and_core_batch.inl`（552）
- `native/cuda_galaxy/src/galaxy_core_modules/planet_gas_and_chunk_eval_api.inl`（404）

## 验证记录

### 编译
```bash
cmake --build native/cuda_galaxy/build -j
```
结果：通过。

### 运行回归（最小正确性）
```bash
mono SeedCli/bin/Release/net48/SeedCli.exe \
  --seed 100000 --stars 64 --count 2000 \
  --compare-pipeline-mix-veins-f32 \
  --mix-collision-fp64 \
  --use-cuda-galaxy --use-cuda-planet-core --use-cuda-planet \
  --cpu-threads 1 --gpu-streams 4 --gpu-chunk-seeds 2000 \
  --mix-core-group-seeds 2000 --show-mismatches 3
```
结果：`galaxyMismatch=0/2000`、`planetMismatch=0/2000`、`veinMismatch=0/2000`、`pipelineMismatch=0/2000`。
