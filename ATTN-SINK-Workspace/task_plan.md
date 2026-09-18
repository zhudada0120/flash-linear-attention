# attn(sink)算子 NPU 适配 — 任务计划与跟踪

## 目标

`tests/ops/test_attn_sink.py` 在 910B4(triton-ascend 3.2.0 / CANN 9.0)上修到全绿。
参照 KDA 适配流程(KDA-NPU-Workspace/fixed-issues.md)。

## 算子现状(origin/main @ 864a87f6)

- 算子位置:`fla/ops/attn/`(parallel.py + decoding.py)
- 语义:gpt-oss 风格 sink attention(softmax 分母加 sink_bias 项,不参与 value matmul)
  - `parallel_attn`:训练/预填充,online-softmax flash attention 变体,
    dispatch 挂在 `parallel_attn_fwd` / `parallel_attn_bwd` 上
  - `attn_decoding_one_step`:单步解码,无 dispatch,纯 mainline
- backend:仅注册 TileLangBackend(common 共享,NPU 不可用)→ **NPU 上全部走 mainline Triton**
- mainline 的 NPU 风险点(预判):
  - wrapper 传 `num_warps=8/4/2`(triton-ascend 3.2 接受,待证)
  - `check_shared_mem('hopper'|'ampere', device.index)` 在 NPU 的返回值决定 tile 档位
  - tile:BS≤64 × BK≤256 × BV≤256(Hopper 档)——UB 溢出风险
  - bwd 两个 kernel(dq / dkv)更大

## ✅ 最终结果(2026-09-18)

**12 passed / 3 skipped / 0 failed**(3 个 skip 为 fp64 平台能力声明;Triton kernel 层零改动全绿)

## 进度清单

```
- [x] Phase 0 摸底:3 failed / 12 passed;失败全为 fp64 参考对照测试
- [x] Phase 1 诊断:bmm 崩在 torch_npu aclnnBatchMatMul(161002),最小 repro 确认 fp64 不支持
- [x] Phase 2 修复:2 个测试函数加 skipif(IS_NPU)(repo 有 test_attnres.py:141 先例)
- [x] Phase 3 全量回归 12 passed / 3 skipped + 报告 fixed-issues.md
```

## 修改记录(四栏)

| # | 报错 | 改动 | 原因 | 测试结果 |
|---|------|------|------|---------|
| 1 | 3 个 fp64 测试:RuntimeError bmm aclnnBatchMatMul 161002 | test_attn_sink.py 两个函数加 `@pytest.mark.skipif(IS_NPU, reason='torch_npu matmul does not support float64')` + import IS_NPU | 平台能力缺失(fp64 matmul),非算子缺陷;skipif 是能力声明不是放宽断言,CUDA 上照跑 | 3 用例 SKIPPED;全量 12 passed / 3 skipped ✅ |
