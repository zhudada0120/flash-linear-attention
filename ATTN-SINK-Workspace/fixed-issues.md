# attn(sink)算子 NPU 适配 — 问题解决报告

> 环境:Ascend 910B4(20 AI Core / 40 Vector),triton-ascend 3.2.0 + CANN 9.0 + torch_npu
> 分支:`attn-sink-npu-adapt`(基于 origin/main @ `864a87f6`)
> 基线:tests/ops/test_attn_sink.py **3 failed / 12 passed**(529s,2026-09-18)
> 对照:KDA 适配实录见 `KDA-NPU-Workspace/fixed-issues.md`

---

## 结论速览

与 KDA 不同,**这个算子的 Triton kernel 层在 NPU 上本来就能跑**——12 个真正执行 Triton kernel 的测试(parallel fwd/bwd + decoding,含 varlen/window/sink/gate 全特性组合)全部原生通过,零 kernel 适配工作。仅有的 3 个失败是 **fp64 参考对照测试撞上 torch_npu 的平台能力缺口**(fp64 matmul 不支持),与算子本身无关。修复 = 按仓库惯例加平台 skipif。

**最终结果:12 passed / 3 skipped(平台能力声明)/ 0 failed。**

---

## 问题:3 个 fp64 参考对照测试在 NPU 上崩溃

### 现象

```
FAILED test_attn_sink_ref_matches_gpt_oss_eager[full]
FAILED test_attn_sink_ref_matches_gpt_oss_eager[swa]
FAILED test_attn_sink_empty_row_ref_matches_gpt_oss_eager

RuntimeError: bmm:... NPU function error: call aclnnBatchMatMul failed,
              error code is 161002        ← 纯 PyTorch 层,torch/functional.py:422
```

### 诊断过程

1. **报错位置定位**:traceback 停在 `torch.functional.bmm`——错误发生在**纯 PyTorch 代码**,Triton kernel 根本没被调用。三个失败测试的共同点:它们是"参考 vs 参考"的对照测试(`naive_parallel_attn` ↔ `_gpt_oss_eager_sink_reference`,验证两个 eager 实现互相一致,容差 1e-10),不测任何 Triton 路径。
2. **共同变量**:三个测试全部 `dtype = torch.float64`(test 文件 76/123 行)——刻意的精度选择(1e-10 容差需要 fp64)。
3. **最小复现**(决定性):

```python
a = torch.randn(2, 8, 96, 64, dtype=torch.float64, device='npu')
b = torch.randn(2, 8, 64, 96, dtype=torch.float64, device='npu')
torch.matmul(a, b.transpose(2, 3))
# → RuntimeError: matmul_implement_npu ... (aclnnBatchMatMul 参数错误)
```

fp64 的 2D/3D matmul/bmm 在 910B4 + CANN 9.0 上**一律不支持**。旁证:KDA 适配期间编译器日志就警告过 `Device do not support double dtype now, dtype cast replace with float`——Triton 侧编译器会自动降级,但 torch_npu 的 aclnn 层直接报错。

### 根因

**平台能力缺失**:昇腾 CANN 的 aclnnBatchMatMul 不支持 float64 输入。三个测试的 fp64 需求超出平台能力,属"无法运行的测试前置条件",不是算子缺陷,也不是可以在 fla/ 代码里修复的问题(参考实现在测试文件内,且 fp64 是容差语义的一部分,不能降精度)。

### 解决方案

按仓库既有惯例([test_attnres.py:141](../tests/ops/test_attnres.py#L141)、test_kda.py、test_gdn2.py 均有 `skipif` 平台先例)给两个测试函数加平台 skip:

```python
@pytest.mark.skipif(IS_NPU, reason='torch_npu matmul does not support float64')
```

**关于"测试是冻结契约"的纪律边界**:冻结契约禁止的是*放宽数值断言/删减形状来掩盖 kernel bug*;而 skipif 声明的是*"此平台不具备运行该测试的前置条件"*——性质是能力声明,不弱化任何在可运行平台上的验证强度(CUDA 上照常全量执行)。注释中明确写出了理由和错误码,便于未来 NPU 支持 fp64 时移除。

### 验证

- 3 个用例 → `SKIPPED (torch_npu matmul does not support float64)`,0.07s
- 全量回归:12 passed / 3 skipped / 0 failed(见文末)

---

## 顺带确认的事实(为什么这个算子不需要 kernel 适配)

| 检查项 | 结果 |
|--------|------|
| backend 现状 | `fla/ops/attn/backends/` 只注册了 TileLangBackend(common 共享);NPU 上 is_available=False → 全部落 mainline Triton |
| mainline fwd kernel(parallel_attn_fwd) | ✅ 通过全部测试。online-softmax + sink 分母合并,gpt-oss 语义 |
| mainline bwd kernel(dq / dkv) | ✅ 通过(梯度对比测试全绿) |
| decoding kernel(attn_decoding_one_step) | ✅ 通过(5 个 decoding 测试:basic/value-split/empty-row/with-g ×2) |
| varlen / sliding window / sink / gate | ✅ 各参数化组合全绿 |

**为什么 mainline 能直接跑?**(与 KDA 对比的有意思之处)
1. `check_shared_mem('hopper'/'ampere', device.index)` 在 NPU 上走 `except → False` → 落**最保守 tile 档**(BS≤32、BV≤64)——恰好避开了 UB 溢出;KDA 当初挂在 Hopper 级大 tile 上。
2. kernel 是 attention 形状(BS×BK 的 score 矩阵主导),不是 KDA 那种多 stage 大并集 task-loop,UB 天然小。
3. `num_warps=8/4/2` 直接传入——triton-ascend 3.2 接受该参数(不报错),wrapper 无需改。
4. 无 `tl.dot` 左操作数复用(lhs clobber)模式、无就地读改写、无 varlen int64 风险点(bos/eos 已按 int64 写)。

这也验证了适配指南(ascend-op-adaptation-guide.md)的一个判断:**"未适配 ≠ 不能跑",mainline Triton kernel 在 NPU 上的可行性取决于其写法是否恰好落在 NPU 的约束内**——attn_sink 落进去了(保守 tile + 标准 attention 模式),KDA 没落进去(大 tile + 多 stage)。

---

## 与 KDA 任务的对比总结

| | KDA(31 failed) | attn_sink(3 failed) |
|---|-----------------|---------------------|
| 失败性质 | kernel 层真实缺陷(UB 溢出 ×3 处、就地读写 ×2 处) | 平台能力缺失(fp64 matmul) |
| Triton kernel 是否需要改 | 是(5 个问题,3 个修复 commit) | **否,零改动** |
| 修复位置 | fla/ops/**/triton_ascend/*.py | tests/ops/test_attn_sink.py(skipif) |
| 修复量 | +69/-5 行代码 + 文档 | +5 行测试装饰器 |
| 遗留 | 性能未评估(后续 optimization loop) | 若未来需要 NPU 上跑参考对照,可考虑 fp64→fp32 重写参考(容差语义会变,需上游讨论) |

## 修改清单

| commit | 内容 |
|--------|------|
| (本分支首个) | tests/ops/test_attn_sink.py:+IS_NPU import、2 处 skipif 装饰器(3 个用例) |

---

*报告时间:2026-09-18;分支 `attn-sink-npu-adapt`;工作区 ATTN-SINK-Workspace/*
