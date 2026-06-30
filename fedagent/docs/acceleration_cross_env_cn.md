# 跨环境加速对比 —— WebShop vs ALFWorld

**这篇文档一句话回答:** *哪些加速选择能从 WebShop transfer 到 ALFWorld,哪些会翻转,以及预测这一切的那条单一原则。*

这是自包含的跨环境综述。各环境的细节在
[`acceleration_cn.md`](./acceleration_cn.md)(WebShop 杠杆 + 分析)、
[`acceleration_results_cn.md`](./acceleration_results_cn.md)(WebShop 实测数)、
[`alfworld_testing_cn.md`](./alfworld_testing_cn.md)(ALFWorld 策略 + §6 结果)。两者都是
Qwen2.5-1.5B-Instruct,4×H100,GRPO(G=8),paper 设置。

> ### ⚠️ 先读这个 —— 什么能比、什么不能比
> **绝对墙钟秒数在两个环境之间不可比。** 它们的 val 规模(WebShop eval-mode sweep n=500 vs ALFWorld n=48)、
> episode 长度(15 vs 50 轮)、每步 env 重量都不同。**比的是*排名*、*相对 %* 惩罚、*机制* —— 绝不要
> "ALFWorld 3509s vs WebShop 2493s"。** 凡是*指标*要紧的数(每步 vs 整跑墙钟),都就地标注。

---

## 1. 一张表总览

| 维度 | WebShop(15-turn) | ALFWorld(50-turn) | 跨环境裁定 |
|---|---|---|---|
| **Eval-mode —— 最快** | `parallel`(2493s, n=500) | **`worker`**(3509s, n=48) | **翻转** —— worker 反超 parallel |
| **Eval-mode —— 最慢** | `shared`(3316s, n=500) | **`inline`**(4738s, n=48) | **翻转** —— inline 最慢,不是 shared |
| **Eval-mode —— 结构** | parallel < worker < inline < shared | **worker < parallel < shared < inline** | 解耦胜耦合不变;内部次序移动 |
| **1-GPU 惩罚** | **+37%**(t1 墙钟,995/725) | **每步 +38%**(534/387);单步墙钟 +21% | **transfer**(每步基本一致) |
| **GPU↔rollout 耦合** | rollout 对 GPU 敏感¹ | **gen 在 1/2/4 GPU 下平坦(env-bound)** | ALFWorld 专属发现 |
| **2-job 并发(ZMQ 修复)** | PASS(3-job) | PASS(2-job,两个 rc=0) | **transfer** |
| **持久训练器(#4)** | −43%/轮,跨轮 −62% | 本轮未单独测²| 预测相对收益更小² |

¹ 推断(无 `_TW_LOCK`、env 更轻),本轮未对 WebShop 单独分解。
² ALFWorld 更大的 rollout 项压低了冷启动占墙钟的*比例*,所以 #4 的*相对*收益预测更小;尚未单独测。

**三句话:** 把 eval 从训练关键路径解耦在**两个**环境都赢,但 ALFWorld 上 `worker`(跨轮冷启动摊销)反超
`parallel`,且 `inline` 变成*最差* —— 排名翻转。每卡训练惩罚基本**相同**(~+38%),但*它没变更差的原因*不同:
ALFWorld 的 rollout 受**环境延迟约束**(生成时间随卡数平坦)。并发修复与环境无关,两边都成立。

---

## 2. Eval-mode 排名 —— 大翻转

同一个 4-mode sweep(inline / parallel / shared / worker),每种 = eval 跑在相对训练的不同位置。2 client × 2 round、
每轮 eval 的整跑墙钟:

| 名次 | WebShop(n=500) | ALFWorld(n=48) |
|---|---|---|
| 1(最快) | parallel 2493s | **worker 3509s** |
| 2 | worker 2637s | parallel 3620s |
| 3 | inline 3090s | shared 4560s |
| 4(最慢) | shared 3316s | **inline 4738s** |

**不变的:** 两个 **eval-解耦** 模式(`worker`、`parallel`)赢两个 **eval-耦合** 模式(`shared`、`inline`)。
eval 是否压在 4-GPU 训练关键路径上,在两个环境里都是主导因素。

**翻转的,以及为什么:**
- **`worker` 反超 `parallel`。** ALFWorld 的 eval 引擎冷启动(vLLM 初始化 + CUDA-graph capture + 加载 8810 局服务)
  *很贵*。`worker` 只付**一次**(跨轮常驻)并把 4 张卡全留给训练;`parallel` 藏了 eval 但只用 2 卡训练(+30%/步)。
  eval 越重,摊销冷启动 > 藏 eval。
- **`inline` 变最差(不是 `shared`)。** `inline` 每轮在关键路径上**重启**那个贵引擎。WebShop 上 eval 够轻,inline
  重启便宜,`shared` 的 0.3-util KV 限流才是最差;ALFWorld 上每轮重启重引擎成了主导,于是 `inline` 沉到被限流的
  `shared` 之下。

> **可比性注意。** WebShop "shared 最慢"专门是**大 val(n=500)**效应;ALFWorld 跑的是 n=48。所以 shared↔inline 的
> 次序部分是 val 规模、不是纯环境效应。与 val 规模无关的稳健论断是**机制**:ALFWorld 重量级的*每次 eval 冷启动*
> 让 `inline` 成为输家、并奖励 `worker` 的摊销。

---

## 3. GPU scaling —— transfer 的部分(机制更锐利)

**1-GPU 惩罚在两个环境上~相同。**

| | WebShop | ALFWorld |
|---|---|---|
| 4-GPU | 558s(t1 墙钟) | 298.4s/步 · 778s 墙钟 |
| 2-GPU | 725s(t1 墙钟) | 386.9s/步 · 865s 墙钟 |
| 1-GPU | 995s(t1 墙钟) | 534.5s/步 · 1050s 墙钟 |
| **1-GPU vs 2-GPU** | **+37%**(墙钟) | **每步 +38%**;单步墙钟 +21% |

干净的同口径数字是 **每步 ≈ ALFWorld +38% ≈ WebShop +37%** —— 惩罚在 ALFWorld 上并**没有**变窄。(ALFWorld 更低的
*墙钟*数 +21% 是单步探针伪影:~490s 固定开销 —— 服务加载 + Ray/vLLM 初始化 + 拆除 —— 不 scaling,稀释了单步。真实
多步 run 里墙钟惩罚会爬回每步的 +38%。)

**新机制(仅 ALFWorld,实测):** 把每步拆成 rollout vs 训练 ——

| GPU | gen(rollout) | update_actor(训练) |
|---|---|---|
| 1 | 228.3s | 140.0s |
| 2 | 225.3s | 92.2s |
| 4 | 219.3s | 43.3s |
| scaling | **平坦(−4%)** | **~线性(3.2×)** |

`gen` 随卡数**平坦** → ALFWorld rollout 受**环境延迟约束**:被 `_TW_LOCK` 串行化、`pool_size=8` 节流的 TextWorld
服务决定生成,而非 GPU 算力。只有 `update_actor` scaling。**实用杠杆:** 要加速 ALFWorld rollout,加 **env worker
(`pool_size`)**,不是加卡 —— 每步有 40–73% 是 GPU 在等 env 服务空转。(WebShop 无 `_TW_LOCK`、env 更轻,预期 env-bound
程度更低,但本轮未单独分解其 gen/算力。)

---

## 4. 并发 / ZMQ 修复 —— 与环境无关

FSDP→vLLM 权重传输死锁(每个独立 Ray 集群都取相同首个 job id `01000000` → 同一个 `/tmp` ZMQ socket → 44 分钟挂起)
及其修复(每个 verl 子进程导出 `VERL_RAY_JOB_ID` + 2 行 verl honor-override patch)完全在**与环境无关的 verl/Ray 平面**。

| | WebShop | ALFWorld |
|---|---|---|
| 测试 | 3 个并发 job(client-parallel + eval∥train) | 2 个并发训练 job,GPU {0,1}+{2,3} |
| 结果 | 修复后 PASS(rc=0) | **PASS**(两个 rc=0;A 392s,B 473s) |

ALFWorld 是*更强*的压力测试 —— 它慢的服务冷启动加宽了 socket 竞争窗口 —— 修复依然成立。这是预期结果:bug 和修复都不碰 env 服务。

---

## 5. 那条原则(为什么上面这些都成立)

ALFWorld 与 WebShop 在三个轴上不同 —— **episode 更长(50 vs 15 轮)**、**每步 env 更重(TextWorld + 进程级 `_TW_LOCK`)**、
**eval 更大更重**。每一个都改变墙钟去了哪:

```
            WebShop  ────────────────►  ALFWorld
 成本从:    GPU 算力        转移到:   eval 引擎冷启动  +  env 延迟(rollout)
```

这一个转移就预测了上面每一个结果:
- **eval 冷启动变大** → *摊销*它的模式(`worker`)赢,*重复*它的模式(`inline`)输 → **eval-mode 排名翻转**。
- **rollout 变得受 env 延迟约束** → 加卡不再帮生成(`gen` 平坦) → rollout 的杠杆变成 `pool_size`,而每卡*训练*惩罚不变
  (它本来就与 rollout 无关)。
- **训练器平面不受触动** → 并发修复原样 transfer。

**新环境的决策规则:** 估计(a)eval 引擎冷启动成本、(b)rollout 受 env 延迟约束的程度。(a)高 → 选 `worker`/`parallel`,
避开 `inline`。(b)高 → 先扩 `pool_size` 再加卡,且即使每步惩罚不缩小,1-GPU 训练在*相对*意义上仍可行。

---

## 6. 已定 vs 未决

**已定(两个环境都测了):** eval-mode 排名 + 解耦-eval 原则;~+38% 每步 1-GPU 惩罚;ALFWorld rollout env-bound(`gen` 平坦);
ZMQ 并发修复与环境无关。

**未决 / 尚未单独隔离:**
- **持久训练器(#4)在 ALFWorld 的相对收益** —— 预测更小(冷启动占 ALFWorld 更大墙钟的*比例*更小),但本轮未 A/B。
- **WebShop 的 gen/算力分解** —— 用于确认 WebShop rollout 确实比 ALFWorld 的平坦 gen *更不* env-bound(目前由"无 `_TW_LOCK`"推断)。
- **ALFWorld 全 val 数** —— 这些用 n=48;in-loop `valid_seen` 是 140,offline 集是 274(`tools/verl08_migration/eval_alfworld_by_tasktype.py`)。
- **多步稳态墙钟** —— scaling 探针是 1 步;多轮 run 确认墙钟惩罚收敛到每步的 +38%。

---

## 出处 & 另见
- **WebShop 数:** [`acceleration_results_cn.md`](./acceleration_results_cn.md)、
  [`acceleration_cn.md`](./acceleration_cn.md) §7.4(eval mode)/ §7.7(布局)/ §Lever #3。
- **ALFWorld 数:** [`alfworld_testing_cn.md`](./alfworld_testing_cn.md) §6(预测揭晓 + 记分卡);
  [`EXPERIMENTS.md`](../EXPERIMENTS.md) "ALFWorld acceleration economics (2026-06-30)"。
- **配置:** `tools/verl08_migration/accel/webshop/`、`…/accel/alfworld/`、`…/accel/client_parallel/`(各有 README)。
- **修复:** `tools/verl08_migration/patches/`(`VERL_RAY_JOB_ID` honor-override)。
- 英文版:[`acceleration_cross_env.md`](./acceleration_cross_env.md)。
