# TritonAscend / Final-EA Ascend 910B 环境重建教程

> 适用工程：`project3230617-388987-main` 及同版本 Final-EA 工程  
> 目标设备：Ascend 910B 系列（B3/B4 子型号以平台实际分配为准）  
> 已实测设备：Ascend 910B4-1  
> 本文记录的是从空白远程环境重建成功的完整流程。

## 1. 最终验证通过的组件组合

| 层级 | 组件 | 已验证版本/位置 | 来源 |
|---|---|---|---|
| 硬件与驱动 | NPU、驱动、`npu-smi` | 910B4-1，`npu-smi 25.5.2` | 远程平台镜像 |
| 主机工具 | GNU g++ | `/usr/bin/g++` | 远程平台镜像/系统包 |
| Python环境 | TriTrans | `/workspace/TriTrans`，Python 3.10.0 | `env_dep/TriTrans.tar.gz` |
| Python包 | torch | 2.6.0+cpu | TriTrans |
| Python包 | torch_npu | 2.6.0rc1 | TriTrans |
| Python包 | triton | 3.2.0 | TriTrans |
| CANN | Toolkit/HCCL/OPP | 8.3.RC1.alpha003，内部构建 8.3.T14.0.B101 | `env_dep` Toolkit `.run` |
| 算子包 | 910B Kernels | 8.3.RC1.alpha003 | `env_dep` Kernels `.run` |
| 数学库 | NNAL/ATB | 8.3.RC1.alpha003 | `env_dep` NNAL `.run` |
| 编译工具 | AscendNPU-IR | 1.0.0 | `env_dep/ascendnpu-ir...run` |

必须区分：平台提供 NPU 驱动、设备和 `npu-smi`；本流程确实从工程 `env_dep`
安装了 CANN、910B Kernels、NNAL、NPU-IR，并从压缩包建立 TriTrans。

## 2. 工程与资源位置

本文使用：

```bash
export PROJECT_DIR=/workspace/user_data/project3230617-388987-main
export SETUP_DIR=/workspace/user_data/TritonEA_910B_setup_20260813
```

`$PROJECT_DIR/env_dep` 应包含：

```text
TriTrans.tar.gz
Ascend-cann-toolkit_8.3.RC1.alpha003_linux-aarch64.run
Ascend-cann-kernels-910b_8.3.RC1.alpha003_linux-aarch64.run
Ascend-cann-nnal_8.3.RC1.alpha003_linux-aarch64.run
ascendnpu-ir_1.0.0_linux-aarch64.run
```

推荐每一步单独执行，看到 `PASS` 后再继续，不要把所有安装命令粘成一条长命令。

## 3. 第0步：只读审计

```bash
cd "$SETUP_DIR"
bash 00_readonly_audit.sh
bash 00_verify_uploaded_packages.sh
```

确认：

- `npu-smi info` 能识别 `910B*`，设备健康且无其他进程；
- 五个资源包均存在；
- 五个 SHA256 均为 `OK`；
- `/workspace` 至少预留数十 GB；
- 系统存在 `/usr/bin/g++`。

`g++` 不是 Python 包，不能通过 `pip install g++` 安装。若空白镜像缺少它，必须先获得
平台或老师授权，再使用系统包管理器，例如 Ubuntu：

```bash
apt-get update
apt-get install -y g++
```

## 4. 第1步：建立 TriTrans

```bash
cd "$SETUP_DIR"
bash 01_install_tritrans.sh
```

应存在：

```text
/workspace/TriTrans/bin/python
/workspace/TriTrans/bin/pip3
/workspace/TriTrans/bin/activate
/workspace/TriTrans/bin/conda-unpack
```

### 重要：此时不要导入 torch_npu

在 CANN 尚未安装时，`torch_npu` 依赖的 `libhccl.so` 不存在。如果此时运行
`import torch_npu`，会出现：

```text
ImportError: libhccl.so: cannot open shared object file
```

这不代表 TriTrans 压缩包损坏。只要上述四个文件存在且 Python/pip 可执行，就继续安装
CANN。真正的 `torch_npu` 导入与 NPU 实算放在完整环境验收阶段。

## 5. 第2步：安装 CANN、910B Kernels 与 NNAL

```bash
cd "$SETUP_DIR"
bash 02_install_cann_stack.sh
```

安装顺序：

```text
CANN Toolkit（含 runtime/compiler/HCCL/OPP）
→ 910B Kernels
→ NNAL/ATB
```

最关键的路径规则：传给安装器的根目录必须是：

```text
--install-path=/usr/local/Ascend
```

Kernels 还需要：

```text
--type=toolkit
```

不要写成：

```text
--install-path=/usr/local/Ascend/ascend-toolkit
```

错误的子目录会让 Kernels 安装器误报 Toolkit/NNAE/NNRT 未安装。

成功标志：

```text
PASS /usr/local/Ascend/ascend-toolkit/latest/bin/atc
PASS /usr/local/Ascend/ascend-toolkit/latest/bin/msprof
PASS /usr/local/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin/ccec
PASS 910B OPP/Kernel configuration
PASS /usr/local/Ascend/nnal/atb/set_env.sh
CANN Toolkit + 910B Kernels + NNAL: PASS
```

安装过程中 `perl locale` 和 `Running pip as root` 通常只是安装器警告；是否成功以明确的
`install success`、最终 `PASS` 和退出码为准。

NNAL 标准路径是：

```text
/usr/local/Ascend/nnal/atb/set_env.sh
```

若出现 `/usr/local/Ascend/nnal/nnal/atb`，说明安装根目录多嵌套了一层。优先按正确根目录
重装，不把符号链接当作标准安装结果。

## 6. 第3步：安装 AscendNPU-IR

```bash
cd "$SETUP_DIR"
bash 03_install_ascendnpu_ir.sh
```

成功后应能执行：

```bash
/usr/local/Ascend/tools/bishengir/bin/bishengir-compile --version
```

TriTrans/CANN 的环境脚本可能引用未定义的 `CONDA_PREFIX`、`ZSH_VERSION`。不要在
`set -u` 生效时直接 source；本工具包已临时关闭 nounset 后再恢复。

## 7. 第4步：配置统一环境和API

```bash
cd "$SETUP_DIR"
CHECK_API=1 bash 04_configure_runtime_and_api.sh
```

推荐配置：

```text
API_URL=https://api.deepseek.com
ENGINE=deepseek-v4-flash
LLM_MODELS=deepseek-v4-flash
```

如确需多模型，可填写：

```text
deepseek-v4-pro,deepseek-v4-flash
```

密钥保存到：

```text
/workspace/user_data/triton_ea_model.env
```

权限应为 `600`。不要把这个文件复制进工程、日志、备份分享包或提交包。

统一环境入口：

```text
/workspace/user_data/activate_project3230617_env.sh
```

新终端加载方式：

```bash
set +u
source /workspace/user_data/activate_project3230617_env.sh
set -u
```

## 8. 第5步：完整环境验收

```bash
cd "$SETUP_DIR"
bash 05_verify_environment.sh
```

必须检查：

- `python`、`pip3` 指向 `/workspace/TriTrans/bin`；
- `atc`、`msprof`、`ccec` 指向 `/usr/local/Ascend/ascend-toolkit/latest`；
- `bishengir-compile` 指向 `/usr/local/Ascend/tools/bishengir/bin`；
- Toolkit 版本为 8.3.RC1.alpha003 / 8.3.T14.0.B101；
- 910B OPP/Kernel 配置存在；
- NNAL 的 `libatb.so`、`libasdops.so` 存在；
- torch 2.6.0+cpu、torch_npu 2.6.0rc1、triton 3.2.0；
- `torch.npu.is_available()` 为 `True`；
- 安全的最小 NPU 张量计算通过；
- C++17 编译运行通过；
- 工程核心 Python 文件语法与导入通过。

参考成功结果：

```text
PASS=17 WARN=0 FAIL=0
Preflight result: PASS
Project imports: PASS
Environment and project verification: PASS
```

不要直接打印 NPU Tensor。把结果搬到 CPU 后再比较，避免部分 TorchNPU 构建额外触发
不受支持的比较内核。

## 9. 第6步：核验21算子数据与baseline

### 9.1 静态核验

```bash
cd "$SETUP_DIR"
source /workspace/user_data/activate_project3230617_env.sh

/workspace/TriTrans/bin/python 06_validate_dataset_baseline.py \
  /workspace/user_data/project3230617-388987-main
```

参考成功结果：

```text
Operators: 21 / 21
Dataset-matched baseline test_1 entries: 21
Required source/test files parsed: 168
Static validation: PASS
```

`baseline.json` 是50算子的超集，因此会提示另外29个 baseline 算子不在当前21算子
datasets 中；这是提示，不是失败。

### 9.2 实测21个官方主代码

```bash
cd "$SETUP_DIR"
export PROBE_MODE=main
export PROBE_REPEATS=1
export PROBE_TIMEOUT=360
bash 08_launch_baseline_probe.sh
```

查看进度：

```bash
bash 09_check_baseline_probe.sh
```

主代码 `<kernel>.py` 的21次结果是判断新环境与官方 baseline 是否吻合的主要依据。

如果要完整验证官方两个初始父代，可以在当前21次任务结束后再运行：

```bash
PROBE_MODE=both PROBE_REPEATS=1 bash 08_launch_baseline_probe.sh
```

这会测试42次，但会重复 main。不要与正在运行的21次任务并行，否则两个 profiler 任务会争抢
同一NPU并污染结果。`<kernel>_1.py` 可能不是生成 baseline 的那份实现，因此第二父代用于
确认功能和候选可执行性，不作为环境偏差的首要依据。

## 10. baseline结果解释

```text
difference_pct = (local_us / baseline_us - 1) × 100%
```

- 绝对偏差不超过10%：高度吻合；
- 10%～20%：通常可接受，必要时重复测量；
- 超过20%：先重复测量，再检查设备频率、系统负载、CANN/Kernels和profiler；
- 数微秒级算子受 profiler 固定开销影响明显，不能只看一次百分比；
- `success=True` 只代表公开测试与 msprof 成功，不代表隐藏测试通过。

## 11. 正式运行

完成环境和 baseline 验收后：

```bash
set +u
source /workspace/user_data/activate_project3230617_env.sh
set -u

cd /workspace/user_data/project3230617-388987-main
python -u main.py --input-dir ./datasets --output-dir ./output \
  2>&1 | tee ./full_run.log
```

打包前按工程说明使用 `package_output.py`，不要手工改变 output 目录层级。

## 12. 常见问题速查

### `libhccl.so` 缺失

- 若发生在只安装了 TriTrans、尚未安装 CANN 时：属于阶段性依赖尚未就绪，继续安装 CANN。
- 若 CANN 安装并加载环境后仍发生：检查 HCCL 安装日志、`LD_LIBRARY_PATH` 和
  `/usr/local/Ascend/ascend-toolkit/latest`。

### `CONDA_PREFIX: unbound variable` 或 `ZSH_VERSION: unbound variable`

加载环境前临时：

```bash
set +u
source <环境脚本>
set -u
```

### Kernels 报 Toolkit/NNAE/NNRT 未安装

检查是否错误使用：

```text
--install-path=/usr/local/Ascend/ascend-toolkit
```

正确安装根目录是 `/usr/local/Ascend`，Kernels 同时指定 `--type=toolkit`。

### NNAL 出现双层目录

标准路径应为 `/usr/local/Ascend/nnal/atb`。优先使用正确根目录重装；符号链接只作为记录在案
的临时兼容措施。

### GlusterFS出现 stale file handle

安装包和项目可保存在 `/workspace/user_data`，但大量 profiler 中间结果和密集目录操作优先放在
本地 `/workspace`，结束后再把摘要和压缩包复制回持久化目录。

## 13. 需要保存和分享的非敏感证据

建议保存：

- 五个安装包 SHA256；
- `npu-smi info`；
- CANN、Kernels、NNAL、NPU-IR版本文件；
- `python/pip3/g++/atc/msprof/ccec/bishengir-compile`实际路径；
- torch、torch_npu、triton版本和模块路径；
- `05_verify_environment.sh`日志；
- baseline静态报告和实测 `summary.txt`。

绝对不要分享：API_KEY、`triton_ea_model.env`、包含密钥的终端历史或日志。

