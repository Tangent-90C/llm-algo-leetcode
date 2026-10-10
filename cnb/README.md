# CNB 环境说明

本目录用于本项目的 CNB 云端统一教学环境。CNB 的角色不是替代本地开发，而是提供一套更一致、可复现、对 GPU 更友好的远程运行入口。

可以把 CNB 理解为一个基于容器化的云端统一开发与验证环境：它包含 Git 仓库工作区、平台级流水线、在线编辑与运行能力，并可按需接入 GPU 算力。

## 定位

- **GitHub**：主源仓库，负责正文、练习、导学和正式发布
- **本地环境**：适合日常开发、章节编辑和 Notebook 验证
- **CNB 环境**：适合统一镜像、云端运行和跨平台复现

## 初学者怎么选

如果你第一次接触这个项目，按下面顺序选环境：

1. 先看在线站点和 Chapter 0 / 1 导学。
2. 如果只是入门练习，先用 Colab 或其他在线 Notebook。
3. 如果要系统学 Chapter 2，优先本地 conda 环境。
   如果不想手动配置本地环境，Chapter 2 也可以直接用 CNB 作为第二选择。
4. 如果要做 Chapter 3，再切到本地 GPU 或 CNB GPU 入口。
5. 如果你想和别人用同一套环境跑，优先 CNB。

默认原则：

- **先能看懂，再考虑配置**
- **先用 CPU 跑通，再考虑 GPU**
- **先验证 Chapter 0 / 1 / 2，再单独处理 Chapter 3**

## 环境骨架

- `cnb/environment.yml`：CNB CPU-first Conda 环境入口
- `cnb/environment-gpu.yml`：CNB GPU Conda 环境入口
- `environment.yml`：本地主入口环境

CNB 使用两个独立入口：`.ide/Dockerfile` 服务 CPU `vscode`，`.ide/Dockerfile.gpu` 服务 GPU `vscode-gpu`。GPU 镜像只安装用户态 CUDA PyTorch 与 Triton；实际 GPU、驱动和可用 backend 仍以会话预检为准。

## 为什么需要 `vscode`

本项目的 CNB 不是只跑流水线的 CI 容器，还需要给读者一个可以直接进入、直接运行、直接调试的交互式工作区。

- `push` / `pull_request`：负责校验、构建、自动合并等流水线任务
- `vscode`：负责读者打开后的交互开发环境，要求能直接使用 Python、Notebook、PyTorch、Triton 等基础工具链

这样 CNB 才能同时满足两类需求：

1. 仓库改动的自动化验证
2. 学习者和维护者的交互式开发与运行

## 当前状态

- CNB 已提供 CPU `vscode` 与 GPU `vscode-gpu` 两个镜像入口。
- `main.push` 只运行 CPU smoke，不把包导入当作 GPU 验证。
- GPU 会话需执行预检脚本；GPU 型号、驱动、PyTorch、CUDA 与 Triton 版本应随正式结果写入 runtime 快照。
- 当前尚未提交固定镜像 digest 或 lock 文件，版本组合仍需要在首次实机验证后冻结。

## 在 CNB 里怎么验

先激活项目环境，再按章节验证。

```bash
. /opt/conda/etc/profile.d/conda.sh
conda activate llm_algo_cnb_dev

python --version
python -m pip --version
python -c "import torch; print(torch.__version__)"
python tools/test_chapter0_1_notebooks.py
python tools/test_notebook_answers.py --all --dir 02_PyTorch_Algorithms --mode both
```

如果当前 CNB 实例没有 GPU，`nvidia-smi` 可能不可用，`torch.cuda.is_available()` 也可能是 `False`。这种情况下，Chapter 3 只能验证导学页、链接和非 GPU 路径，不能把 GPU 结果当成最终通过。

## Chapter 3 GPU 入口

Chapter 3 需要独立的 GPU 会话，不应默认依赖当前这套 CPU 交互环境。

- CPU 交互环境：用于 Chapter 0 / 1 / 2
- GPU 验证环境：用于 Chapter 3 的内核、显存和通信验证
- 对应的 CNB 入口名：`vscode-gpu`
- Conda 环境名：`llm_algo_cnb_gpu`
- 统一预检脚本：`bash tools/validate_cnb_gpu.sh`

如果平台分配的当前会话没有 GPU，只能把它视为 Chapter 0 / 1 / 2 的统一环境，不要把 Chapter 3 的 GPU 结果当作最终验收。

从仓库根目录创建 Conda 环境时，可使用：

```bash
conda env create -f cnb/environment.yml       # CPU-first
conda env create -f cnb/environment-gpu.yml   # GPU
```

## 适用场景

- 希望减少本地系统差异带来的配置成本
- 希望为 Chapter 3 提供更稳定的 GPU / Triton 实验入口
- 希望为课程、训练营或团队协作提供统一环境

## 说明

CNB 的具体镜像、启动方式和预检约定由 `.cnb.yml`、本目录文件与 `docs/guide.md` 共同维护；环境 profile 与运行时字段登记在 `docs/gpu_environment_assets.md`。
