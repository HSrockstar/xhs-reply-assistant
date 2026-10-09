# xhs-reply-assistant · 个人回复助手

搭配 [xhs-comment-collector](https://github.com/HSrockstar/xhs-comment-collector) 使用的本机网页应用：导入其导出的 Q&A，整理回复单元，提取并编辑风格规则，再生成回复草稿。本项目独立于采集工具；通过 JSON 文件衔接。

默认入口为 **Codex CLI · gpt-6-luna · high**，复用已有登录。可主动切换到 DeepSeek Flash。向量编码及相似度计算使用 **BGE small zh v1.5 / CUDA**；GPU 或模型不可用时界面明确显示关键词检索降级。

## 现在使用

当前机器已安装依赖及固定版本模型，服务地址：<http://127.0.0.1:8081>。

以后重新启动，在 PowerShell 执行：

```powershell
cd 'E:\关于我和你\个人回复助手'
.\scripts\start.ps1
```

脚本只绑定 `127.0.0.1`。在启动终端按 Ctrl+C 停止。初次交付的后台实例 PID 保存在 `.verification/server.pid`；结束该实例前可用 `Get-NetTCPConnection -LocalPort 8081 -State Listen` 核对所属进程。

1. 在 **Q&A 资料库**选择原工作台 `data/qa/` 中导出的 JSON，先预览统计和原文，再确认导入。
2. 按“待整理”筛选，人工选择问题、上下文和回答，或点击“模型辅助建议”；检查分组后确认。待整理单元默认不参与检索或风格提取。
3. 在 **风格设置**点击“两轮提取”。查看原文依据、场景差异和规则，修改后保存并启用。历史版本可载入、比较、导出。失败时可从有效批次检查点继续。
4. 在 **回复工作台**填写问题和必要背景，选择长度、场景和入口。生成后检查参考，编辑正文，再复制或保存最终稿。“采用为示例”是保存后的独立操作。
5. 在 **模型设置**保存默认入口、超时、明确确认的背景和个人偏好，查看 CUDA 状态，下载备份。

正式资料库初始为空。本次没有把真实评论发给生成模型；点击风格提取、辅助整理或生成后，选定材料会传给你选择的模型入口。

## 安装到另一台 Windows 机器

需要 Python 3.11、可用的 NVIDIA 驱动及 CUDA GPU；Codex CLI 需要独立安装并登录。

```powershell
.\scripts\setup.ps1 -Python 'C:\路径\python.exe'
.\scripts\start.ps1
```

安装脚本创建独立 `.venv`，安装锁定依赖与 PyTorch 2.8.0 CUDA 12.8，下载固定模型 revision，并实际验证模型参数、编码张量和相似度计算设备。首次下载的 PyTorch 包约 3.4 GB，需要稳定网络。安装失败可重试；不会修改原采集项目环境。

模型 revision：`7999e1d3359715c523056ef9478215996d62a620`。模型文件位于 `data/models/bge-small-zh-v1.5/`；索引缓存位于 `data/index/`，损坏后可重建。模型权重和缓存不包含在业务备份中。

DeepSeek 密钥可通过服务启动时的 `DEEPSEEK_API_KEY` 环境变量，或工作台临时密码框提供。临时密钥提交后清空，不写入数据库、导出、日志或浏览器存储。任务失败后由用户主动重试或切换入口。

## 数据与恢复

- `data/reply.sqlite3`：示例、画像版本、任务、反馈与非敏感设置。
- `data/imports/`：原始导入快照；编辑只改变派生文本，原文保留。
- `data/exports/`：一致性备份和恢复前自动备份。
- `data/index/`：可重建的向量缓存。
- `.verification/`：本机验证结果与临时文件，不作为资料库。

备份包括数据库及原始导入文件。恢复会先校验归档、schema 和数据结构，并备份当前资料；失败时保留或恢复原库。有活动任务时先等待或取消再恢复。服务重启后旧任务标记中断，不自动重复调用模型。

画像按原始会话去重，只统计目标回答；完整单元按最多 18,000 字、30 条分批，两轮共 `2 × 批次数` 次调用。每个已验证批次独立保存检查点；单个完整单元超预算会提示人工整理。各批次候选按轮次合并为最多八条自动核心规则，完整分析结果保留在画像中。删除和停用的手工规则在刷新时保留用户决定，最终启用仍由用户选择。

## 验证与评测

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ruff check app tests scripts
node --check app/static/app.js
.venv\Scripts\python.exe -m scripts.verify_gpu --output .verification/gpu.json --local-model-path data/models/bge-small-zh-v1.5
```

以下两条会调用真实 Codex，按需主动运行：

```powershell
.venv\Scripts\python.exe -m scripts.check_provider
.venv\Scripts\python.exe -m scripts.check_workflow
```

仅解析文件、输出脱敏数量、不入库：

```powershell
.venv\Scripts\python.exe -m scripts.inspect_import 'E:\关于我和你\小红书爬虫\data\qa\603923a9-0ba9-474e-a5be-b3007fa6bc89.json'
```

效果评测需先导入并整理至少八个独立来源的可用单元。`python -m scripts.evaluate prepare` 会先备份，固定留出会话并停用相应训练/检索用途；检查生成的 `data/evaluation/study.json`，提取并启用不含留出资料的新画像，再执行 `python -m scripts.evaluate run`。后者明确发起 20 题 × 3 方案的真实模型任务，结果保存到同目录 `results.json`，由用户填写语气、适用性、可采用性和编辑评价。不能用合成测试代替这一步。

当前验证结果与未完成验收见 [验收记录](docs/验收记录.md)。原始开发依据保存在 [开发计划 v1.1](docs/开发计划.md)，其“尚未实施”状态作为计划原文保留；以验收记录了解当前进度。

## 实现边界

这是单用户本机草稿助手。历史材料只作内容与表达参考，模型可以补充一般知识；身份、经历、成绩和未来承诺仅使用明确确认的背景。来源面板记录提供的参考与模型声明采用的参考，不能据此认为每句话都经过事实核实。实时事实仍需用户核对。

本版不微调、不自动发到社交平台、不提供远程多人入口。语气是否像本人需要真实资料和用户评价。DeepSeek 适配器已做模拟错误/凭据测试，当前尚未用真实密钥完成端到端验收。

## 官方依据

[Codex 非交互运行](https://learn.chatgpt.com/docs/non-interactive-mode)、[BGE 模型卡](https://huggingface.co/BAAI/bge-small-zh-v1.5)、[Sentence Transformers 编码文档](https://www.sbert.net/examples/sentence_transformer/applications/computing-embeddings/README.html)、[PyTorch 历史版本](https://pytorch.org/get-started/previous-versions/)、[DeepSeek Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/)。
