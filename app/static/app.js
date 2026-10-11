"use strict";

(function () {
  const $ = (selector, root) => (root || document).querySelector(selector);
  const $$ = (selector, root) => Array.from((root || document).querySelectorAll(selector));
  const pageNames = { workbench: "回复工作台", library: "Q&A 资料库", style: "风格设置", settings: "模型设置" };
  const pageDescriptions = {
    workbench: "参考历史回复与风格画像，写出更贴近自己语气的回复草稿。",
    library: "保存完整对话，让历史回复成为内容与风格的参考。",
    style: "分析回复策略与表达习惯，建立可编辑、可追溯的风格画像。",
    settings: "配置模型入口与检索设备，管理本地工作空间。"
  };
  const state = {
    page: "workbench", mode: "full", settings: {}, providers: {}, runtime: {}, style: null,
    styleProfile: null, styleRules: [], styleDraftResult: null, examples: [], examplesTotal: 0,
    examplesOffset: 0, selectedExample: null, importFile: null, importPreview: null,
    organizeSuggestions: null, jobs: new Map(), selectedJobId: null, drafts: new Map(),
    savedFeedback: new Map(), feedbackMeta: new Map(), feedbackItems: [], completionHandled: new Set(), providerTouched: false
  };
  const storageKey = "reply-assistant-session-draft-v1";
  let toastTimer = 0, pollTimer = 0, pollBusy = false;

  function make(tag, className, value) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (value !== undefined && value !== null) node.textContent = String(value);
    return node;
  }
  function setText(id, value) {
    const node = document.getElementById(id);
    if (node) node.textContent = value === undefined || value === null ? "" : String(value);
  }
  function showToast(message, kind) {
    const node = $("#toast");
    node.textContent = String(message || "");
    node.className = "toast" + (kind ? " " + kind : "");
    node.hidden = false;
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(function () { node.hidden = true; }, 4200);
  }
  function asObject(value) { return value && typeof value === "object" && !Array.isArray(value) ? value : {}; }
  function asArray(value) { return Array.isArray(value) ? value : []; }
  function displayValue(value) {
    if (value === null || value === undefined || value === "") return "—";
    if (typeof value === "object") {
      try { return JSON.stringify(value, null, 2); } catch (_) { return "[无法显示]"; }
    }
    if (typeof value === "boolean") return value ? "是" : "否";
    return String(value);
  }
  function cloneValue(value) {
    if (value === undefined || value === null) return value;
    try { return structuredClone(value); } catch (_) { return JSON.parse(JSON.stringify(value)); }
  }
  function requestId() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") return window.crypto.randomUUID();
    return "req-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2);
  }
  async function api(path, options) {
    const config = options || {}, method = String(config.method || "GET").toUpperCase();
    const headers = new Headers(config.headers || {});
    const init = { method: method, headers: headers, credentials: "same-origin" };
    if (method !== "GET" && method !== "HEAD") headers.set("X-Requested-With", "reply-assistant");
    if (config.formData) init.body = config.formData;
    else if (config.body !== undefined) {
      headers.set("Content-Type", "application/json");
      init.body = JSON.stringify(config.body);
    }
    const response = await fetch(path, init);
    let payload = null;
    if ((response.headers.get("content-type") || "").includes("json")) {
      try { payload = await response.json(); } catch (_) { payload = null; }
    } else if (!config.rawResponse) {
      try { payload = await response.text(); } catch (_) { payload = null; }
    }
    if (!response.ok) {
      const outer = asObject(payload), detail = asObject(outer.error || outer.detail || payload);
      const code = detail.code ? String(detail.code) + "：" : "";
      const message = detail.message || outer.message || (typeof payload === "string" ? payload : "") || ("请求失败（HTTP " + response.status + "）");
      const error = new Error(code + String(message) + (detail.retryable === true ? "（可以重试）" : ""));
      error.code = detail.code || ""; error.retryable = detail.retryable === true; error.status = response.status;
      throw error;
    }
    return config.rawResponse ? response : payload;
  }
  function reportError(error, prefix) {
    showToast((prefix ? prefix + "：" : "") + (error && error.message ? error.message : String(error)), "error");
  }
  function saveSessionDraft() {
    const draft = {
      question: $("#question-input").value, background: $("#background-input").value,
      instructions: $("#instructions-input").value, length: $("#length-input").value,
      scene: $("#scene-input").value, mode: state.mode, drafts: Array.from(state.drafts.entries()),
      feedbackMeta: Array.from(state.feedbackMeta.entries())
    };
    try { window.sessionStorage.setItem(storageKey, JSON.stringify(draft)); } catch (_) { /* 保留当前页面编辑 */ }
  }
  function loadSessionDraft() {
    try {
      const raw = window.sessionStorage.getItem(storageKey);
      if (!raw) return;
      const saved = asObject(JSON.parse(raw));
      $("#question-input").value = typeof saved.question === "string" ? saved.question : "";
      $("#background-input").value = typeof saved.background === "string" ? saved.background : "";
      $("#instructions-input").value = typeof saved.instructions === "string" ? saved.instructions : "";
      if (typeof saved.length === "string") $("#length-input").value = saved.length;
      if (typeof saved.scene === "string") $("#scene-input").value = saved.scene;
      if (["full", "style", "base"].includes(saved.mode)) setMode(saved.mode, false);
      for (const pair of asArray(saved.drafts)) if (Array.isArray(pair) && pair.length === 2) state.drafts.set(String(pair[0]), String(pair[1] || ""));
      for (const pair of asArray(saved.feedbackMeta)) if (Array.isArray(pair) && pair.length === 2) state.feedbackMeta.set(String(pair[0]), asObject(pair[1]));
    } catch (_) { /* 忽略页签草稿缓存错误 */ }
  }
  function setMode(mode, save) {
    state.mode = mode;
    for (const button of $$(".mode-button")) {
      button.classList.toggle("active", button.dataset.mode === mode);
      button.setAttribute("aria-pressed", String(button.dataset.mode === mode));
    }
    if (save !== false) saveSessionDraft();
  }
  function navigate(page) {
    if (!pageNames[page]) return;
    state.page = page;
    for (const section of $$(".page-section")) section.hidden = section.dataset.section !== page;
    for (const button of $$(".nav-link")) {
      button.classList.toggle("active", button.dataset.page === page);
      if (button.dataset.page === page) button.setAttribute("aria-current", "page");
      else button.removeAttribute("aria-current");
    }
    setText("page-title", pageNames[page]);
    setText("page-breadcrumb", pageNames[page]);
    setText("page-description", pageDescriptions[page]);
    window.scrollTo({ top: 0, behavior: "smooth" });
    if (page === "library" && !state.examples.length) loadExamples(false);
    if (page === "style" && !state.style) loadStyle(false);
    if (page === "settings" && !Object.keys(state.providers).length) loadProviders(false);
  }
  function consumeApiKey() {
    const field = $("#provider-key"), value = field.value;
    field.value = "";
    return value;
  }
  function writeWithTemporaryKey(path, body, providerName) {
    const key = consumeApiKey();
    const payload = Object.assign({}, body, { provider: providerName || $("#provider-select").value, request_id: requestId() });
    if (key) payload.api_key = key;
    return api(path, { method: "POST", body: payload });
  }
  function statusLabel(status) {
    const labels = {
      queued: "排队中", running: "运行中", succeeded: "已完成", failed: "失败",
      cancelled: "已取消", interrupted: "已中断", usable: "可用", ready: "可用", unresolved: "待整理", needs_review: "待整理", excluded: "已排除",
      disabled: "已停用", excluded: "已排除"
    };
    return labels[status] || String(status || "未知");
  }
  function statusPill(status) {
    const pill = make("span", "status-pill " + String(status || ""));
    pill.textContent = statusLabel(status);
    return pill;
  }
  function isTerminal(status) { return ["succeeded", "failed", "cancelled", "interrupted"].includes(status); }
  function isActive(status) { return status === "queued" || status === "running"; }
  function jobKindLabel(kind) {
    const labels = { generate: "回复生成", generation: "回复生成", organize: "示例整理", style: "风格提取", index: "索引重建", test: "入口测试" };
    return labels[kind] || String(kind || "后台任务");
  }
  function jobTitle(job) {
    const input = asObject(job.input), question = String(input.question || input.example_title || "");
    return question ? jobKindLabel(job.kind) + " · " + question.slice(0, 26) : jobKindLabel(job.kind);
  }
  function jobTime(value) {
    if (!value) return "";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    return new Intl.DateTimeFormat("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }).format(date);
  }
  function currentDraft(job) {
    const id = String(job.id);
    if (state.drafts.has(id)) return state.drafts.get(id);
    try {
      const raw = window.sessionStorage.getItem(storageKey);
      if (raw) for (const pair of asArray(asObject(JSON.parse(raw)).drafts)) {
        if (Array.isArray(pair) && String(pair[0]) === id) {
          state.drafts.set(id, String(pair[1] || ""));
          return state.drafts.get(id);
        }
      }
    } catch (_) { /* 忽略损坏的页签草稿 */ }
    if (typeof asObject(job.result).reply_text !== "string") return "";
    const initial = job.result.reply_text;
    state.drafts.set(id, initial);
    return initial;
  }
  function isGenerationJob(job) {
    return typeof asObject(job.result).reply_text === "string" || ["generate", "generation"].includes(job.kind);
  }
  function renderJobList() {
    const list = $("#job-list");
    list.replaceChildren();
    const jobs = Array.from(state.jobs.values()).sort(function (a, b) { return String(b.created_at || "").localeCompare(String(a.created_at || "")); });
    if (!jobs.length) { list.append(make("p", "muted empty-state", "还没有生成任务。")); return; }
    for (const job of jobs) {
      const button = make("button", "job-item" + (String(job.id) === String(state.selectedJobId) ? " active" : ""));
      button.type = "button"; button.dataset.jobId = String(job.id);
      button.setAttribute("aria-label", jobTitle(job) + "，" + statusLabel(job.status));
      button.append(make("span", "job-item-title", jobTitle(job)), statusPill(job.status));
      const time = make("span", "job-item-time", jobTime(job.created_at)); time.dataset.jobTime = "true";
      const stage = make("span", "job-item-stage", job.stage || ""); stage.dataset.jobStage = "true";
      button.append(time, stage); list.append(button);
    }
  }
  function renderJobError(parent, error) {
    const message = asObject(error).message || error || "任务失败";
    parent.append(make("div", "inline-note", message));
  }
  function renderJobDetail(job) {
    const host = $("#job-detail");
    host.replaceChildren();
    if (!job) {
      const empty = make("div", "empty-state");
      empty.append(make("span", "empty-orbit", "✦"), make("p", "", "选择一个任务查看进度和回复草稿。"));
      host.append(empty); return;
    }
    const header = make("div", "job-detail-header"), titleBox = make("div"), right = make("div");
    titleBox.append(make("h4", "job-detail-title", jobTitle(job)));
    titleBox.append(make("div", "job-detail-meta", "任务 " + String(job.id) + (job.created_at ? " · " + jobTime(job.created_at) : "")));
    const pill = statusPill(job.status); pill.classList.add("job-detail-status"); right.append(pill);
    if (isActive(job.status)) {
      const cancel = make("button", "text-button danger-text", "取消任务");
      cancel.type = "button"; cancel.dataset.action = "cancel-job"; cancel.dataset.jobId = String(job.id); right.append(cancel);
    }
    header.append(titleBox, right); host.append(header);
    const resultMeta = asObject(job.result);
    if (resultMeta.provider || resultMeta.model) host.append(make("div", "job-detail-meta", "入口：" + displayValue(resultMeta.provider) + " · 模型：" + displayValue(resultMeta.model)));
    const stage = make("div", "job-stage", job.stage ? "当前阶段：" + String(job.stage) : "任务状态：" + statusLabel(job.status));
    stage.id = "selected-job-stage"; host.append(stage);
    if (job.kind === "style") {
      const batchCount = asArray(asObject(job.input).batches).length || 1;
      host.append(make("p", "tiny muted", "完整单元分批：" + batchCount + " 批；两轮计划 " + (batchCount * 2) + " 次调用；本任务已调用 " + asArray(job.calls).length + " 次。"));
    }
    if (job.status === "failed") renderJobError(host, job.error);
    else if (job.status === "interrupted") renderJobError(host, job.error || "服务重启后任务中断；如支持，可在风格设置中继续。");
    const checkpoint = asObject(job.checkpoint);
    if (job.status === "failed" && checkpoint.index_result !== undefined) {
      const failure = make("section", "reference-group");
      failure.append(make("h4", "", "索引故障检查点"));
      failure.append(make("pre", "safe-text", displayValue(checkpoint.index_result)));
      host.append(failure);
    }
    if (isGenerationJob(job)) {
      const result = asObject(job.result);
      if (job.status === "succeeded" || result.reply_text) {
        const label = make("label", "field-label", "可编辑回复正文"), editor = make("textarea", "result-editor");
        editor.id = "selected-reply-draft"; editor.rows = 9; editor.value = currentDraft(job);
        editor.addEventListener("input", function () { state.drafts.set(String(job.id), editor.value); saveSessionDraft(); });
        label.append(editor); host.append(label);
        const feedbackControls = make("div", "form-grid two-columns feedback-controls");
        const ratingLabel = make("label", "field-label"); ratingLabel.append(make("span", "", "这次回复评分"));
        const rating = make("select"); rating.id = "selected-rating";
        for (const choice of [["", "不评分"], ["1", "1 分"], ["2", "2 分"], ["3", "3 分"], ["4", "4 分"], ["5", "5 分"]]) {
          const option = make("option", "", choice[1]); option.value = choice[0]; rating.append(option);
        }
        const meta = state.feedbackMeta.get(String(job.id)) || {};
        rating.value = meta.rating === null || meta.rating === undefined ? "" : String(meta.rating);
        rating.addEventListener("change", function () { state.feedbackMeta.set(String(job.id), { rating: rating.value, reasons: splitTags($("#selected-reasons").value) }); saveSessionDraft(); });
        ratingLabel.append(rating);
        const reasonsLabel = make("label", "field-label"); reasonsLabel.append(make("span", "", "修改原因（可选，逗号分隔）"));
        const reasons = make("input"); reasons.type = "text"; reasons.id = "selected-reasons";
        reasons.value = asArray(meta.reasons).join(", ");
        reasons.addEventListener("input", function () { state.feedbackMeta.set(String(job.id), { rating: rating.value, reasons: splitTags(reasons.value) }); saveSessionDraft(); });
        reasonsLabel.append(reasons);
        feedbackControls.append(ratingLabel, reasonsLabel); host.append(feedbackControls);
        const actions = make("div", "result-actions");
        for (const spec of [["copy-draft", "button button-secondary", "复制正文"], ["save-final", "button button-primary", "保存最终稿"], ["adopt-final", "button button-quiet", "采用为示例"]]) {
          const button = make("button", spec[1], spec[2]); button.type = "button"; button.dataset.action = spec[0]; actions.append(button);
        }
        host.append(actions);
        const saved = state.savedFeedback.get(String(job.id));
        if (saved) host.append(make("p", "tiny muted", saved.final_text === editor.value ? "当前稿已保存，可单独采用为示例。" : "稿件已保存；编辑后请再次保存，再采用为示例。"));
        host.append(renderResultNotes(result));
      } else {
        const question = asObject(job.input).question;
        if (question) host.append(make("div", "reference-group", question));
      }
    } else if (job.status === "succeeded" && job.result !== undefined) {
      host.append(make("pre", "safe-text", displayValue(job.result)));
    }
  }
  function renderSelectedJobStatus(job) {
    if (String(job.id) !== String(state.selectedJobId)) return;
    const host = $("#job-detail"), pill = $(".job-detail-status", host), stage = $("#selected-job-stage");
    if (pill) { pill.className = "status-pill job-detail-status " + String(job.status || ""); pill.textContent = statusLabel(job.status); }
    if (stage) stage.textContent = job.stage ? "当前阶段：" + String(job.stage) : "任务状态：" + statusLabel(job.status);
  }
  function upsertJob(rawJob) {
    const job = asObject(rawJob);
    if (job.id === undefined || job.id === null) return null;
    const id = String(job.id), old = state.jobs.get(id), merged = Object.assign({}, old || {}, job);
    state.jobs.set(id, merged);
    if (isGenerationJob(merged)) currentDraft(merged);
    if (!old || old.status !== merged.status) renderJobList();
    renderSelectedJobStatus(merged);
    if (!old || old.status !== merged.status || (old.result === undefined && merged.result !== undefined)) {
      if (id === String(state.selectedJobId)) renderJobDetail(merged);
    }
    if (merged.status === "succeeded" && old && isActive(old.status) && !state.completionHandled.has(id)) {
      state.completionHandled.add(id); handleCompletedJob(merged);
    }
    return merged;
  }
  async function loadJobs(showErrors) {
    try {
      const payload = await api("/api/jobs?limit=30"), items = asArray(asObject(payload).items);
      for (const job of items) upsertJob(job);
      if (state.selectedJobId && state.jobs.has(String(state.selectedJobId))) renderJobDetail(state.jobs.get(String(state.selectedJobId)));
      else if (items.length && state.selectedJobId === null) {
        const first = items.slice().sort(function (a, b) { return String(b.created_at || "").localeCompare(String(a.created_at || "")); })[0];
        if (first && first.id !== undefined) selectJob(first.id);
      }
      return items;
    } catch (error) {
      if (showErrors) reportError(error, "读取任务失败");
      else $("#job-list").replaceChildren(make("p", "muted empty-state", "暂时无法读取任务状态。"));
      return [];
    }
  }
  function selectJob(id) {
    state.selectedJobId = String(id);
    renderJobList(); renderJobDetail(state.jobs.get(String(id)));
  }
  async function pollJobs() {
    if (pollBusy) return;
    const active = Array.from(state.jobs.values()).filter(function (job) { return isActive(job.status); });
    if (!active.length) return;
    pollBusy = true;
    try {
      await Promise.all(active.map(async function (job) {
        try { const payload = await api("/api/jobs/" + encodeURIComponent(String(job.id))); upsertJob(asObject(payload).job || payload); }
        catch (_) { /* 单次轮询错误不清空已有结果 */ }
      }));
    } finally { pollBusy = false; }
  }
  async function cancelJob(id) {
    try {
      const payload = await api("/api/jobs/" + encodeURIComponent(String(id)) + "/cancel", { method: "POST", body: {} });
      const job = asObject(payload).job || payload;
      if (job && job.id !== undefined) upsertJob(job); else await loadJobs(false);
      showToast("已提交取消请求。", "success");
    } catch (error) { reportError(error, "取消任务失败"); }
  }
  async function createGeneration() {
    const question = $("#question-input").value.trim();
    if (!question) { showToast("请先填写要回复的问题或评论。", "error"); $("#question-input").focus(); return; }
    const body = {
      question: question,
      background: $("#background-input").value || state.settings.confirmed_background || "",
      instructions: $("#instructions-input").value,
      length: $("#length-input").value, scene: $("#scene-input").value, provider: $("#provider-select").value,
      api_key: consumeApiKey(), request_id: requestId(), mode: state.mode
    };
    const button = $("#generate-button"); button.disabled = true;
    try {
      const payload = await api("/api/generations", { method: "POST", body: body }), job = asObject(payload).job || payload;
      if (job.id === undefined) throw new Error("服务没有返回任务编号。");
      const added = upsertJob(job); selectJob(added.id); saveSessionDraft();
      showToast("任务已创建，进度会在这里更新。", "success");
    } catch (error) { reportError(error, "提交生成失败"); }
    finally { button.disabled = false; }
  }
  function addReferenceItems(parent, title, value) {
    const group = make("section", "reference-group"); group.append(make("h4", "", title));
    const entries = asArray(value);
    if (typeof value === "string") group.append(make("p", "tiny", value));
    else if (entries.length) for (const item of entries) {
      const outer = asObject(item), obj = outer.example ? asObject(outer.example) : outer, row = make("div", "reference-item");
      row.append(make("p", "", obj.content || obj.text || obj.answer_text || obj.note || displayValue(item)));
      const id = obj.example_id !== undefined ? obj.example_id : obj.id;
      if (id !== undefined && id !== null) {
        const link = make("button", "reference-link", "查看示例 " + String(id));
        link.type = "button"; link.dataset.action = "open-example"; link.dataset.exampleId = String(id); row.append(link);
      }
      group.append(row);
    } else if (value !== undefined && value !== null && value !== "") group.append(make("pre", "safe-text", displayValue(value)));
    else group.append(make("p", "tiny muted", "本次没有返回这类参考。"));
    parent.append(group);
  }
  function renderResultNotes(result) {
    const notes = make("div", "result-notes");
    if (result.supplement_summary) {
      const box = make("div", "reference-group");
      box.append(make("h4", "", "模型补充说明"), make("p", "tiny", displayValue(result.supplement_summary))); notes.append(box);
    }
    if (result.clarification_needed) notes.append(make("div", "inline-note", "建议确认：" + displayValue(result.clarification_needed)));
    if (asArray(result.review_notes).length) {
      const box = make("div", "reference-group");
      box.append(make("h4", "", "检查提示"), make("p", "tiny", result.review_notes.join("\n"))); notes.append(box);
    }
    if (result.references) {
      const refs = asObject(result.references); addReferenceItems(notes, "内容参考", refs.content); addReferenceItems(notes, "风格参考", refs.style);
    }
    if (asArray(result.reference_usage).length) addReferenceItems(notes, "本次引用用途", result.reference_usage);
    return notes;
  }
  async function copyDraft() {
    const editor = $("#selected-reply-draft");
    if (!editor) return;
    try { await navigator.clipboard.writeText(editor.value); showToast("已复制回复正文。", "success"); }
    catch (_) {
      editor.focus(); editor.select();
      const copied = document.execCommand("copy");
      showToast(copied ? "已复制回复正文。" : "复制失败，请手动选择正文。", copied ? "success" : "error");
    }
  }
  async function loadSettings(showErrors) {
    try {
      const payload = await api("/api/settings");
      state.settings = asObject(asObject(payload).settings || payload);
      const s = state.settings;
      $("#setting-provider").value = s.provider || "codex";
      $("#setting-codex-model").value = s.codex_model || "gpt-6-luna";
      $("#setting-codex-reasoning").value = s.codex_reasoning_effort || "high";
      $("#setting-deepseek-model").value = s.deepseek_model || "";
      $("#setting-codex-timeout").value = s.codex_timeout_seconds || 240;
      $("#setting-deepseek-timeout").value = s.deepseek_timeout_seconds || 180;
      $("#setting-background").value = s.confirmed_background || "";
      $("#setting-preferences").value = s.user_preferences || "";
      if (!state.providerTouched && s.provider) $("#provider-select").value = s.provider;
      setText("settings-saved-state", "已读取本机保存设置");
      updateCapabilityNote();
    } catch (error) {
      if (showErrors) reportError(error, "读取设置失败");
      else setText("settings-saved-state", "暂时无法读取设置");
    }
  }
  async function saveSettings() {
    const body = {
      provider: $("#setting-provider").value, codex_model: $("#setting-codex-model").value.trim(),
      codex_reasoning_effort: $("#setting-codex-reasoning").value,
      deepseek_model: $("#setting-deepseek-model").value.trim(),
      codex_timeout_seconds: Number($("#setting-codex-timeout").value),
      deepseek_timeout_seconds: Number($("#setting-deepseek-timeout").value),
      confirmed_background: $("#setting-background").value, user_preferences: $("#setting-preferences").value
    };
    try {
      const payload = await api("/api/settings", { method: "PATCH", body: body });
      state.settings = asObject(asObject(payload).settings || payload);
      if (!state.providerTouched) $("#provider-select").value = body.provider;
      setText("settings-saved-state", "保存于 " + new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" }));
      updateCapabilityNote(); showToast("非敏感设置已保存。", "success");
    } catch (error) { reportError(error, "保存设置失败"); }
  }
  async function restoreDefaults() {
    if (!window.confirm("将恢复服务默认设置并覆盖当前已保存的设置。是否继续？")) return;
    try {
      await api("/api/settings", { method: "PATCH", body: { restore_defaults: true } });
      await loadSettings(false); showToast("已恢复默认设置。", "success");
    } catch (error) { reportError(error, "恢复默认设置失败"); }
  }
  async function loadProviders(showErrors) {
    try {
      state.providers = asObject(await api("/api/providers")); renderProviders(); updateCapabilityNote();
    } catch (error) {
      state.providers = {}; $("#provider-status").replaceChildren(make("p", "muted", "暂时无法读取模型入口状态。"));
      if (showErrors) reportError(error, "读取模型状态失败");
    }
  }
  function renderProviders() {
    const host = $("#provider-status"); host.replaceChildren();
    for (const pair of [["codex", "Codex"], ["deepseek", "DeepSeek"]]) {
      const name = pair[0], provider = asObject(state.providers[name]), card = make("article", "provider-card");
      const heading = make("h4");
      heading.append(make("span", "", pair[1]));
      heading.append(statusPill(provider.configured || provider.logged_in ? "usable" : "unresolved"));
      card.append(heading);
      if (name === "codex") {
        card.append(make("p", "", "已安装：" + displayValue(provider.installed) + " · 已登录：" + displayValue(provider.logged_in)));
        card.append(make("p", "", "支持状态：" + displayValue(provider.supported) + (provider.version ? " · 版本 " + String(provider.version) : "")));
      } else card.append(make("p", "", "服务端凭据已配置：" + displayValue(provider.configured)));
      if (provider.message) card.append(make("p", "provider-message", provider.message));
      host.append(card);
    }
    if (state.providers.default_provider) host.append(make("p", "tiny muted full-width", "服务默认入口：" + String(state.providers.default_provider)));
  }
  async function testProvider() {
    const name = $("#provider-select").value, key = consumeApiKey(), body = { request_id: requestId() };
    if (key) body.api_key = key;
    const button = $("#test-provider-button"); button.disabled = true;
    setText("provider-test-status", "正在提交测试任务…");
    try {
      const payload = await api("/api/providers/" + encodeURIComponent(name) + "/test", { method: "POST", body: body });
      const job = asObject(payload).job || payload;
      if (job.id !== undefined) {
        const added = upsertJob(job); selectJob(added.id);
        setText("provider-test-status", "测试任务 " + String(job.id) + " 已创建，可在工作台查看。");
      } else setText("provider-test-status", "测试请求已提交。");
      showToast("入口测试任务已创建。", "success");
    } catch (error) { setText("provider-test-status", error.message); reportError(error, "提交入口测试失败"); }
    finally { button.disabled = false; }
  }
  async function loadRuntime(showErrors) {
    try { state.runtime = asObject(await api("/api/retrieval/runtime")); renderRuntime(); updateCapabilityNote(); }
    catch (error) {
      state.runtime = {};
      $("#retrieval-runtime").replaceChildren(make("p", "muted", "暂时无法读取检索设备状态。"));
      $("#workbench-runtime").replaceChildren(make("p", "muted", "暂时无法读取检索状态。"));
      if (showErrors) reportError(error, "读取检索状态失败");
    }
  }
  function flattenEntries(object, prefix, result) {
    const target = result || [];
    for (const entry of Object.entries(asObject(object))) {
      const key = prefix ? prefix + "." + entry[0] : entry[0], value = entry[1];
      if (value && typeof value === "object" && !Array.isArray(value)) flattenEntries(value, key, target);
      else target.push([key, value]);
    }
    return target;
  }
  function renderDiagnostics() {
    const host = $("#retrieval-runtime"); host.replaceChildren();
    const values = flattenEntries(state.runtime);
    if (!values.length) { host.append(make("p", "muted", "尚无检索运行信息。")); return; }
    for (const pair of values) {
      const item = make("div", "diagnostic-item");
      item.append(make("span", "", pair[0]), make("strong", "", displayValue(pair[1]))); host.append(item);
    }
  }
  function renderWorkbenchRuntime() {
    const host = $("#workbench-runtime"); host.replaceChildren();
    for (const pair of [
      ["设备", state.runtime.actual_device || state.runtime.device || state.runtime.embedding_device],
      ["精度", state.runtime.dtype || state.runtime.precision],
      ["索引", state.runtime.index_version || state.runtime.index_status]
    ]) {
      const row = make("div", "runtime-line");
      row.append(make("span", "", pair[0]), make("strong", "", displayValue(pair[1]))); host.append(row);
    }
  }
  function renderRuntime() { renderDiagnostics(); renderWorkbenchRuntime(); }
  function updateCapabilityNote() {
    const hasProfile = Boolean(state.style && state.style.active) || Boolean(state.styleProfile && state.styleProfile.active);
    const usable = state.examples.filter(function (item) { return item.content_enabled !== false && item.status !== "disabled"; }).length;
    const notes = [];
    if (!hasProfile) notes.push("当前没有启用的风格画像");
    if (!usable) notes.push("当前没有可用示例");
    if (notes.length) {
      setText("capability-note", notes.join("；") + "；你仍可选择“基础”模式生成。");
      setText("active-profile-state", "可按基础模式开始");
    } else {
      setText("capability-note", "");
      setText("active-profile-state", "画像与示例已就绪");
    }
  }
  async function rebuildIndex() {
    const button = $("#rebuild-index-button"); button.disabled = true;
    try {
      const payload = await api("/api/retrieval/rebuild", { method: "POST", body: { request_id: requestId() } });
      const job = asObject(payload).job || payload;
      if (job.id !== undefined) { const added = upsertJob(job); selectJob(added.id); showToast("索引重建任务已创建。", "success"); }
      else showToast("索引重建请求已提交。", "success");
    } catch (error) { reportError(error, "提交索引重建失败"); }
    finally { button.disabled = false; }
  }
  async function previewRetrieval() {
    const question = $("#question-input").value.trim();
    if (!question) { showToast("请先填写问题，再预览相关参考。", "error"); return; }
    const button = $("#preview-retrieval-button"); button.disabled = true;
    try {
      const payload = await api("/api/retrieval/preview", {
        method: "POST",
        body: { question: question, background: $("#background-input").value, scene: $("#scene-input").value, length: $("#length-input").value }
      });
      $("#retrieval-panel").hidden = false; renderRetrievalPreview(payload);
    } catch (error) { reportError(error, "检索预览失败"); }
    finally { button.disabled = false; }
  }
  function renderRetrievalPreview(payload) {
    const host = $("#retrieval-content"), data = asObject(payload); host.replaceChildren();
    addReferenceItems(host, "内容参考", data.content);
    addReferenceItems(host, "风格参考", data.style);
    if (data.runtime) {
      const box = make("div", "reference-group");
      box.append(make("h4", "", "检索运行状态"), make("pre", "safe-text", displayValue(data.runtime))); host.append(box);
    }
    if (asArray(data.warnings).length) {
      const list = make("ul", "warning-list");
      for (const warning of data.warnings) list.append(make("li", "", displayValue(warning)));
      host.append(list);
    }
  }
  async function saveFinal() {
    const jobId = String(state.selectedJobId || ""), editor = $("#selected-reply-draft");
    if (!jobId || !editor) { showToast("请先选择一条已生成的回复任务。", "error"); return; }
    try {
      const payload = await api("/api/feedback", {
        method: "POST",
        body: {
          job_id: jobId, final_text: editor.value,
          rating: $("#selected-rating").value ? Number($("#selected-rating").value) : null,
          reasons: $("#selected-reasons").value, adopt: false, request_id: requestId()
        }
      });
      const feedback = asObject(asObject(payload).feedback || payload);
      if (feedback.id === undefined) throw new Error("服务没有返回最终稿编号。");
      state.savedFeedback.set(jobId, { id: feedback.id, final_text: editor.value });
      state.feedbackMeta.set(jobId, { rating: $("#selected-rating").value, reasons: splitTags($("#selected-reasons").value) });
      saveSessionDraft();
      renderJobDetail(state.jobs.get(jobId)); await loadFeedback(false);
      showToast("最终稿已保存；需要入库时再单独采用。", "success");
    } catch (error) { reportError(error, "保存最终稿失败"); }
  }
  async function adoptFinal() {
    const jobId = String(state.selectedJobId || ""), editor = $("#selected-reply-draft"), saved = state.savedFeedback.get(jobId);
    if (!saved || !editor) { showToast("请先保存最终稿，再单独采用为示例。", "error"); return; }
    if (saved.final_text !== editor.value) { showToast("当前文本有修改，请先保存这版最终稿，再采用为示例。", "error"); return; }
    try {
      const payload = await api("/api/feedback/" + encodeURIComponent(String(saved.id)) + "/adopt", { method: "POST", body: {} });
      showToast(asObject(payload).message || "已提交采用请求。", "success");
      await Promise.allSettled([loadFeedback(false), loadExamples(false)]);
    } catch (error) { reportError(error, "采用最终稿失败"); }
  }
  async function loadFeedback(showErrors) {
    try {
      state.feedbackItems = asArray(asObject(await api("/api/feedback")).items);
      for (const item of state.feedbackItems) {
        if (item.job_id !== undefined && item.id !== undefined && !state.savedFeedback.has(String(item.job_id))) {
          state.savedFeedback.set(String(item.job_id), { id: item.id, final_text: item.final_text || item.text || "" });
        }
      }
      renderFeedback();
    }
    catch (error) {
      $("#feedback-list").replaceChildren(make("p", "muted empty-state", "暂时无法读取已保存稿件。"));
      if (showErrors) reportError(error, "读取最终稿失败");
    }
  }
  function renderFeedback() {
    const host = $("#feedback-list"); host.replaceChildren();
    if (!state.feedbackItems.length) { host.append(make("p", "muted empty-state", "最终稿由你主动保存。")); return; }
    for (const item of state.feedbackItems) {
      const row = make("article", "feedback-item");
      row.append(make("p", "", item.final_text || item.text || ""));
      row.append(make("small", "", (item.created_at ? jobTime(item.created_at) : "") + (item.adopted ? " · 已采用" : " · 未采用")));
      host.append(row);
    }
  }
  function splitTags(value) {
    return String(value || "").split(/[,，、]/).map(function (part) { return part.trim(); }).filter(Boolean);
  }
  function exampleMessages(example) {
    const source = asObject(example);
    const raw = asObject(source.raw);
    return asArray(source.messages || raw.messages || source.original_messages || asObject(source.unit).messages);
  }
  async function loadExamples(showErrors) {
    const params = new URLSearchParams();
    const status = $("#example-status-filter").value;
    if (status) params.set("status", status);
    params.set("offset", String(state.examplesOffset));
    params.set("limit", "50");
    try {
      const payload = await api("/api/examples?" + params.toString());
      state.examples = asArray(asObject(payload).items);
      state.examplesTotal = Number(asObject(payload).total || 0);
      renderExamples(); updateCapabilityNote();
      if (state.selectedExample && !state.examples.some(function (item) { return String(item.id) === String(state.selectedExample.id); })) {
        state.selectedExample = null; renderExampleDetail(null);
      }
      return payload;
    } catch (error) {
      $("#example-list").replaceChildren(make("p", "muted empty-state", "暂时无法读取示例列表。"));
      if (showErrors) reportError(error, "读取示例列表失败");
      return null;
    }
  }
  function renderExamples() {
    const host = $("#example-list"); host.replaceChildren();
    setText("example-count", "共 " + state.examplesTotal + " 条");
    setText("examples-page", "第 " + (Math.floor(state.examplesOffset / 50) + 1) + " 页");
    $("#examples-prev").disabled = state.examplesOffset <= 0;
    $("#examples-next").disabled = state.examplesOffset + 50 >= state.examplesTotal;
    if (!state.examples.length) { host.append(make("p", "muted empty-state", "这个筛选条件下还没有示例。")); return; }
    for (const item of state.examples) {
      const button = make("button", "example-item" + (state.selectedExample && String(state.selectedExample.id) === String(item.id) ? " active" : ""));
      button.type = "button"; button.dataset.exampleId = String(item.id);
      const heading = make("div", "example-item-heading");
      heading.append(make("span", "example-item-title", "示例 " + String(item.id)));
      heading.append(statusPill(item.status || "usable"));
      const messages = exampleMessages(item);
      const preview = item.question_text || (messages[0] && messages[0].content) || item.answer_text || "暂无摘要";
      button.append(heading, make("span", "example-item-preview", preview));
      const tags = make("span", "example-tags");
      tags.textContent = (item.scene_tags || []).join(" · ") || item.source_type || "";
      button.append(tags); host.append(button);
    }
  }
  async function openExample(id) {
    try {
      const payload = await api("/api/examples/" + encodeURIComponent(String(id)));
      state.selectedExample = asObject(payload).example || payload;
      renderExamples(); renderExampleDetail(state.selectedExample);
    } catch (error) { reportError(error, "读取示例失败"); }
  }
  function renderRawMessages(messages) {
    const list = make("div", "raw-message-list");
    if (!messages.length) { list.append(make("p", "muted tiny", "当前响应未包含原始消息。")); return list; }
    messages.forEach(function (message, index) {
      const m = asObject(message), card = make("article", "raw-message");
      const heading = make("div", "raw-message-header");
      heading.append(make("span", "", "原始消息 " + (index + 1)));
      heading.append(make("span", "", m.target ? "目标回复" : (m.context_only ? "仅作上下文" : "消息")));
      card.append(heading, make("div", "raw-message-content", m.content || ""));
      const meta = make("div", "raw-meta");
      for (const key of ["id", "comment_id", "note_id", "context_only", "target"]) {
        if (m[key] !== undefined && m[key] !== null && m[key] !== "") meta.append(make("span", "", key + ": " + displayValue(m[key])));
      }
      if (meta.childNodes.length) card.append(meta);
      list.append(card);
    });
    return list;
  }
  function inputField(labelText, id, value, type) {
    const label = make("label", "field-label"), caption = make("span", "", labelText);
    const input = make(type === "textarea" ? "textarea" : "input");
    input.id = id;
    if (type === "textarea") input.rows = 3;
    else input.type = type || "text";
    input.value = value === undefined || value === null ? "" : String(value);
    label.append(caption, input);
    return label;
  }
  function renderExampleDetail(example) {
    state.exampleEditorDirty = false;
    const host = $("#example-detail"); host.replaceChildren();
    if (!example) {
      const empty = make("div", "empty-state");
      empty.append(make("span", "empty-orbit", "▤"), make("p", "", "选择一个示例查看原文和整理状态。"));
      host.append(empty); setText("example-version", ""); return;
    }
    setText("example-version", "版本 " + displayValue(example.version));
    host.append(make("h4", "job-detail-title", "示例 " + String(example.id)));
    host.append(renderRawMessages(exampleMessages(example)));
    const form = make("div", "detail-form");
    form.append(inputField("问题文本", "edit-example-question", example.question_text || "", "textarea"));
    form.append(inputField("补充上下文", "edit-example-context", example.context_text || "", "textarea"));
    form.append(inputField("回复正文", "edit-example-answer", example.answer_text || "", "textarea"));
    form.append(inputField("资料状态", "edit-example-status", example.status || "usable", "text"));
    const checks = make("div", "check-row");
    for (const pair of [["edit-example-content-enabled", "用于内容检索", example.content_enabled !== false], ["edit-example-style-enabled", "用于风格参考", example.style_enabled !== false]]) {
      const label = make("label", "check-label"), check = make("input");
      check.type = "checkbox"; check.id = pair[0]; check.checked = Boolean(pair[2]);
      label.append(check, document.createTextNode(pair[1])); checks.append(label);
    }
    form.append(checks);
    form.append(inputField("主题标签（逗号分隔）", "edit-example-topic-tags", asArray(example.topic_tags).join(", "), "text"));
    form.append(inputField("场景标签（逗号分隔）", "edit-example-scene-tags", asArray(example.scene_tags).join(", "), "text"));
    const actions = make("div", "detail-actions");
    const save = make("button", "button button-primary", "保存示例编辑");
    save.type = "button"; save.dataset.action = "save-example";
    actions.append(save);
    const needsReview = ["needs_review", "unresolved"].includes(example.status);
    if (needsReview) {
      const manual = make("button", "button button-secondary", "人工整理");
      manual.type = "button"; manual.dataset.action = "manual-organize";
      const model = make("button", "button button-quiet", "模型辅助建议");
      model.type = "button"; model.dataset.action = "model-organize";
      actions.append(manual, model);
    }
    form.append(actions); host.append(form);
    if (needsReview || state.organizeSuggestions && state.organizeSuggestions.exampleId === String(example.id)) {
      const panel = make("section", "organize-panel"); panel.id = "organize-panel";
      panel.append(make("h4", "", "整理分组"));
      panel.append(make("p", "tiny muted", "可创建多组问题、上下文和回答。模型结果只作建议，确认后才会保存。"));
      const groups = make("div", "unit-groups"); groups.id = "groups-editor"; panel.append(groups);
      const groupActions = make("div", "organize-actions");
      const add = make("button", "button button-quiet", "＋ 添加分组"); add.type = "button"; add.dataset.action = "add-group";
      const confirm = make("button", "button button-primary", "确认分组"); confirm.type = "button"; confirm.dataset.action = "confirm-groups";
      groupActions.append(add, confirm); panel.append(groupActions); host.append(panel);
      if (state.organizeSuggestions && state.organizeSuggestions.exampleId === String(example.id)) {
        fillGroups(groups, exampleMessages(example), state.organizeSuggestions.units);
        panel.prepend(make("div", "inline-note", "模型建议已载入。请检查每个选择，再点击“确认分组”。"));
      }
    }
  }
  function fillGroups(host, messages, units) {
    host.replaceChildren();
    const groups = asArray(units);
    if (!groups.length) createGroup(host, messages, null, 0);
    else groups.forEach(function (unit, index) { createGroup(host, messages, unit, index); });
  }
  function createGroup(host, messages, initial, index) {
    const unit = asObject(initial), card = make("div", "unit-group");
    const heading = make("div", "unit-group-heading");
    heading.append(make("span", "", "回复单元 " + (index + 1)));
    const remove = make("button", "text-button danger-text", "移除");
    remove.type = "button"; remove.dataset.action = "remove-group"; heading.append(remove); card.append(heading);
    const selects = make("div", "unit-selects");
    const fields = [
      ["question_ids", "问题"], ["context_ids", "上下文"], ["answer_ids", "回答"]
    ];
    for (const field of fields) {
      const label = make("label", "field-label", field[1]);
      const select = make("select");
      select.multiple = true; select.dataset.role = field[0]; select.setAttribute("aria-label", field[1] + "消息");
      const chosen = asArray(unit[field[0]]).map(String);
      messages.forEach(function (message, i) {
        const m = asObject(message), option = make("option");
        const id = m.id !== undefined && m.id !== null ? m.id : ("message-" + i);
        option.value = String(id);
        option.textContent = String(id) + " · " + String(m.content || "").slice(0, 76);
        option.selected = chosen.includes(String(id)); select.append(option);
      });
      label.append(select); selects.append(label);
    }
    card.append(selects); host.append(card);
  }
  function selectedIds(select, messages) {
    return Array.from(select.selectedOptions).map(function (option) {
      const found = messages.find(function (message, index) {
        const id = asObject(message).id;
        return String(id === undefined ? "message-" + index : id) === option.value;
      });
      return found && asObject(found).id !== undefined ? asObject(found).id : option.value;
    });
  }
  async function confirmGroups() {
    const example = state.selectedExample;
    if (!example) return;
    const messages = exampleMessages(example), groups = [];
    for (const card of $$(".unit-group", $("#groups-editor"))) {
      const unit = {};
      for (const field of ["question_ids", "context_ids", "answer_ids"]) {
        const select = $("select[data-role='" + field + "']", card);
        const apiKey = field === "question_ids" ? "question_ids" : (field === "context_ids" ? "context_ids" : "answer_ids");
        unit[apiKey] = selectedIds(select, messages);
      }
      if (unit.question_ids.length || unit.answer_ids.length || unit.context_ids.length) groups.push(unit);
    }
    if (!groups.length || groups.some(function (unit) { return !unit.question_ids.length || !unit.answer_ids.length; })) {
      showToast("每个回复单元都需要至少一条问题和一条回答。", "error"); return;
    }
    try {
      await api("/api/examples/" + encodeURIComponent(String(example.id)) + "/resolve", {
        method: "POST", body: { expected_version: example.version, units: groups }
      });
      state.organizeSuggestions = null;
      showToast("分组已确认并保存。", "success");
      await Promise.allSettled([loadExamples(false), openExample(example.id)]);
    } catch (error) { reportError(error, "确认整理失败"); }
  }
  async function saveExample() {
    const example = state.selectedExample;
    if (!example) return;
    const body = {
      expected_version: example.version,
      question_text: $("#edit-example-question").value,
      context_text: $("#edit-example-context").value,
      answer_text: $("#edit-example-answer").value,
      content_enabled: $("#edit-example-content-enabled").checked,
      style_enabled: $("#edit-example-style-enabled").checked,
      status: $("#edit-example-status").value.trim(),
      topic_tags: splitTags($("#edit-example-topic-tags").value),
      scene_tags: splitTags($("#edit-example-scene-tags").value)
    };
    try {
      const payload = await api("/api/examples/" + encodeURIComponent(String(example.id)), { method: "PATCH", body: body });
      state.selectedExample = asObject(payload).example || payload;
      renderExampleDetail(state.selectedExample); await loadExamples(false);
      showToast("示例已更新。", "success");
    } catch (error) { reportError(error, "保存示例失败"); }
  }
  async function organizeExample() {
    const example = state.selectedExample;
    if (!example) return;
    try {
      const payload = await writeWithTemporaryKey("/api/examples/" + encodeURIComponent(String(example.id)) + "/organize", {});
      const job = asObject(payload).job || payload;
      if (job.id === undefined) throw new Error("服务没有返回整理任务编号。");
      const added = upsertJob(job); selectJob(added.id);
      showToast("整理任务已创建；结果会作为待确认建议显示。", "success");
    } catch (error) { reportError(error, "提交整理任务失败"); }
  }
  async function previewImport() {
    const file = $("#import-file").files[0];
    if (!file) { showToast("请先选择一个 Q&A JSON 文件。", "error"); return; }
    const button = $("#preview-import-button"); button.disabled = true;
    const form = new FormData(); form.append("file", file, file.name);
    try {
      const payload = await api("/api/imports/preview", { method: "POST", formData: form });
      state.importFile = file; state.importPreview = asObject(payload); renderImportPreview();
      showToast("文件预览已就绪，请核对数量和问题。", "success");
    } catch (error) { reportError(error, "预览导入失败"); }
    finally { button.disabled = false; }
  }
  function renderImportPreview() {
    const host = $("#import-preview"); host.replaceChildren(); host.hidden = false;
    const data = state.importPreview || {};
    host.append(make("p", "tiny muted", "文件哈希：" + displayValue(data.file_hash) + " · 预览编号：" + displayValue(data.id)));
    const stats = asObject(data.stats), grid = make("div", "stat-grid");
    const statLabels = { groups: "原始组", target_comments: "目标评论", ready: "可用单元", needs_review: "待整理单元", excluded: "排除单元", duplicates: "重复单元", conflicts: "冲突单元" };
    for (const entry of Object.entries(stats)) {
      if (!statLabels[entry[0]]) continue;
      const card = make("div", "stat-card");
      card.append(make("span", "", statLabels[entry[0]]), make("strong", "", displayValue(entry[1]))); grid.append(card);
    }
    if (grid.childNodes.length) host.append(grid);
    if (data.metadata) host.append(make("pre", "safe-text", displayValue(data.metadata)));
    const warnings = asArray(data.warnings);
    if (warnings.length) {
      const list = make("ul", "warning-list");
      for (const warning of warnings) list.append(make("li", "", displayValue(warning)));
      host.append(list);
    }
    const units = asArray(data.units), preview = make("div", "stack");
    for (const unit of units.slice(0, 6)) {
      const card = make("article", "preview-unit");
      card.append(make("h4", "", "待导入单元 · " + statusLabel(unit.status)));
      card.append(make("p", "tiny", "问题：" + (unit.question_text || "未确定，请整理")));
      if (unit.context_text) card.append(make("p", "tiny muted", "上下文：" + unit.context_text));
      card.append(make("p", "tiny", "回答：" + (unit.answer_text || "未确定，请整理")));
      for (const reason of asArray(unit.review_reasons)) card.append(make("p", "tiny muted", reason));
      card.append(renderRawMessages(exampleMessages(unit)));
      preview.append(card);
    }
    if (units.length > 6) preview.append(make("p", "tiny muted", "另有 " + (units.length - 6) + " 个单元未展开。"));
    host.append(preview);
    const actions = make("div", "preview-actions");
    actions.append(make("span", "tiny muted", "确认后才会写入本机资料库。"));
    const confirm = make("button", "button button-primary", "确认导入");
    confirm.type = "button"; confirm.dataset.action = "confirm-import"; actions.append(confirm); host.append(actions);
  }
  async function confirmImport() {
    if (!state.importPreview || !state.importPreview.id) return;
    try {
      const payload = await api("/api/imports/" + encodeURIComponent(String(state.importPreview.id)) + "/confirm", { method: "POST", body: {} });
      const result = asObject(payload);
      const summary = "导入 " + displayValue(result.imported) + " 条；重复 " + displayValue(result.duplicates) + " 条；冲突 " + displayValue(result.conflicts) + " 条。";
      showToast(summary, "success"); state.importPreview = null; $("#import-preview").hidden = true;
      await loadExamples(false);
    } catch (error) { reportError(error, "确认导入失败"); }
  }
  async function loadStyle(showErrors) {
    try {
      state.style = asObject(await api("/api/style"));
      const active = state.style.active;
      state.styleProfile = active && typeof active === "object" ? active : null;
      if (state.styleProfile) renderStyleProfile(state.styleProfile);
      else {
        state.styleRules = []; setText("style-version-label", "暂无启用画像");
        setText("style-strategy", "尚无策略摘要。"); setText("style-expression", "尚无表达摘要。");
        setText("style-gaps", "暂无记录。"); setText("style-statistics", "尚无统计。");
        $("#style-prompt").value = ""; renderRules();
      }
      renderStyleVersions(); renderStyleAlerts(); loadStyleJobActions(); updateCapabilityNote();
    } catch (error) {
      if (showErrors) reportError(error, "读取风格画像失败");
      else $("#style-rules").replaceChildren(make("p", "muted empty-state", "暂时无法读取风格画像。"));
    }
  }
  function renderStyleProfile(profile) {
    state.styleEditorDirty = false;
    state.styleProfile = profile;
    state.styleRules = cloneValue(asArray(profile.rules));
    setText("style-version-label", "版本 " + displayValue(profile.version) + (profile.active ? " · 启用中" : ""));
    renderAnalysis("style-strategy", asObject(profile.strategy), ["observations", "content_candidates", "contradictions", "boundaries"]);
    renderAnalysis("style-expression", asObject(profile.expression), ["dimensions", "cross_checks"]);
    setText("style-gaps", asArray(profile.gaps).join("\n") || "暂无记录。");
    const stats = asObject(profile.statistics);
    setText("style-statistics", stats.sample_count === undefined ? "尚无统计。" :
      "样本 " + stats.sample_count + " 条；回答共 " + stats.answer_chars + " 字。\n回答长度中位数 " + stats.length_median +
      " 字；范围 " + stats.length_min + "–" + stats.length_max + " 字。\n" + (stats.scope || ""));
    $("#style-prompt").value = profile.prompt || "";
    renderRules();
  }
  function renderAnalysis(id, analysis, fields) {
    const host = document.getElementById(id); host.replaceChildren();
    const labels = { observations: "策略观察", content_candidates: "内容候选", contradictions: "场景差异", boundaries: "边界", dimensions: "表达观察", cross_checks: "两轮核对" };
    for (const field of fields) {
      const items = asArray(analysis[field]);
      if (!items.length) continue;
      host.append(make("h4", "", labels[field]));
      for (const item of items) {
        if (typeof item === "string") { host.append(make("p", "tiny", item)); continue; }
        const card = make("article", "reference-group"), observation = asObject(item);
        card.append(make("p", "tiny", observation.description));
        if (asArray(observation.applicable_scenes).length) card.append(make("p", "tiny muted", "适用：" + observation.applicable_scenes.join("、")));
        for (const evidence of asArray(observation.evidence)) card.append(make("blockquote", "tiny", evidence.quote));
        evidenceButtons(card, observation.evidence_ids, "原文"); host.append(card);
      }
    }
    if (!host.childNodes.length) host.append(make("p", "tiny", "暂无记录。"));
  }
  function renderStyleAlerts() {
    const host = $("#style-alerts"); host.replaceChildren();
    if (state.styleDraftResult && Array.isArray(state.styleDraftResult.rules)) {
      const row = make("div", "inline-note");
      row.append(document.createTextNode("有一份风格提取初稿待检查。"));
      const apply = make("button", "text-button", "载入初稿");
      apply.type = "button"; apply.dataset.action = "apply-style-draft"; row.append(apply); host.append(row);
    }
    if (state.style && state.style.stale) {
      const stale = typeof state.style.stale === "string" ? state.style.stale : displayValue(state.style.stale);
      host.append(make("div", "inline-note", "资料更新后画像可能需要复核：" + stale));
    }
    if (state.styleDraftResult && !Array.isArray(state.styleDraftResult.rules)) {
      host.append(make("div", "inline-note", "最近一次提取结果已完成，请在任务列表中查看分析结果。"));
    }
  }
  function renderStyleVersions() {
    const host = $("#style-versions"); host.replaceChildren();
    const versions = asArray(state.style && state.style.versions);
    if (!versions.length) { host.append(make("p", "muted tiny", "尚无历史版本。")); return; }
    versions.forEach(function (version, index) {
      const item = asObject(version), row = make("div", "version-item" + (state.styleProfile && String(item.id) === String(state.styleProfile.id) ? " active" : ""));
      const pick = make("button", "version-item");
      pick.type = "button"; pick.dataset.action = "select-style-version"; pick.dataset.versionIndex = String(index);
      const details = make("span");
      details.append(make("strong", "", "版本 " + displayValue(item.version)));
      details.append(make("small", "", (item.active ? "启用中 · " : "") + (item.created_at ? jobTime(item.created_at) : String(item.id || ""))));
      pick.append(details);
      const exportButton = make("button", "text-button", "导出");
      exportButton.type = "button"; exportButton.dataset.action = "export-style-version";
      exportButton.dataset.versionId = String(item.id || "");
      const compare = make("button", "text-button", "比较");
      compare.type = "button"; compare.dataset.action = "compare-style-version"; compare.dataset.versionIndex = String(index);
      row.append(pick, compare, exportButton); host.append(row);
    });
  }
  function compareStyleVersion(index) {
    const target = asObject(asArray(state.style && state.style.versions)[Number(index)]), current = state.styleProfile;
    if (!target.id || !current) { showToast("请先载入一个风格版本。", "error"); return; }
    const host = $("#style-comparison"); host.replaceChildren(); host.hidden = false;
    host.append(make("h4", "", "当前版本 " + current.version + " → 版本 " + target.version));
    const before = new Map(asArray(current.rules).map(function (rule) { return [rule.rule_id, rule]; }));
    const after = new Map(asArray(target.rules).map(function (rule) { return [rule.rule_id, rule]; }));
    let changes = 0;
    for (const id of new Set([...before.keys(), ...after.keys()])) {
      const left = before.get(id), right = after.get(id);
      if (JSON.stringify(left) === JSON.stringify(right)) continue;
      const card = make("article", "reference-group");
      card.append(make("strong", "tiny", id + " · " + (!left ? "新增" : !right ? "删除" : "修改")));
      if (left) card.append(make("p", "tiny", "当前：" + left.description + (left.enabled ? "（启用）" : "（停用）")));
      if (right) card.append(make("p", "tiny", "目标：" + right.description + (right.enabled ? "（启用）" : "（停用）")));
      if (left && right) card.append(make("p", "tiny muted", "比较包含适用场景、证据、强度、来源和用途开关。"));
      if (left && right) {
        const labels = { dimension: "维度", applicable_scenes: "场景", evidence_ids: "证据", counterexample_ids: "反例", strength: "强度", origin: "来源" };
        for (const field of Object.keys(labels)) {
          if (JSON.stringify(left[field]) !== JSON.stringify(right[field])) card.append(make("p", "tiny", labels[field] + "：" + displayValue(left[field]) + " → " + displayValue(right[field])));
        }
      }
      host.append(card); changes += 1;
    }
    if (!changes) host.append(make("p", "tiny", "规则一致。"));
  }
  function evidenceButtons(host, ids, label) {
    const row = make("div", "evidence-row");
    row.append(make("span", "", label + "："));
    if (!asArray(ids).length) row.append(make("span", "muted", "无"));
    for (const id of asArray(ids)) {
      const button = make("button", "reference-link", String(id));
      button.type = "button"; button.dataset.action = "open-example"; button.dataset.exampleId = String(id); row.append(button);
    }
    host.append(row);
  }
  function renderRules() {
    const host = $("#style-rules"); host.replaceChildren();
    if (!state.styleRules.length) { host.append(make("p", "muted empty-state", "暂无规则。")); return; }
    state.styleRules.forEach(function (ruleValue, index) {
      const rule = asObject(ruleValue), card = make("article", "rule-card");
      const heading = make("div", "rule-card-header");
      heading.append(make("strong", "", "规则 " + (index + 1) + (rule.enabled === false ? " · 已停用" : "")));
      const remove = make("button", "text-button danger-text", "删除");
      remove.type = "button"; remove.dataset.action = "remove-rule"; remove.dataset.ruleIndex = String(index);
      heading.append(remove); card.append(heading);
      const fields = make("div", "rule-fields");
      fields.append(ruleInput(index, "rule_id", "规则编号", rule.rule_id || ""));
      fields.append(ruleInput(index, "dimension", "分析维度", rule.dimension || ""));
      fields.append(ruleTextarea(index, "description", "规则描述", rule.description || ""));
      fields.append(ruleInput(index, "applicable_scenes", "适用场景（逗号分隔）", asArray(rule.applicable_scenes).join(", ")));
      fields.append(ruleInput(index, "evidence_ids", "证据 ID（逗号分隔）", asArray(rule.evidence_ids).join(", ")));
      fields.append(ruleInput(index, "counterexample_ids", "反例 ID（逗号分隔）", asArray(rule.counterexample_ids).join(", ")));
      const strength = make("label", "field-label"); strength.append(make("span", "", "规则强度"));
      const select = make("select"); select.dataset.ruleIndex = String(index); select.dataset.ruleField = "strength";
      for (const value of ["核心倾向", "场景倾向", "待核实"]) {
        const option = make("option", "", value); option.value = value; option.selected = rule.strength === value; select.append(option);
      }
      strength.append(select); fields.append(strength);
      const origin = make("label", "field-label"); origin.append(make("span", "", "规则来源"));
      const originSelect = make("select"); originSelect.dataset.ruleIndex = String(index); originSelect.dataset.ruleField = "origin";
      for (const pair of [["observed", "从样本观察"], ["user_defined", "用户设定"]]) {
        const option = make("option", "", pair[1]); option.value = pair[0]; option.selected = (rule.origin || "observed") === pair[0]; originSelect.append(option);
      }
      origin.append(originSelect); fields.append(origin);
      const enabledLabel = make("label", "check-label");
      const enabled = make("input"); enabled.type = "checkbox"; enabled.checked = rule.enabled !== false;
      enabled.dataset.ruleIndex = String(index); enabled.dataset.ruleField = "enabled";
      enabledLabel.append(enabled, document.createTextNode("启用此规则")); fields.append(enabledLabel);
      card.append(fields);
      evidenceButtons(card, rule.evidence_ids, "证据");
      evidenceButtons(card, rule.counterexample_ids, "反例");
      host.append(card);
    });
  }
  function ruleInput(index, field, labelText, value) {
    const label = make("label", "field-label");
    label.append(make("span", "", labelText));
    const input = make("input"); input.type = "text"; input.value = value;
    input.dataset.ruleIndex = String(index); input.dataset.ruleField = field;
    label.append(input); return label;
  }
  function ruleTextarea(index, field, labelText, value) {
    const label = make("label", "field-label full-width");
    label.append(make("span", "", labelText));
    const input = make("textarea"); input.rows = 3; input.value = value;
    input.dataset.ruleIndex = String(index); input.dataset.ruleField = field;
    label.append(input); return label;
  }
  function updateRuleFromInput(event) {
    const field = event.target.dataset.ruleField, index = Number(event.target.dataset.ruleIndex);
    if (!field || !Number.isInteger(index) || !state.styleRules[index]) return;
    state.styleEditorDirty = true;
    if (field === "applicable_scenes" || field === "evidence_ids" || field === "counterexample_ids") {
      state.styleRules[index][field] = splitTags(event.target.value);
    } else if (field === "enabled") state.styleRules[index][field] = event.target.checked;
    else state.styleRules[index][field] = event.target.value;
  }
  function addRule() {
    state.styleEditorDirty = true;
    state.styleRules.push({
      rule_id: "user-" + requestId().slice(0, 8), dimension: "", description: "",
      applicable_scenes: [], evidence_ids: [], counterexample_ids: [],
      strength: "待核实", origin: "user_defined", enabled: true
    });
    renderRules();
  }
  async function saveStyleVersion() {
    const profile = state.styleProfile || {};
    try {
      const payload = await api("/api/style/versions", {
        method: "POST",
        body: {
          base_id: profile.id || null,
          expected_version: profile.version === undefined ? null : profile.version,
          rules: cloneValue(state.styleRules),
          activate: $("#activate-style-checkbox").checked
        }
      });
      const saved = asObject(asObject(payload).profile || payload);
      state.styleProfile = saved;
      state.styleDraftResult = null;
      state.styleEditorDirty = false;
      if (state.style) {
        state.style.active = saved.active ? saved : state.style.active;
        const versions = asArray(state.style.versions).slice(); versions.unshift(saved); state.style.versions = versions;
      }
      await loadStyle(false);
      showToast("风格规则已保存为新版本。", "success");
    } catch (error) { reportError(error, "保存风格版本失败"); }
  }
  async function startStyleExtract() {
    const button = $("#extract-style-button"); button.disabled = true;
    try {
      const payload = await writeWithTemporaryKey("/api/style/extract", {});
      const job = asObject(payload).job || payload;
      if (job.id === undefined) throw new Error("服务没有返回风格提取任务编号。");
      const added = upsertJob(job); selectJob(added.id);
      showToast("风格提取任务已创建。", "success");
    } catch (error) { reportError(error, "提交风格提取失败"); }
    finally { button.disabled = false; }
  }
  function loadStyleJobActions() {
    const host = $("#style-job-actions"); host.replaceChildren();
    const jobs = Array.from(state.jobs.values()).filter(function (job) {
      return String(job.kind || "").includes("style") && ["failed", "interrupted", "cancelled"].includes(job.status);
    });
    if (!jobs.length) { host.append(make("p", "tiny muted", "当前没有可继续的风格任务。")); return; }
    for (const job of jobs) {
      const button = make("button", "button button-secondary", "继续任务 " + String(job.id));
      button.type = "button"; button.dataset.action = "resume-style"; button.dataset.jobId = String(job.id); host.append(button);
    }
  }
  async function resumeStyleJob(id) {
    try {
      const key = consumeApiKey(), body = { request_id: requestId() };
      if (key) body.api_key = key;
      const payload = await api("/api/style/jobs/" + encodeURIComponent(String(id)) + "/resume", { method: "POST", body: body });
      const job = asObject(payload).job || payload;
      if (job.id !== undefined) { const added = upsertJob(job); selectJob(added.id); }
      showToast("已提交继续请求。", "success");
    } catch (error) { reportError(error, "继续风格任务失败"); }
  }
  async function selectStyleVersion(index) {
    const version = asObject(asArray(state.style && state.style.versions)[Number(index)]);
    if (!version.id) return;
    if (!Array.isArray(version.rules)) {
      showToast("此版本仅返回列表摘要，当前接口没有提供读取历史规则的路径。", "error"); return;
    }
    renderStyleProfile(version); renderStyleVersions();
  }
  function exportStyleVersion(id) {
    if (!id) { showToast("版本编号缺失，无法导出。", "error"); return; }
    const anchor = make("a");
    anchor.href = "/api/style/versions/" + encodeURIComponent(String(id)) + "/export";
    anchor.download = ""; document.body.append(anchor); anchor.click(); anchor.remove();
  }
  function handleCompletedJob(job) {
    const kind = String(job.kind || ""), result = asObject(job.result);
    if (kind.includes("style")) {
      const profile = asObject(result.profile || result);
      if (Array.isArray(profile.rules)) {
        state.styleDraftResult = profile;
        if (state.styleEditorDirty) {
          renderStyleAlerts();
          showToast("风格提取已完成；为保留你正在编辑的规则，初稿已暂存，请点击“载入初稿”查看。", "success");
        } else {
          renderStyleProfile(profile); renderStyleAlerts();
          showToast("风格提取已完成，初稿已载入；检查规则后再保存版本。", "success");
        }
      } else {
        state.styleDraftResult = result; showToast("风格提取已完成，请查看任务结果并核对分析内容。", "success");
      }
      renderStyleAlerts(); loadStyleJobActions();
    }
    if (kind.includes("organize")) {
      const input = asObject(job.input);
      const exampleId = result.example_id || input.example_id || asObject(input.example).id;
      if (exampleId !== undefined && exampleId !== null && Array.isArray(result.units)) {
        state.organizeSuggestions = { exampleId: String(exampleId), units: cloneValue(result.units), jobId: String(job.id) };
        if (state.selectedExample && String(state.selectedExample.id) === String(exampleId) && !state.exampleEditorDirty) renderExampleDetail(state.selectedExample);
        showToast(state.exampleEditorDirty ? "整理建议已就绪；为保留当前示例编辑，建议会在你重新打开示例时显示。" : "整理建议已生成，请逐组检查并点击确认后才会写入。", "success");
      }
    }
    if (kind.includes("style")) loadStyleJobActions();
  }
  async function backupData() {
    const button = $("#create-backup-button"); button.disabled = true;
    try {
      const payload = asObject(await api("/api/backups", { method: "POST", body: {} }));
      if (!payload.url) throw new Error("备份服务没有返回下载地址。");
      const url = new URL(String(payload.url), window.location.href);
      if (url.origin !== window.location.origin) throw new Error("备份地址不属于当前本机服务。");
      const anchor = make("a"); anchor.href = url.href; anchor.download = "";
      document.body.append(anchor); anchor.click(); anchor.remove();
      showToast("备份 " + displayValue(payload.id) + " 已开始下载。", "success");
    } catch (error) { reportError(error, "创建备份失败"); }
    finally { button.disabled = false; }
  }
  async function restoreData() {
    const file = $("#restore-file").files[0];
    if (!file) { showToast("请先选择一个备份文件。", "error"); return; }
    if (!window.confirm("恢复会替换当前本机资料。服务会先保留自动备份；请确认文件来源可信后继续。")) return;
    const form = new FormData(); form.append("file", file, file.name);
    const button = $("#restore-backup-button"); button.disabled = true;
    try {
      const result = asObject(await api("/api/restores", { method: "POST", formData: form }));
      state.jobs.clear(); state.selectedJobId = null; state.savedFeedback.clear(); state.feedbackMeta.clear();
      state.selectedExample = null; state.organizeSuggestions = null; state.styleDraftResult = null;
      state.completionHandled.clear(); renderExampleDetail(null); renderJobDetail(null);
      showToast("恢复已完成；恢复前的自动备份编号：" + (asObject(result.backup).id || "请查看备份目录"), "success");
      await Promise.allSettled([loadSettings(false), loadProviders(false), loadRuntime(false), loadStyle(false), loadExamples(false), loadFeedback(false), loadJobs(false)]);
    } catch (error) { reportError(error, "恢复备份失败"); }
    finally { button.disabled = false; }
  }
  function onGlobalAction(event) {
    const button = event.target.closest("[data-action]");
    if (!button) return;
    const action = button.dataset.action;
    if (action === "open-example") { navigate("library"); openExample(button.dataset.exampleId); }
    if (action === "confirm-import") confirmImport();
    if (action === "add-rule") addRule();
    if (action === "remove-rule") {
      const index = Number(button.dataset.ruleIndex);
      if (Number.isInteger(index)) { state.styleEditorDirty = true; state.styleRules.splice(index, 1); renderRules(); }
    }
    if (action === "save-style") saveStyleVersion();
    if (action === "apply-style-draft" && state.styleDraftResult) renderStyleProfile(state.styleDraftResult);
    if (action === "resume-style") resumeStyleJob(button.dataset.jobId);
    if (action === "select-style-version") selectStyleVersion(button.dataset.versionIndex);
    if (action === "export-style-version") exportStyleVersion(button.dataset.versionId);
    if (action === "compare-style-version") compareStyleVersion(button.dataset.versionIndex);
  }
  function onJobAction(event) {
    const button = event.target.closest("[data-action]");
    if (!button) return;
    const actions = {
      "cancel-job": function () { cancelJob(button.dataset.jobId); },
      "copy-draft": copyDraft,
      "save-final": saveFinal,
      "adopt-final": adoptFinal
    };
    if (actions[button.dataset.action]) actions[button.dataset.action]();
  }
  function onExampleAction(event) {
    const button = event.target.closest("[data-action]");
    if (!button) return;
    const actions = {
      "save-example": saveExample,
      "model-organize": organizeExample,
      "manual-organize": function () {
        fillGroups($("#groups-editor"), exampleMessages(state.selectedExample), []);
      },
      "add-group": function () {
        const host = $("#groups-editor");
        createGroup(host, exampleMessages(state.selectedExample), null, host.children.length);
      },
      "remove-group": function () { button.closest(".unit-group").remove(); },
      "confirm-groups": confirmGroups
    };
    if (actions[button.dataset.action]) actions[button.dataset.action]();
  }
  function initialize() {
    loadSessionDraft();
    $("#question-input").addEventListener("input", saveSessionDraft);
    $("#background-input").addEventListener("input", saveSessionDraft);
    $("#instructions-input").addEventListener("input", saveSessionDraft);
    $("#length-input").addEventListener("change", saveSessionDraft);
    $("#scene-input").addEventListener("change", saveSessionDraft);
    $("#provider-select").addEventListener("change", function () { state.providerTouched = true; });
    for (const button of $$(".mode-button")) button.addEventListener("click", function () { setMode(button.dataset.mode); });
    for (const button of $$(".nav-link")) button.addEventListener("click", function () { navigate(button.dataset.page); });
    for (const button of $$("[data-navigate]")) button.addEventListener("click", function () { navigate(button.dataset.navigate); });
    for (const button of $$("[data-close]")) button.addEventListener("click", function () {
      const panel = document.getElementById(button.dataset.close); if (panel) panel.hidden = true;
    });
    $("#generate-button").addEventListener("click", createGeneration);
    $("#preview-retrieval-button").addEventListener("click", previewRetrieval);
    $("#refresh-jobs-button").addEventListener("click", function () { loadJobs(true); });
    $("#job-list").addEventListener("click", function (event) {
      const button = event.target.closest("[data-job-id]"); if (button) selectJob(button.dataset.jobId);
    });
    $("#job-detail").addEventListener("click", onJobAction);
    $("#example-list").addEventListener("click", function (event) {
      const button = event.target.closest("[data-example-id]"); if (button) openExample(button.dataset.exampleId);
    });
    $("#example-detail").addEventListener("click", onExampleAction);
    $("#example-detail").addEventListener("input", function () { state.exampleEditorDirty = true; });
    $("#example-status-filter").addEventListener("change", function () { state.examplesOffset = 0; loadExamples(true); });
    $("#refresh-examples-button").addEventListener("click", function () { loadExamples(true); });
    $("#examples-prev").addEventListener("click", function () { state.examplesOffset = Math.max(0, state.examplesOffset - 50); loadExamples(false); });
    $("#examples-next").addEventListener("click", function () { state.examplesOffset += 50; loadExamples(false); });
    $("#import-file").addEventListener("change", function () {
      const file = this.files[0]; setText("selected-file-name", file ? file.name + " · " + file.size.toLocaleString() + " 字节" : "尚未选择文件");
    });
    $("#preview-import-button").addEventListener("click", previewImport);
    $("#style-rules").addEventListener("input", updateRuleFromInput);
    $("#style-rules").addEventListener("change", function (event) {
      updateRuleFromInput(event);
      if (["evidence_ids", "counterexample_ids"].includes(event.target.dataset.ruleField)) renderRules();
    });
    $("#add-rule-button").addEventListener("click", addRule);
    $("#save-style-button").addEventListener("click", saveStyleVersion);
    $("#extract-style-button").addEventListener("click", startStyleExtract);
    $("#refresh-providers-button").addEventListener("click", function () { loadProviders(true); });
    $("#test-provider-button").addEventListener("click", testProvider);
    $("#save-settings-button").addEventListener("click", saveSettings);
    $("#restore-settings-button").addEventListener("click", restoreDefaults);
    $("#refresh-runtime-button").addEventListener("click", function () { loadRuntime(true); });
    $("#rebuild-index-button").addEventListener("click", rebuildIndex);
    $("#create-backup-button").addEventListener("click", backupData);
    $("#restore-backup-button").addEventListener("click", restoreData);
    $("#refresh-feedback-button").addEventListener("click", function () { loadFeedback(true); });
    document.addEventListener("click", onGlobalAction);
    Promise.allSettled([
      loadSettings(false), loadProviders(false), loadRuntime(false), loadStyle(false),
      loadExamples(false), loadFeedback(false), loadJobs(false)
    ]);
    pollTimer = window.setInterval(pollJobs, 1800);
  }
  document.addEventListener("DOMContentLoaded", initialize);
})();
