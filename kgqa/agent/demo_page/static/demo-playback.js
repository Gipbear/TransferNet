/* ─── Task 7：阶段回放状态机 + 芯片交互 + 参数联动 + 评测视角 ───── */
(function () {
  'use strict';

  const { debounce, escHtml, numOr } = window.Demo.util;

  /* ── 文案映射 ──────────────────────────────────────────────────── */
  const STOP_REASON_ZH = {
    path_exhausted: '路径耗尽',
    mixed:          '混合停止比例达标',
    mixed_ratio:    '混合停止比例达标',
    all_wrong:      '全批拒收',
    max_batches:    '达到最大批次数',
  };
  const BATCH_STATUS_ZH = {
    all_correct: '全部通过',
    mixed:       '部分通过',
    all_wrong:   '全批拒收',
  };
  const zh = (map, key) => map[key] || key;

  /* ── 参数读取 ──────────────────────────────────────────────────── */
  function readCheckParams() {
    return {
      tau:          numOr('param-tau', 2.0),
      enableRelExp: document.getElementById('toggle-relexp').checked,
      nMin:         Math.round(numOr('param-nmin', 8)),
      groups:       Math.round(numOr('param-groups', 3)),
    };
  }

  const isKgPid     = pid => String(pid).startsWith('kg');
  const pathIdsOf   = graph => graph.paths.map(p => p.id);
  const pathById    = (graph, pid) =>
    (graph.paths || []).find(p => String(p.id) === String(pid));
  const pathLabel   = pid => isKgPid(pid) ? `P_kg${String(pid).slice(2)}` : `P${pid}`;
  const pathsByTail = (graph, name) =>
    graph.paths.filter(p => p.tail === name).map(p => p.id);
  const uniquePush = (arr, value) => {
    if (value && !arr.includes(value)) arr.push(value);
  };

  function ensureGraphNode(graph, id, layer) {
    const node = graph.nodes.find(n => n.id === id);
    if (!node) {
      graph.nodes.push({ id, layer });
    } else if (layer < node.layer) {
      node.layer = layer;
    }
  }

  function addKgPathToGraph(graph, kgPath) {
    const pid = kgPath.id;
    if (!pid || pathById(graph, pid)) return;
    const triples = kgPath.path || kgPath.triples || [];
    graph.paths.push({
      id: pid,
      label: kgPath.label || pathLabel(pid),
      score: kgPath.score ?? null,
      tail: kgPath.tail || (triples.length ? triples[triples.length - 1][2] : ''),
      triples,
      text: kgPath.text || '',
      synthetic: true,
    });
    triples.forEach(([head, rel, tail], hop) => {
      ensureGraphNode(graph, head, hop);
      ensureGraphNode(graph, tail, hop + 1);
      let edge = graph.edges.find(e =>
        e.source === head && e.relation === rel && e.target === tail);
      if (!edge) {
        edge = { source: head, target: tail, relation: rel, path_ids: [] };
        graph.edges.push(edge);
      }
      uniquePush(edge.path_ids, pid);
    });
  }

  function nextKgOrdinal(replay) {
    const ids = [];
    for (const p of replay.kg_completion_paths || []) ids.push(p.id);
    for (const p of replay.graph?.paths || []) ids.push(p.id);
    for (const a of replay.final_answers || []) ids.push(...(a.kg_path_ids || []));
    for (const item of replay.calibration?.group_expanded_items || []) {
      ids.push(...(item.kg_path_ids || []));
    }
    return ids.reduce((mx, id) => {
      const m = String(id).match(/^kg(\d+)$/);
      return m ? Math.max(mx, Number(m[1])) : mx;
    }, 0) + 1;
  }

  function topicNameOf(graph) {
    const topic = (graph.nodes || []).find(n => Number(n.layer) === 0);
    if (topic) return topic.id;
    const firstEdge = (graph.edges || [])[0];
    return firstEdge ? firstEdge.source : 'Topic';
  }

  function ensureKgCompletionPaths(replay) {
    if (!replay || !replay.graph) return replay;
    const graph = replay.graph;
    graph.nodes = graph.nodes || [];
    graph.edges = graph.edges || [];
    graph.paths = graph.paths || [];
    replay.kg_completion_paths = replay.kg_completion_paths || [];
    replay.kg_completion_paths.forEach(p => addKgPathToGraph(graph, p));

    const cal = replay.calibration || (replay.calibration = {});
    const groupNames = cal.group_expanded_names || [];
    const groupItems = cal.group_expanded_items ||
      groupNames.map(name => ({ name, source_labels: [], kg_path_ids: [] }));
    cal.group_expanded_items = groupItems;
    const itemByName = new Map(groupItems.map(item => [item.name, item]));
    const topic = topicNameOf(graph);
    let next = nextKgOrdinal(replay);

    for (const a of replay.final_answers || []) {
      if (a.via !== 'group_expansion') continue;
      if ((a.kg_path_ids || []).length) {
        for (const pid of a.kg_path_ids) {
          const kgPath = (replay.kg_completion_paths || [])
            .find(p => String(p.id) === String(pid));
          if (kgPath) addKgPathToGraph(graph, kgPath);
        }
        continue;
      }
      let item = itemByName.get(a.name);
      if (!item) {
        item = { name: a.name, source_labels: [], kg_path_ids: [] };
        groupItems.push(item);
        itemByName.set(a.name, item);
      }
      const sourceLabels = [];
      for (const label of [...(item.source_labels || []), ...(a.group_source_labels || [])]) {
        uniquePush(sourceLabels, label);
      }
      item.source_labels = sourceLabels;
      uniquePush(groupNames, a.name);
      const pid = `kg${next++}`;
      const label = pathLabel(pid);
      const sourceLabel = sourceLabels[0] || 'relation_completion';
      const kgPath = {
        id: pid,
        label,
        path: [[topic, `KG: ${sourceLabel}`, a.name]],
        tail: a.name,
        text: `${label}: ${topic} -> ${sourceLabel} -> ${a.name}`,
        source_label: sourceLabel,
        synthetic: true,
      };
      replay.kg_completion_paths.push(kgPath);
      addKgPathToGraph(graph, kgPath);
      a.kg_path_ids = [pid];
      item.kg_path_ids = [pid];
    }
    cal.group_expanded_names = groupNames;
    return replay;
  }

  function visiblePathSignature(pathIds) {
    return [...pathIds].map(String).sort().join('\x1f');
  }

  function graphForVisiblePaths(graph, visiblePathIds) {
    const visible = new Set([...visiblePathIds].map(String));
    const keepNodeIds = new Set();
    const nodeLayer = new Map((graph.nodes || []).map(n => [n.id, n.layer]));
    const paths = (graph.paths || [])
      .filter(p => visible.has(String(p.id)))
      .map(p => Object.assign({}, p, {
        triples: p.triples ? p.triples.map(t => [...t]) : p.triples,
      }));

    const edges = [];
    for (const e of graph.edges || []) {
      const pathIds = (e.path_ids || []).filter(pid => visible.has(String(pid)));
      if (!pathIds.length) continue;
      edges.push(Object.assign({}, e, { path_ids: pathIds }));
      keepNodeIds.add(e.source);
      keepNodeIds.add(e.target);
    }

    if (!keepNodeIds.size && paths.length) {
      for (const p of paths) {
        for (const [head, , tail] of p.triples || []) {
          keepNodeIds.add(head);
          keepNodeIds.add(tail);
        }
      }
    }

    const nodes = (graph.nodes || [])
      .filter(n => keepNodeIds.has(n.id))
      .map(n => Object.assign({}, n));
    for (const id of keepNodeIds) {
      if (!nodes.some(n => n.id === id)) {
        nodes.push({ id, layer: nodeLayer.get(id) ?? 1 });
      }
    }

    return { nodes, edges, paths };
  }

  function ensureFrameGraph(graph, styleByPid) {
    const visiblePathIds = new Set();
    for (const p of graph.paths || []) {
      if (styleByPid[p.id] !== 'hidden') visiblePathIds.add(p.id);
    }
    const sig = visiblePathSignature(visiblePathIds);
    if (Demo.graph._visiblePathSig !== sig) {
      Demo.graph.render(graphForVisiblePaths(graph, visiblePathIds), Demo.graph._prediction || {});
      Demo.graph._visiblePathSig = sig;
    }
  }

  /* ═══════════════════════════════════════════════════════════════
     帧渲染：frame 完整描述图与右栏的可见状态（幂等，可反复重放）
     frame = { graph, replay|null, mmr:'pending'|'done',
               verdictUpTo, answeringBatch|null, calibrated }
  ═══════════════════════════════════════════════════════════════ */
  function applyFrame(f) {
    const g = Demo.graph;

    /* 先在"路径级"归类样式，再由 setPathStyles 按边合成：
       共享公共前缀边取优先级最高的路径样式（如 P1→Mobile 与被剔除的
       P13→Mobile→Alabama 共享第一跳，第一跳保持 evidence 不被刷灰） */
    const styleByPid = {};
    const base = f.mmr === 'done' ? 'normal' : 'dim';
    for (const p of f.graph.paths) styleByPid[p.id] = isKgPid(p.id) ? 'hidden' : base;

    if (f.replay) {
      const iters = f.replay.iterations;
      for (let i = 0; i < f.verdictUpTo; i++) {
        for (const pid of iters[i].accepted_path_ids) styleByPid[pid] = 'evidence';
        for (const pid of iters[i].rejected_path_ids) styleByPid[pid] = 'rejected';
      }
      if (f.answeringBatch !== null && iters[f.answeringBatch]) {
        /* 答题阶段：本批引用先标为蓝色 cited，校验后再转 evidence/rejected */
        for (const pid of iters[f.answeringBatch].cited_path_ids) {
          styleByPid[pid] = 'cited';
        }
      }
      if (f.calibrated) {
        const cal = f.replay.calibration;
        for (const name of cal.dropped_answers) {
          /* 路径级覆盖：被剔除答案的支撑路径整条降为 calibrated-out */
          for (const pid of pathsByTail(f.graph, name)) {
            styleByPid[pid] = 'calibrated-out';
          }
        }
        for (const pid of cal.relation_expanded_path_ids) {
          styleByPid[pid] = 'calibrated-in';
        }
        for (const item of (cal.group_expanded_items || [])) {
          for (const pid of (item.kg_path_ids || [])) {
            styleByPid[pid] = 'calibrated-in';
          }
        }
      }
      ensureFrameGraph(f.graph, styleByPid);
      g.setPathStyles(styleByPid);
      renderTrace(f);
      renderAnswerCard(f.calibrated ? f.replay : null);
    } else {
      ensureFrameGraph(f.graph, styleByPid);
      g.setPathStyles(styleByPid);
      renderTrace(null);
      renderAnswerCard(null);
    }
  }

  /* ── 右栏：批次时间线 ──────────────────────────────────────────── */
  function chipHtml(pid, cls) {
    const mark = cls === 'chip-ok' ? ' ✓' : cls === 'chip-bad' ? ' ✗' : '';
    const data = escHtml(JSON.stringify([pid]));
    return `<span class="chip ${cls}" data-pids='${data}'>${pathLabel(pid)}${mark}</span>`;
  }

  function renderTrace(f) {
    const panel = document.getElementById('trace-panel');
    if (!f || !f.replay) { panel.innerHTML = ''; return; }
    const iters  = f.replay.iterations;
    const shown = Math.max(
      f.verdictUpTo, f.answeringBatch !== null ? f.answeringBatch + 1 : 0);
    let html = '';
    for (let i = 0; i < shown && i < iters.length; i++) {
      const it      = iters[i];
      const verdict = i < f.verdictUpTo;
      const acc     = new Set(it.accepted_path_ids);
      const chips   = it.cited_path_ids.map(pid =>
        chipHtml(pid, verdict ? (acc.has(pid) ? 'chip-ok' : 'chip-bad') : '')
      ).join(' ');
      /* 注意:数据中 batch_index 本身就是 1 起,批次序与 loose/strict 一律
         以循环位置 i 判定,不再依赖 batch_index */
      const mode = i === 0 ? 'loose' : 'strict';

      /* 校验明细子面板(默认收起):模式、接受/拒绝条数、拒绝路径 */
      let checkHtml = '';
      if (verdict) {
        const rej = it.rejected_path_ids;
        checkHtml = `<details class="check-detail">
          <summary>校验明细（${mode === 'loose' ? '宽松' : '严格'}策略 · 接受 ${it.accepted_path_ids.length} 条 / 拒绝 ${rej.length} 条）</summary>
          <div class="check-detail-body">
            ${rej.length
              ? '拒绝路径：<span class="batch-chips">'
                + rej.map(pid => chipHtml(pid, 'chip-bad')).join(' ') + '</span>'
              : '本批全部引用通过校验'}
          </div></details>`;
      }

      html += `<details class="batch-card" data-batch="${i}" open>
        <summary class="batch-card-head">批次 ${i + 1}（${mode} · 路径 P${Number(it.batch_start_rank) | 0}–P${Number(it.batch_end_rank) | 0}）
          <span class="mono-text">${verdict ? escHtml(zh(BATCH_STATUS_ZH, it.batch_status)) : '答题中…'}</span></summary>
        <div class="batch-card-body">
          <div class="batch-chips">${chips}</div>
          ${checkHtml}
        </div></details>`;
    }
    if (html) html = '<div class="card-title">渐进式校验</div>' + html;

    /* 校准帧：追加"图谱引导答案校准"折叠看板（精确性 / 完整性两块） */
    if (f.calibrated) {
      const cal     = f.replay.calibration;
      const dropped = cal.dropped_answers;
      const relexp  = cal.relation_expanded_path_ids;
      const groups  = cal.group_expanded_names;
      const groupItems = cal.group_expanded_items ||
        groups.map(name => ({ name, source_labels: [], kg_path_ids: [] }));

      const droppedBody = dropped.length
        ? '剔除答案：<span class="batch-chips">' + dropped.map(n =>
            `<span class="chip chip-bad">${escHtml(n)} ×</span>`).join(' ') + '</span>'
        : '无答案被剪枝剔除';

      const relexpBody = relexp.length
        ? '关系扩展收回路径：<span class="batch-chips">'
          + relexp.map(pid => chipHtml(pid, 'chip-info')).join(' ') + '</span>'
        : '';
      const groupBody = groups.length
        ? '关系补全答案：<span class="batch-chips">' + groupItems.map(item =>
            `<span class="chip chip-info">${escHtml(item.name)}</span> ${
              (item.kg_path_ids || []).map(pid => chipHtml(pid, 'chip-info')).join(' ')
            }`).join(' ') + '</span>'
        : '';

      html += `<div class="card-title">图谱引导答案校准</div>
        <details class="batch-card calib-card" open>
          <summary class="batch-card-head">精确性校准（相对分数剪枝 τ）
            <span class="mono-text">剔除 ${dropped.length} 个答案</span></summary>
          <div class="batch-card-body">${droppedBody}</div>
        </details>
        <details class="batch-card calib-card" open>
          <summary class="batch-card-head">完整性校准（关系扩展 / 关系补全）
            <span class="mono-text">收回 ${relexp.length} 条 · 补回 ${groups.length} 个</span></summary>
          <div class="batch-card-body">${
            (relexpBody || groupBody)
              ? relexpBody + (relexpBody && groupBody ? '<div style="height:4px"></div>' : '') + groupBody
              : '未触发扩展与补全'
          }</div>
        </details>`;
    }
    panel.innerHTML = html;
    panel.scrollTop = panel.scrollHeight;
  }

  /* ── 右栏：最终答案卡（校准帧才出现；评测视角追加 gold+指标） ──── */
  const ANSWER_CARD_H_KEY = 'kgDemo.answerCardHeight.v2';
  const ANSWER_CARD_MIN_H = 180;

  function answerCardMaxHeight() {
    const panel = document.getElementById('right-panel');
    const h = panel ? panel.clientHeight : window.innerHeight;
    return Math.max(ANSWER_CARD_MIN_H, Math.floor(h * 0.65));
  }

  function answerCardDefaultHeight() {
    const panel = document.getElementById('right-panel');
    const h = panel ? panel.clientHeight : window.innerHeight;
    return Math.max(ANSWER_CARD_MIN_H, Math.floor(h * 0.5));
  }

  function setAnswerCardHeight(px, persist = true) {
    const card = document.getElementById('answer-card');
    if (!card) return;
    const h = Math.min(answerCardMaxHeight(), Math.max(ANSWER_CARD_MIN_H, Math.round(px)));
    card.style.setProperty('--answer-card-h', `${h}px`);
    Demo.state.answerCardHeight = h;
    if (persist) {
      Demo.state.answerCardHeightUserSet = true;
      try { localStorage.setItem(ANSWER_CARD_H_KEY, String(h)); } catch {}
    }
  }

  function restoreAnswerCardHeight() {
    let saved = 0;
    try { saved = Number(localStorage.getItem(ANSWER_CARD_H_KEY) || 0); } catch {}
    Demo.state.answerCardHeightUserSet = !!saved;
    setAnswerCardHeight(saved || answerCardDefaultHeight(), false);
  }

  function syncAnswerCardCollapsed(card) {
    const shell = card.querySelector('.answer-card-shell');
    card.classList.toggle('is-collapsed', !!shell && !shell.open);
  }

  function bindAnswerCardResize() {
    const card = document.getElementById('answer-card');
    if (!card) return;
    card.addEventListener('toggle', ev => {
      if (ev.target.matches('.answer-card-shell')) syncAnswerCardCollapsed(card);
    }, true);
    card.addEventListener('pointerdown', ev => {
      const handle = ev.target.closest('.answer-resize-handle');
      if (!handle || !card.contains(handle)) return;
      const shell = card.querySelector('.answer-card-shell');
      if (shell && !shell.open) {
        shell.open = true;
        syncAnswerCardCollapsed(card);
      }
      ev.preventDefault();
      handle.classList.add('dragging');
      const startY = ev.clientY;
      const startH = card.getBoundingClientRect().height ||
        Demo.state.answerCardHeight || answerCardDefaultHeight();
      const move = moveEv => {
        setAnswerCardHeight(startH + (startY - moveEv.clientY));
      };
      const up = () => {
        handle.classList.remove('dragging');
        window.removeEventListener('pointermove', move);
        window.removeEventListener('pointerup', up);
        window.removeEventListener('pointercancel', up);
      };
      window.addEventListener('pointermove', move);
      window.addEventListener('pointerup', up);
      window.addEventListener('pointercancel', up);
    });
    window.addEventListener('resize', () =>
      setAnswerCardHeight(
        Demo.state.answerCardHeightUserSet
          ? (Demo.state.answerCardHeight || answerCardDefaultHeight())
          : answerCardDefaultHeight(),
        false));
    restoreAnswerCardHeight();
  }

  function renderAnswerCard(replay) {
    const card = document.getElementById('answer-card');
    if (!replay) {
      card.innerHTML = '';
      card.classList.remove('is-collapsed');
      return;
    }
    const existingShell = card.querySelector('.answer-card-shell');
    const keepOpen = existingShell ? existingShell.open : true;
    let bodyHtml = '';
    if (!replay.final_answers.length) bodyHtml += '<div class="ans-row">（空）</div>';
    const VIA_ZH = { relation_expansion: '关系扩展', group_expansion: '关系补全' };
    for (const a of replay.final_answers) {
      /* 校验接受路径 ✓ 芯片 + 关系扩展路径 info 芯片（被拒后经同关系收回，无 ✓） */
      const chips = a.path_ids.map(pid => chipHtml(pid, 'chip-ok')).join(' ')
        + (a.expansion_path_ids || []).map(pid => ' ' + chipHtml(pid, 'chip-info')).join('')
        + (a.kg_path_ids || []).map(pid => ' ' + chipHtml(pid, 'chip-info')).join('');
      const via = VIA_ZH[a.via]
        ? ` <span class="chip chip-info">${VIA_ZH[a.via]}</span>` : '';
      bodyHtml += `<div class="ans-row"><b>${escHtml(a.name)}</b>${via} ${chips}</div>`;
    }
    bodyHtml += `<div class="ans-meta">终止原因：${escHtml(zh(STOP_REASON_ZH, replay.stop_reason))}</div>`;

    if (document.getElementById('toggle-eval').checked && replay.eval) {
      const ev = replay.eval;
      bodyHtml += `<div class="card-title">评测视角</div>
        <div class="ans-meta">
          <span class="chip">Hit@1 ${ev.hit1}</span>
          <span class="chip">F1 ${Number(ev.f1).toFixed(4)}</span>
          <span class="chip">EM ${ev.exact_match ? '✓' : '✗'}</span>
          <span class="chip">Cit-P ${Number(ev.citation_accuracy).toFixed(4)}</span>
        </div>`;
    }
    const answerCount = replay.final_answers.length;
    card.innerHTML = `<div class="answer-resize-handle" title="拖动调整最终答案栏高度"></div>
    <details class="answer-card-shell" ${keepOpen ? 'open' : ''}>
      <summary>
        <span>最终答案</span>
        <span class="mono-text">${answerCount} 个答案</span>
      </summary>
      <div class="answer-card-scroll">${bodyHtml}</div>
    </details>`;
    syncAnswerCardCollapsed(card);
  }

  /* ═══════════════════════════════════════════════════════════════
     Demo.playback  —  阶段回放引擎（play / pause / step / jumpToEnd）
  ═══════════════════════════════════════════════════════════════ */
  const playback = {
    stages: [], cursor: -1, playing: false,
    _timer: null, _subTimers: [],

    load(stages) {
      this.stop();
      this.stages = stages;
      this.cursor = -1;
      this._syncButtons();
    },
    stop() {
      clearTimeout(this._timer); this._timer = null;
      this._clearSub();
      this.playing = false;
    },
    _clearSub() {
      for (const t of this._subTimers) clearTimeout(t);
      this._subTimers = [];
    },
    play() {
      if (!this.stages.length) return;
      if (this.cursor >= this.stages.length - 1) this.cursor = -1; /* 重播 */
      this.playing = true;
      this._syncButtons();
      this._advance();
    },
    pause() {
      clearTimeout(this._timer); this._timer = null;
      this._clearSub();
      this.playing = false;
      this._syncButtons();
    },
    _advance() {
      if (!this.playing) return;
      this.step();
      if (this.cursor >= this.stages.length - 1) {
        this.playing = false; this._syncButtons(); return;
      }
      const dur = this.stages[this.cursor].duration || 1600;
      this._timer = setTimeout(() => this._advance(), dur);
    },
    step() {
      if (this.cursor >= this.stages.length - 1) return;
      this._clearSub();
      this.cursor++;
      this._applyCurrent(false);
      this._syncButtons();
    },
    jumpTo(index) {
      if (!this.stages.length) return;
      const next = Math.max(0, Math.min(this.stages.length - 1, Number(index) || 0));
      this.pause();
      this.cursor = next;
      this._applyCurrent(true);
      this._syncButtons();
    },
    jumpToEnd() {
      if (!this.stages.length) return;
      this.stop();
      this.cursor = this.stages.length - 1;
      this._applyCurrent(true);
      this._syncButtons();
    },
    /* 芯片 hover 离开后恢复当前阶段帧 */
    reapply() {
      if (this.cursor < 0 || !this.stages[this.cursor]) return;
      this._clearSub();
      this.stages[this.cursor].apply({ instant: true });
    },
    _applyCurrent(instant) {
      if (this.cursor < 0 || !this.stages[this.cursor]) return;
      const st = this.stages[this.cursor];
      Demo.ui.setStatus(
        `阶段 ${this.cursor + 1}/${this.stages.length} · ${st.label}`,
        this.cursor === this.stages.length - 1 ? 'ok' : 'info');
      st.apply({ instant });
    },
    _syncButtons() {
      const pauseBtn = document.getElementById('btn-pause');
      const stepBtn  = document.getElementById('btn-step');
      const range    = document.getElementById('playback-range');
      const count    = document.getElementById('playback-count');
      const label    = document.getElementById('playback-step-label');
      const has   = this.stages.length > 0;
      const atEnd = this.cursor >= this.stages.length - 1;
      pauseBtn.disabled    = !has || (atEnd && !this.playing);
      pauseBtn.textContent = this.playing ? '❚❚' : '▶';
      pauseBtn.title = this.playing ? '暂停' : '播放 / 继续';
      pauseBtn.setAttribute('aria-label', pauseBtn.title);
      stepBtn.disabled     = !has || this.playing || atEnd;
      stepBtn.textContent  = '▶|';
      stepBtn.title = '下一步';
      stepBtn.setAttribute('aria-label', '下一步');
      if (range) {
        range.disabled = !has;
        range.max = has ? String(this.stages.length - 1) : '0';
        range.value = String(Math.max(0, this.cursor));
      }
      if (count) {
        count.textContent = has ? `${this.cursor >= 0 ? this.cursor + 1 : 0}/${this.stages.length}` : '0/0';
      }
      if (label) {
        label.textContent = has && this.cursor >= 0
          ? this.stages[this.cursor].label
          : '';
      }
    },
  };
  window.Demo.playback = playback;

  /* ── 阶段构建 ──────────────────────────────────────────────────── */
  function buildStages(graph, replay) {
    const stages = [];
    const retrieved = pathIdsOf(graph).filter(pid => !isKgPid(pid));

    /* 阶段 1：路径检索（候选以浅色淡入） */
    stages.push({
      label: '路径检索：TransferNet 候选生成', duration: 1400,
      apply() {
        applyFrame({ graph, replay: null, mmr: 'pending',
                     verdictUpTo: 0, answeringBatch: null, calibrated: false });
      },
    });

    /* 阶段 2：MMR 筛选（按得分次序逐条亮起，间隔 60ms） */
    const scoreOrder = [...graph.paths]
      .filter(p => !isKgPid(p.id))
      .sort((a, b) => (b.score ?? -1e9) - (a.score ?? -1e9))
      .map(p => p.id);
    stages.push({
      label: 'AP-MMR 多样性筛选：K 条路径入选',
      duration: 60 * scoreOrder.length + 900,
      apply(opt) {
        applyFrame({ graph, replay: null, mmr: 'pending',
                     verdictUpTo: 0, answeringBatch: null, calibrated: false });
        if (opt && opt.instant) {
          Demo.graph.setPathStyle(retrieved, 'normal');
          return;
        }
        scoreOrder.forEach((pid, i) => {
          playback._subTimers.push(
            setTimeout(() => Demo.graph.setPathStyle([pid], 'normal'), 60 * i));
        });
      },
    });

    if (!replay) return stages;   /* 非终版配置：仅播放检索/筛选 */

    /* 每批两阶段：答题（引用高亮）→ 校验（✓/✗） */
    replay.iterations.forEach((it, i) => {
      stages.push({
        label: `批次 ${i + 1} 答题（PFIT-Cite 引用生成）`, duration: 1800,
        apply() {
          applyFrame({ graph, replay, mmr: 'done',
                       verdictUpTo: i, answeringBatch: i, calibrated: false });
        },
      });
      stages.push({
        label: `批次 ${i + 1} 校验（引用一致性 ${i === 0 ? 'loose' : 'strict'}）`,
        duration: 1800,
        apply() {
          applyFrame({ graph, replay, mmr: 'done',
                       verdictUpTo: i + 1, answeringBatch: null, calibrated: false });
        },
      });
    });

    /* 终帧：图谱引导校准（论文截图帧） */
    stages.push({
      label: '图谱引导答案校准', duration: 2000,
      apply() {
        applyFrame({ graph, replay, mmr: 'done',
                     verdictUpTo: replay.iterations.length,
                     answeringBatch: null, calibrated: true });
      },
    });
    return stages;
  }

  /* ── 运行入口（含重放请求竞态令牌） ────────────────────────────── */
  let _replaySeq = 0;

  async function run() {
    const idx = Demo.state.sampleIndex;
    if (idx === null) { Demo.ui.setStatus('请先选择题目', 'error'); return; }
    Demo.util.syncFinalConfigFromInputs();

    if (!Demo.state.isFinalConfig) {
      /* 非终版检索配置：点击提交后再检索，只演示检索/筛选两阶段。 */
      const resp = await Demo.actions.retrieve();
      if (!resp) return;
      Demo.state.replayData = null;
      playback.load(buildStages(resp.graph, null));
      playback.play();
      return;
    }

    const seq = ++_replaySeq;
    Demo.ui.setStatus('加载重放轨迹…', 'info');
    const evalView = document.getElementById('toggle-eval').checked;
    const resp = await Demo.api.replay(idx, readCheckParams(), evalView);
    if (seq !== _replaySeq || !resp) return;
    ensureKgCompletionPaths(resp);

    Demo.state.replayData = resp;
    playback.load(buildStages(resp.graph, resp));
    playback.play();
  }

  /* 校验参数变化：不重播动画，直接刷新到校准完成帧 */
  async function refreshReplay() {
    const idx = Demo.state.sampleIndex;
    if (idx === null || !Demo.state.isFinalConfig || !Demo.state.replayData) return;
    const seq = ++_replaySeq;
    Demo.ui.setStatus('校验参数变更，重放刷新中…', 'info');
    const evalView = document.getElementById('toggle-eval').checked;
    const resp = await Demo.api.replay(idx, readCheckParams(), evalView);
    if (seq !== _replaySeq || !resp) return;
    ensureKgCompletionPaths(resp);
    Demo.state.replayData = resp;
    playback.load(buildStages(resp.graph, resp));
    playback.jumpToEnd();
  }
  const debouncedRefresh = debounce(refreshReplay, 400);

  /* ── 参数联动置灰 ──────────────────────────────────────────────── */
  Demo.ui.applyGating = function () {
    const final = !!Demo.state.isFinalConfig;
    document.getElementById('check-fieldset').disabled = !final;
    document.getElementById('check-disabled-tip').hidden = final;
    const rel = document.getElementById('toggle-relexp').checked;
    document.getElementById('param-nmin').disabled   = !rel;
    document.getElementById('param-groups').disabled = !rel;
  };

  /* ── 评测视角开关 ──────────────────────────────────────────────── */
  /* 独立令牌：不与 run/refresh 共用 _replaySeq，否则播放中开评测会把
     进行中的 run 响应误判为过期，导致回放中断 */
  let _evalSeq = 0;

  async function onEvalToggle() {
    const on   = document.getElementById('toggle-eval').checked;
    const data = Demo.state.replayData;
    if (!data) return;
    if (on && !data.eval) {
      /* 缓存无 gold 数据：带 eval_view 重新请求一次 */
      const seq = ++_evalSeq;
      Demo.ui.setStatus('加载评测数据…', 'info');
      const resp = await Demo.api.replay(
        Demo.state.sampleIndex, readCheckParams(), true);
      if (seq !== _evalSeq || !resp) return;
      ensureKgCompletionPaths(resp);
      if (Demo.state.replayData !== data) return; /* 期间已有新 run/refresh，放弃合并 */
      /* 原地合并：stage 闭包持有同一对象，播放中开启后校准帧也能读到 eval */
      Object.assign(data, resp);
      if (document.getElementById('answer-card').innerHTML) renderAnswerCard(data);
      Demo.ui.setStatus('评测视角已开启', 'ok');
      return;
    }
    /* 关闭（或缓存已含 eval）：直接用缓存重画，不再请求 */
    if (document.getElementById('answer-card').innerHTML) renderAnswerCard(data);
  }

  /* ── 芯片交互（事件委托：hover 高亮 / click 复制） ─────────────── */
  function chipPids(el) {
    try { return JSON.parse(el.dataset.pids || '[]'); } catch { return []; }
  }

  function onChipEnter(el) {
    const pids = chipPids(el);
    const g = Demo.graph._graphData;
    if (!pids.length || !g) return;
    /* 悬停聚焦用中性的蓝色 cited，避免把被拒路径误染成"接受"绿 */
    Demo.graph.setPathStyle(pathIdsOf(g).filter(pid => !isKgPid(pid)), 'dim');
    Demo.graph.setPathStyle(pathIdsOf(g).filter(isKgPid), 'hidden');
    Demo.graph.setPathStyle(pids, 'cited');
  }

  function bubble(el, text) {
    const b = document.createElement('span');
    b.className = 'copy-bubble';
    b.textContent = text;
    el.appendChild(b);
    setTimeout(() => b.remove(), 900);
  }

  async function onChipClick(el) {
    const pids = chipPids(el);
    const data = Demo.state.replayData;
    const g = (data && data.graph) || Demo.graph._graphData;
    if (!pids.length || !g) return;
    const texts = pids.map(pid => (pathById(g, pid) || {}).text || pathLabel(pid));
    let answers = '';
    const card = el.closest('.batch-card');
    if (card && data && data.iterations[+card.dataset.batch]) {
      answers = '\n答案: ' + data.iterations[+card.dataset.batch].answers.join(' | ');
    } else if (data) {
      answers = '\n答案: ' + data.final_answers.map(a => a.name).join(' | ');
    }
    try {
      await navigator.clipboard.writeText(texts.join('\n') + answers);
      bubble(el, '已复制');
    } catch { bubble(el, '复制失败'); }
  }

  function bindChipEvents(panelId) {
    const panel = document.getElementById(panelId);
    panel.addEventListener('mouseover', ev => {
      const chip = ev.target.closest('.chip[data-pids]');
      if (chip && panel.contains(chip)) onChipEnter(chip);
    });
    panel.addEventListener('mouseout', ev => {
      const chip = ev.target.closest('.chip[data-pids]');
      if (chip) playback.reapply();
    });
    panel.addEventListener('click', ev => {
      const chip = ev.target.closest('.chip[data-pids]');
      if (chip) onChipClick(chip);
    });
  }

  /* ── 事件绑定 ──────────────────────────────────────────────────── */
  document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('btn-run').addEventListener('click', run);
    document.getElementById('btn-pause').addEventListener('click', () => {
      playback.playing ? playback.pause() : playback.play();
    });
    document.getElementById('btn-step').addEventListener('click', () => playback.step());
    document.getElementById('playback-range').addEventListener('input', ev => {
      playback.jumpTo(Number(ev.target.value));
    });
    document.querySelectorAll('[data-layout-mode]').forEach(btn => {
      btn.addEventListener('click', () => Demo.graph.setLayoutMode(btn.dataset.layoutMode));
    });
    Demo.graph._syncLayoutButtons();

    for (const id of ['param-tau', 'param-nmin', 'param-groups']) {
      document.getElementById(id).addEventListener('change', debouncedRefresh);
    }
    document.getElementById('toggle-relexp').addEventListener('change', () => {
      Demo.ui.applyGating();
      debouncedRefresh();
    });
    document.getElementById('toggle-eval').addEventListener('change', onEvalToggle);

    bindChipEvents('trace-panel');
    bindChipEvents('answer-card');
    bindAnswerCardResize();

    /* 页面初始按当前检索参数判定是否为终版配置。 */
    Demo.util.syncFinalConfigFromInputs();
  });

})();
