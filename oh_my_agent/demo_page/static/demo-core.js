/* ─── 全局命名空间占位 ──────────────────────────────────────────────── */
window.Demo = { state: {}, api: {}, graph: {}, playback: {}, ui: {} };

/* ─── Task 6：前端接线与 KG 分层渲染 ──────────────────────────────── */
(function () {
  'use strict';

  /* ── 工具函数 ──────────────────────────────────────────────────── */
  function debounce(fn, ms) {
    let t;
    return function (...args) {
      clearTimeout(t);
      t = setTimeout(() => fn.apply(this, args), ms);
    };
  }

  function escHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  /* Finding 1: NaN 守卫——防止 0 值被 || default 吞掉 */
  function numOr(id, def) {
    const v = parseFloat(document.getElementById(id).value);
    return isNaN(v) ? def : v;
  }

  /* Finding 4: 请求序号，防止并发检索竞态（旧响应覆盖新响应） */
  let _retrieveSeq = 0;
  const FINAL_RETRIEVAL = { k: 50, lambda: 0.2, eta: 1.0 };

  function readRetrievalParams() {
    return {
      k:      Math.round(numOr('param-k',      FINAL_RETRIEVAL.k)),
      lambda: numOr('param-lambda', FINAL_RETRIEVAL.lambda),
      eta:    numOr('param-eta',    FINAL_RETRIEVAL.eta),
    };
  }

  function isFinalRetrievalParams(params) {
    return params.k === FINAL_RETRIEVAL.k
      && params.lambda === FINAL_RETRIEVAL.lambda
      && params.eta === FINAL_RETRIEVAL.eta;
  }

  function syncFinalConfigFromInputs() {
    Demo.state.isFinalConfig = isFinalRetrievalParams(readRetrievalParams());
    if (Demo.ui.applyGating) Demo.ui.applyGating();
    return Demo.state.isFinalConfig;
  }

  /* 边唯一键：source + relation + target */
  function eKey(e) {
    return e.source + '\x00' + e.relation + '\x00' + e.target;
  }

  /* 路径边实例键：同一条 KG 边被多条路径使用时，前端分开画、分开着色。 */
  function edgePathKey(e, pid) {
    return eKey(e) + '\x00' + String(pid);
  }

  function pathSortValue(pid) {
    const text = String(pid);
    if (text.startsWith('kg')) {
      const n = Number(text.slice(2));
      return Number.isFinite(n) ? 100000 + n : 999999;
    }
    const n = Number(pid);
    return Number.isFinite(n) ? n : 999999;
  }

  function graphPathLabel(pid) {
    const text = String(pid);
    return text.startsWith('kg') ? `P_kg${text.slice(2)}` : `P${text}`;
  }

  function isReverseRelation(rel) {
    return String(rel || '').endsWith('_reverse');
  }

  function circularMean(angles) {
    if (!angles.length) return undefined;
    let sx = 0;
    let sy = 0;
    for (const a of angles) {
      sx += Math.cos(a);
      sy += Math.sin(a);
    }
    if (Math.abs(sx) < 1e-6 && Math.abs(sy) < 1e-6) return angles[0];
    return Math.atan2(sy, sx);
  }

  /* ── 路径样式映射（styleName → ECharts lineStyle） ────────────────
     按推理流水线阶段编码：
       dim   候选淡入          normal  AP-MMR 选中（灰）
       cited 本批 PFIT 引用（蓝粗）
       evidence 校验接受（青绿粗实线）   rejected 校验拒绝（浅红实线）
       calibrated-out 剪枝剔除（浅红虚线） calibrated-in 扩展收回（青绿虚线,闪现）
     拒绝/剔除同色不同线型、接受/收回同色不同线型：灰度打印仍可区分 */
  const EDGE_STYLES = {
    'hidden':         { color: 'rgba(0,0,0,0)', width: 0.0, type: 'solid',  opacity: 0.0  },
    'dim':            { color: '#C9CED6', width: 1.0, type: 'solid',  opacity: 0.30 },
    'normal':         { color: '#9AA3AE', width: 1.4, type: 'solid',  opacity: 0.70 },
    'cited':          { color: '#3B6FA0', width: 2.8, type: 'solid',  opacity: 1.0  },
    'evidence':       { color: '#3A8A6E', width: 3.2, type: 'solid',  opacity: 1.0  },
    'rejected':       { color: '#C85C5C', width: 1.8, type: 'solid',  opacity: 0.85 },
    'calibrated-out': { color: '#D98E8E', width: 1.6, type: 'dashed', opacity: 0.75 },
    'calibrated-in':      { color: '#3A8A6E', width: 2.8, type: 'dashed', opacity: 1.0  },
    '_calibrated-in-dim': { color: '#3A8A6E', width: 2.8, type: 'dashed', opacity: 0.25 }, /* Finding 3: 闪烁暗相位 */
  };
  const STYLE_PRIORITY = ['hidden', 'dim', 'normal', 'calibrated-out', 'rejected',
                          'cited', 'evidence', 'calibrated-in'];
  const styleRank = s => STYLE_PRIORITY.indexOf(s);
  const activeStyleName = (s, flashDim) =>
    (s === 'calibrated-in' && flashDim) ? '_calibrated-in-dim' : s;

  /* ═══════════════════════════════════════════════════════════════
     Demo.state
  ═══════════════════════════════════════════════════════════════ */
  window.Demo.state = {
    sampleIndex:   null,   // 当前选中题目的 sample_index
    isFinalConfig: false,  // 是否为终版检索参数（Task 7 联动）
  };

  /* ═══════════════════════════════════════════════════════════════
     Demo.api  —  三个后端接口封装
  ═══════════════════════════════════════════════════════════════ */
  window.Demo.api = {

    /** GET /api/questions?q=&limit=20 → [{sample_index, question}] */
    async searchQuestions(q) {
      try {
        const r = await fetch(`/api/questions?q=${encodeURIComponent(q)}&limit=20`);
        if (!r.ok) {
          const err = await r.json().catch(() => ({ detail: r.statusText }));
          Demo.ui.setStatus('搜索失败：' + (err.detail || r.statusText), 'error');
          return [];
        }
        return r.json();
      } catch (ex) {
        Demo.ui.setStatus('网络错误：' + ex.message, 'error');
        return [];
      }
    },

    /** POST /api/retrieve {sample_index, beam_size, lambda_val, alpha_final}
     *  → {graph, prediction, is_final_config, elapsed_ms}            */
    async retrieve(sampleIndex, { k, lambda, eta }) {
      try {
        const r = await fetch('/api/retrieve', {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            sample_index: sampleIndex,
            beam_size:    k,
            lambda_val:   lambda,
            alpha_final:  eta,
          }),
        });
        if (!r.ok) {
          const err = await r.json().catch(() => ({ detail: r.statusText }));
          Demo.ui.setStatus('检索失败：' + (err.detail || r.statusText), 'error');
          return null;
        }
        return r.json();
      } catch (ex) {
        Demo.ui.setStatus('网络错误：' + ex.message, 'error');
        return null;
      }
    },

    /** POST /api/replay {sample_index, score_margin, ...}
     *  → {iterations, final_answers, calibration, stop_reason, graph}
     *  （Task 7 消费；本任务只封装接口）                              */
    async replay(sampleIndex, checkParams = {}, evalView = false) {
      try {
        const r = await fetch('/api/replay', {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            sample_index:              sampleIndex,
            score_margin:              checkParams.tau            ?? 2.0,
            enable_relation_expansion: checkParams.enableRelExp   ?? true,
            expansion_min_answers:     checkParams.nMin           ?? 8,
            expansion_top_groups:      checkParams.groups         ?? 3,
            eval_view:                 evalView,
          }),
        });
        if (!r.ok) {
          const err = await r.json().catch(() => ({ detail: r.statusText }));
          Demo.ui.setStatus('重放失败：' + (err.detail || r.statusText), 'error');
          return null;
        }
        return r.json();
      } catch (ex) {
        Demo.ui.setStatus('网络错误：' + ex.message, 'error');
        return null;
      }
    },
  };

  /* ═══════════════════════════════════════════════════════════════
     Demo.graph  —  ECharts 分层渲染与路径样式通道
  ═══════════════════════════════════════════════════════════════ */
  window.Demo.graph = {
    _chart:       null,      // ECharts 实例
    _graphData:   null,      // 最后一次 render 的 graph JSON
    _prediction:  null,      // 最后一次 render 的 prediction JSON
    _edgeStyles:  new Map(), // edgePathKey → styleName，按 setPathStyle 累积
    _option:      null,      // 完整 ECharts option（用于 _applyStyles 重推）
    _origNodes:   null,      // 原始节点数据（阶段样式重绘时参照）
    _visiblePathSig: null,   // 当前播放帧实际渲染的路径集合签名
    _flashTimers: [],        // Finding 3: calibrated-in 闪烁计时器
    _flashDim:    false,     // Finding 3: 当前是否处于闪烁暗相位
    _layoutMode:  'radial',  // radial | force

    /**
     * 渲染子图。
     * radial: topic 在中心，hop 层映射为同心圆，path id 决定放射角。
     * force:  ECharts 力导向布局，topic 固定在中心，其余节点受斥力/边弹簧展开。
     * prediction score≥0.9 的节点加青绿描边。
     */
    setLayoutMode(mode) {
      if (!['radial', 'force'].includes(mode)) return;
      if (this._layoutMode === mode) return;
      this._layoutMode = mode;
      this._syncLayoutButtons();
      if (this._graphData) {
        this.render(this._graphData, this._prediction || {});
        window.Demo.playback?.reapply?.();
      }
    },

    _syncLayoutButtons() {
      document.querySelectorAll('[data-layout-mode]').forEach(btn => {
        const active = btn.dataset.layoutMode === this._layoutMode;
        btn.classList.toggle('active', active);
        btn.setAttribute('aria-pressed', active ? 'true' : 'false');
      });
    },

    render(graph, prediction) {
      this._graphData  = graph;
      this._prediction = prediction;
      this._edgeStyles.clear();
      this._visiblePathSig = null;
      this._clearFlashTimers();    /* Finding 3: 新图加载时清除旧闪烁 */

      /* ECharts 延迟初始化（DOMContentLoaded 中已完成，此处为回退兜底） */
      const canvas = document.getElementById('kg-canvas');
      if (!this._chart) {
        this._chart = echarts.init(canvas);
      }

      const H = canvas.offsetHeight || 600;
      const W = canvas.offsetWidth  || 900;
      const forceMode = this._layoutMode === 'force';

      /* 得分≥0.9 的候选答案节点集合 */
      const highScore = new Set(
        Object.entries(prediction)
          .filter(([, s]) => s >= 0.9)
          .map(([n]) => n)
      );

      /* 按层分组节点 */
      const byLayer = {};
      for (const n of graph.nodes) {
        (byLayer[n.layer] = byLayer[n.layer] || []).push(n.id);
      }

      /* 各节点首个 path_id（用于同层内排序） */
      const nodeFirstPid = {};
      for (const e of graph.edges) {
        const pidVals = e.path_ids.map(pathSortValue);
        const minPid = pidVals.length ? Math.min(...pidVals) : 999999;
        for (const nid of [e.source, e.target]) {
          if (nodeFirstPid[nid] === undefined || minPid < nodeFirstPid[nid]) {
            nodeFirstPid[nid] = minPid;
          }
        }
      }

      const pathById = new Map((graph.paths || []).map(p => [String(p.id), p]));
      const pathIds = [...new Set((graph.paths || []).map(p => p.id))]
        .sort((a, b) => pathSortValue(a) - pathSortValue(b));
      const tailOfPath = p => p.tail || ((p.triples || []).at(-1) || [])[2] || `path:${p.id}`;
      const tailGroups = new Map();
      for (const pid of pathIds) {
        const p = pathById.get(String(pid));
        const tail = p ? tailOfPath(p) : `path:${pid}`;
        if (!tailGroups.has(tail)) tailGroups.set(tail, []);
        tailGroups.get(tail).push(pid);
      }
      const tails = [...tailGroups.keys()].sort((a, b) => {
        const amin = Math.min(...tailGroups.get(a).map(pathSortValue));
        const bmin = Math.min(...tailGroups.get(b).map(pathSortValue));
        return amin - bmin || a.localeCompare(b);
      });

      /* 按"答案/尾节点簇"均匀分配整圈；簇内多条路径轻微展开，避免重复答案挤在半边。 */
      const angleByPid = new Map();
      const angleStart = -Math.PI / 2;
      const clusterStep = tails.length ? (2 * Math.PI / tails.length) : 0;
      tails.forEach((tail, tailIndex) => {
        const pids = tailGroups.get(tail).sort((a, b) => pathSortValue(a) - pathSortValue(b));
        const clusterAngle = angleStart + clusterStep * tailIndex;
        const spread = Math.min(clusterStep * 0.42, 0.24);
        pids.forEach((pid, i) => {
          const offset = pids.length <= 1 ? 0 : (i - (pids.length - 1) / 2) * (spread / (pids.length - 1));
          angleByPid.set(pid, clusterAngle + offset);
        });
      });

      /* 节点优先取"作为尾节点"的路径角度；中间节点取参与路径的圆形均值。 */
      const nodePathIds = {};
      const nodeTailPathIds = {};
      for (const e of graph.edges) {
        for (const nid of [e.source, e.target]) {
          for (const pid of e.path_ids || []) {
            (nodePathIds[nid] = nodePathIds[nid] || new Set()).add(pid);
            const p = pathById.get(String(pid));
            if (p && tailOfPath(p) === nid) {
              (nodeTailPathIds[nid] = nodeTailPathIds[nid] || new Set()).add(pid);
            }
          }
        }
      }

      /* 同心圆半径：内圈留给 1-hop，中外圈给答案与补全路径。 */
      const maxLayer = graph.nodes.reduce((m, n) => Math.max(m, n.layer), 0);
      const cx = W / 2;
      const cy = H / 2;
      const maxRadius = Math.max(140, Math.min(W, H) * 0.42);
      const radiusOf = layer => {
        if (layer <= 0) return 0;
        if (maxLayer <= 1) return maxRadius * 0.72;
        const inner = 0.42;
        const span = 1 - inner;
        return maxRadius * (inner + span * ((layer - 1) / Math.max(1, maxLayer - 1)));
      };

      /* 各层节点排序 → 计算径向 (x, y) */
      const nodePos = {};
      for (const [layer, ids] of Object.entries(byLayer)) {
        ids.sort((a, b) => (nodeFirstPid[a] ?? 999999) - (nodeFirstPid[b] ?? 999999));
        const ringStep = ids.length ? (2 * Math.PI / ids.length) : 0;
        ids.forEach((id, i) => {
          const layerNum = Number(layer);
          const anglePids = [...(nodeTailPathIds[id] || nodePathIds[id] || [])];
          let angle = circularMean(anglePids.map(pid => angleByPid.get(pid)).filter(a => a !== undefined));
          if (angle === undefined) {
            angle = angleStart + ringStep * i + (layerNum % 2 ? ringStep / 2 : 0);
          }
          if (layerNum === 0) {
            const topicRadius = ids.length > 1 ? 34 : 0;
            const topicAngle = angleStart + ringStep * i;
            nodePos[id] = {
              x: cx + topicRadius * Math.cos(topicAngle),
              y: cy + topicRadius * Math.sin(topicAngle),
              angle: topicAngle,
            };
          } else {
            const r = radiusOf(layerNum);
            nodePos[id] = {
              x: cx + r * Math.cos(angle),
              y: cy + r * Math.sin(angle),
              angle,
            };
          }
        });
      }

      const ringGraphics = [];
      for (let layer = 1; layer <= maxLayer; layer++) {
        const r = radiusOf(layer);
        ringGraphics.push({
          type: 'circle',
          silent: true,
          z: -10,
          shape: { cx, cy, r },
          style: {
            fill: 'transparent',
            stroke: layer === maxLayer ? 'rgba(31,78,121,0.16)' : 'rgba(205,210,220,0.55)',
            lineWidth: layer === maxLayer ? 1.4 : 1,
            lineDash: [5, 7],
          },
        });
      }

      /* ECharts data（节点） */
      const ecNodes = graph.nodes.map(n => {
        const pos    = nodePos[n.id] || { x: cx, y: cy };
        const topic  = n.layer === 0;
        const answer = highScore.has(n.id);
        const lbl    = n.id.length > 22 ? n.id.slice(0, 20) + '…' : n.id;
        const labelPos = topic ? 'bottom' : (Math.cos(pos.angle || 0) < -0.15 ? 'left' : 'right');
        return {
          name:       n.id,
          x:          pos.x,
          y:          pos.y,
          symbolSize: topic ? 24 : (answer ? 16 : 14),
          label: {
            show:      true,
            position:  labelPos,
            formatter: lbl,
            fontSize:  11,
            color:     '#1A1A1A',
          },
          itemStyle: {
            color:       topic ? '#1F4E79' : (answer ? '#26724F' : '#3B6FA0'),
            borderColor: answer ? '#3A8A6E' : (topic ? '#0D2B47' : 'transparent'),
            borderWidth: answer ? 3          : (topic ? 1          : 0),
          },
          fixed: forceMode && topic,
          tooltip: { formatter: escHtml(n.id) },
        };
      });

      /* Finding 2: 存储原始节点数据，供 _applyStyles 重绘时参照 */
      this._origNodes = ecNodes;

      /* ECharts links（边） */
      const ecLinks = this._buildLinks(graph.edges, this._edgeStyles);

      this._option = {
        backgroundColor: 'transparent',
        graphic: forceMode ? [] : ringGraphics,
        tooltip: {
          trigger: 'item',
          formatter: params => {
            if (params.dataType === 'edge') {
              const d = params.data;
              /* 关系全名放 tooltip，短名放 label */
              return `${escHtml(d._src)} → ${escHtml(d._tgt)}`
                + `<br><span style="color:#888;font-size:11px">${escHtml(d._rel)}</span>`
                + `<br>路径: ${(d._pids || []).join(', ')}`;
            }
            return escHtml(params.name);
          },
        },
        series: [{
          type:      'graph',
          layout:    forceMode ? 'force' : 'none',
          center:    [cx, cy],
          zoom:      1,
          data:      ecNodes,
          links:     ecLinks,
          roam:      true,
          draggable: forceMode,
          label:     { show: true },
          lineStyle: { curveness: 0.10 },
          emphasis:  { focus: 'adjacency', lineStyle: { width: 3 } },
          edgeSymbol:     ['none', 'arrow'],
          edgeSymbolSize: [0, 8],
          force: forceMode ? {
            repulsion: 220,
            edgeLength: [90, 180],
            gravity: 0.08,
            friction: 0.42,
            layoutAnimation: true,
          } : undefined,
        }],
      };

      this._chart.setOption(this._option, /* notMerge= */ true);
    },

    /**
     * 按 path_id 命中具体边实例着色。
     * styleName ∈ {normal, dim, evidence, rejected, calibrated-out, calibrated-in}
     */
    setPathStyle(pathIds, styleName) {
      this._clearFlashTimers();         /* Finding 3: 新样式前先清除旧闪烁 */
      if (!this._graphData) return;
      const pidSet = new Set(pathIds);
      for (const e of this._graphData.edges) {
        for (const pid of e.path_ids || []) {
          if (pidSet.has(pid)) {
            this._edgeStyles.set(edgePathKey(e, pid), styleName);
          }
        }
      }
      this._applyStyles();
      if (styleName === 'calibrated-in') this._startCalibFlash(); /* Finding 3 */
    },

    /**
     * 按"路径级样式表"一次性合成边实例样式。
     * 同一条 KG 边被多条路径使用时，每个 path_id 都保留独立线条与状态。
     */
    setPathStyles(styleByPid) {
      this._clearFlashTimers();
      if (!this._graphData) return;
      let hasCalibIn = false;
      for (const e of this._graphData.edges) {
        for (const pid of e.path_ids || []) {
          const s = styleByPid[pid];
          const k = edgePathKey(e, pid);
          if (s) {
            this._edgeStyles.set(k, s);
            if (s === 'calibrated-in') hasCalibIn = true;
          } else {
            this._edgeStyles.delete(k);
          }
        }
      }
      this._applyStyles();
      if (hasCalibIn) this._startCalibFlash();
    },

    /** 重置所有边样式为 normal */
    resetStyles() {
      this._clearFlashTimers();   /* Finding 3 */
      this._edgeStyles.clear();
      this._applyStyles();
    },

    clear() {
      this._clearFlashTimers();
      this._graphData = null;
      this._prediction = null;
      this._edgeStyles.clear();
      this._option = null;
      this._origNodes = null;
      this._visiblePathSig = null;
      if (this._chart) this._chart.clear();
    },

    /* ── 内部辅助 ─────────────────────────────────────────────── */

    /* Finding 3: 清除所有闪烁计时器，重置暗相位标志 */
    _clearFlashTimers() {
      for (const t of this._flashTimers) clearTimeout(t);
      this._flashTimers = [];
      this._flashDim    = false;
    },

    /* Finding 3: calibrated-in 闪烁序列——~1.2s 内切换 3 次 opacity，最终停留实心青绿 */
    _startCalibFlash() {
      const seq = [
        { delay: 0,    dim: true  },
        { delay: 400,  dim: false },
        { delay: 800,  dim: true  },
        { delay: 1200, dim: false },  /* 最终停留实心青绿 */
      ];
      for (const { delay, dim } of seq) {
        this._flashTimers.push(setTimeout(() => {
          this._flashDim = dim;
          this._applyStyles();
        }, delay));
      }
    },

    _buildLinks(edges, styleMap) {
      const links = [];
      for (const e of edges) {
        const pids = [...(e.path_ids || [])].sort((a, b) => pathSortValue(a) - pathSortValue(b));
        for (const pid of pids) {
          links.push({ edge: e, pid });
        }
      }

      /* 同源同目标的多条路径实例用不同曲率展开，避免视觉上继续重叠。 */
      const byPair = new Map();
      for (const link of links) {
        const reverse = isReverseRelation(link.edge.relation);
        const shownSource = reverse ? link.edge.target : link.edge.source;
        const shownTarget = reverse ? link.edge.source : link.edge.target;
        const pairKey = shownSource + '\x00' + shownTarget;
        if (!byPair.has(pairKey)) byPair.set(pairKey, []);
        byPair.get(pairKey).push(link);
      }
      for (const group of byPair.values()) {
        group.sort((a, b) => {
          const relCmp = a.edge.relation.localeCompare(b.edge.relation);
          return relCmp || (pathSortValue(a.pid) - pathSortValue(b.pid));
        });
        const mid = (group.length - 1) / 2;
        const spread = group.length > 6 ? 0.055 : 0.075;
        group.forEach((link, i) => {
          link.curveness = group.length === 1 ? 0.10 : (i - mid) * spread;
        });
      }

      return links.map(({ edge: e, pid, curveness }) => {
        const rawStyle = styleMap.get(edgePathKey(e, pid)) || 'normal';
        /* Finding 3: 闪烁暗相位时将 calibrated-in 临时映射为 dim 版本 */
        const sn = activeStyleName(rawStyle, this._flashDim);
        const ls  = EDGE_STYLES[sn] || EDGE_STYLES['normal'];
        const relShort = e.relation.split('.').pop();
        const pLabel = graphPathLabel(pid);
        const reverse = isReverseRelation(e.relation);
        const shownSource = reverse ? e.target : e.source;
        const shownTarget = reverse ? e.source : e.target;
        return {
          id:     edgePathKey(e, pid),
          source: shownSource,
          target: shownTarget,
          /* 附加字段：tooltip 读取 */
          _src:  shownSource,
          _tgt:  shownTarget,
          _pathSrc: e.source,
          _pathTgt: e.target,
          _rel:  e.relation,
          _pid:  pid,
          _pids: [pid],
          _pathLabel: pLabel,
          label: {
            show:      true,
            formatter: `${pLabel}\n${relShort}`,
            fontSize:  10,
            color:     ls.color,
          },
          lineStyle: {
            color:     ls.color,
            width:     ls.width,
            type:      ls.type,
            opacity:   ls.opacity,
            curveness: curveness,
          },
        };
      });
    },

    _applyStyles() {
      if (!this._chart || !this._graphData || !this._option) return;

      /* 尾节点继承入边阶段色；同一节点多条入边时按路径样式优先级取最高态。 */
      const incomingStyleByNode = {};
      for (const e of this._graphData.edges) {
        for (const pid of e.path_ids || []) {
          const k = edgePathKey(e, pid);
          const s = this._edgeStyles.get(k) || 'normal';
          const prev = incomingStyleByNode[e.target];
          if (!prev || styleRank(s) > styleRank(prev)) {
            incomingStyleByNode[e.target] = s;
          }
        }
      }

      /* 始终从原始节点数据重建，避免反复播放/hover 后样式叠加污染。 */
      const baseNodes = this._origNodes || this._option.series[0].data;
      this._option.series[0].data = baseNodes.map(nd => {
        const rawStyle = incomingStyleByNode[nd.name];
        if (rawStyle) {
          if (rawStyle === 'hidden') {
            return Object.assign({}, nd, {
              symbolSize: 0,
              label: Object.assign({}, nd.label, { show: false }),
              itemStyle: Object.assign({}, nd.itemStyle, { opacity: 0 }),
            });
          }
          const sn = activeStyleName(rawStyle, this._flashDim);
          const ns = EDGE_STYLES[sn] || EDGE_STYLES['normal'];
          const removed = rawStyle === 'rejected' || rawStyle === 'calibrated-out';
          const label = Object.assign({}, nd.label, {
            formatter: (removed ? '❌ ' : '') + nd.label.formatter,
            color: removed ? ns.color : nd.label.color,
          });
          return Object.assign({}, nd, {
            symbolSize: removed ? Math.max(nd.symbolSize, 18) : nd.symbolSize,
            label,
            itemStyle: Object.assign({}, nd.itemStyle, {
              color: ns.color,
              borderColor: ns.color,
              borderWidth: removed ? 2 : 1,
            }),
          });
        }
        return nd;   /* 无入边节点自然恢复原样（来自 _origNodes） */
      });

      this._option.series[0].links =
        this._buildLinks(this._graphData.edges, this._edgeStyles);
      this._chart.setOption(this._option, true);
    },
  };

  /* ═══════════════════════════════════════════════════════════════
     Demo.ui  —  状态栏辅助
  ═══════════════════════════════════════════════════════════════ */
  window.Demo.ui = {
    setStatus(msg, type = 'info') {
      const el = document.getElementById('stage-label');
      if (!el) return;
      el.textContent = msg;
      const c = { error: '#D98E4A', ok: '#3A8A6E', info: '#7A8090' };
      el.style.color       = c[type] || c.info;
      el.style.borderColor = c[type] || '';
    },
    clearStatus() { this.setStatus('', 'info'); },
  };

  /* ═══════════════════════════════════════════════════════════════
     交互逻辑
  ═══════════════════════════════════════════════════════════════ */

  /** 读取当前参数 → 调检索 → 渲染 */
  async function doRetrieve() {
    const idx = Demo.state.sampleIndex;
    if (idx === null) return null;
    const seq = ++_retrieveSeq;           /* Finding 4: 防并发竞态 */

    const { k, lambda, eta } = readRetrievalParams();

    Demo.ui.setStatus('检索中…', 'info');
    const resp = await Demo.api.retrieve(idx, { k, lambda, eta });
    if (seq !== _retrieveSeq) return null;     /* Finding 4: 丢弃过期响应 */
    if (!resp) return null;

    /* 更新终版配置标志（Task 7 读取） */
    Demo.state.isFinalConfig = resp.is_final_config;
    const tipEl = document.getElementById('check-disabled-tip');
    if (tipEl) tipEl.hidden = resp.is_final_config;

    Demo.ui.setStatus(
      `检索完成 · ${resp.graph.paths.length} 条路径 · ${Math.round(resp.elapsed_ms)} ms`
        + (resp.is_final_config ? ' · [终版配置]' : ''),
      resp.is_final_config ? 'ok' : 'info'
    );

    Demo.graph.render(resp.graph, resp.prediction);

    /* Task 7 钩子：检索配置变化后刷新校验区联动置灰 */
    if (Demo.ui.applyGating) Demo.ui.applyGating();
    return resp;
  }

  /* ── 题目下拉列表 ───────────────────────────────────────── */

  function renderList(results) {
    const ul = document.getElementById('question-list');
    ul.innerHTML = '';
    if (!results.length) { ul.classList.remove('open'); return; }
    for (const item of results) {
      const li = document.createElement('li');
      li.innerHTML =
        `<span class="q-idx">#${item.sample_index}</span>${escHtml(item.question)}`;
      li.addEventListener('mousedown', ev => {
        ev.preventDefault();   /* 阻止 input blur，保证 click 能触发 */
        selectQuestion(item);
      });
      ul.appendChild(li);
    }
    ul.classList.add('open');
  }

  function closeList() {
    document.getElementById('question-list').classList.remove('open');
  }

  function selectQuestion(item) {
    Demo.state.sampleIndex = item.sample_index;
    document.getElementById('question-input').value = item.question;
    document.getElementById('selected-question').textContent =
      `#${item.sample_index}  ${item.question}`;
    closeList();
    Demo.state.replayData = null;
    Demo.playback?.load?.([]);
    Demo.graph.clear();
    document.getElementById('trace-panel').innerHTML = '';
    document.getElementById('answer-card').innerHTML = '';
    syncFinalConfigFromInputs();
    Demo.ui.setStatus('已选择题目，点击提交开始', 'info');
  }

  const debouncedSearch = debounce(async function (q) {
    if (!q.trim()) { closeList(); return; }
    const results = await Demo.api.searchQuestions(q);
    renderList(results);
  }, 250);

  /* ═══════════════════════════════════════════════════════════════
     DOMContentLoaded：初始化 ECharts + 绑定事件
  ═══════════════════════════════════════════════════════════════ */
  document.addEventListener('DOMContentLoaded', () => {
    /* ECharts 初始化（在 DOM 就绪后确保画布已有尺寸） */
    const canvas = document.getElementById('kg-canvas');
    Demo.graph._chart = echarts.init(canvas);
    window.addEventListener('resize', () => Demo.graph._chart?.resize());

    /* 题目搜索框 */
    const qInput = document.getElementById('question-input');
    qInput.addEventListener('input',   function () { debouncedSearch(this.value); });
    qInput.addEventListener('blur',    () => setTimeout(closeList, 150));
    qInput.addEventListener('keydown', ev => { if (ev.key === 'Escape') closeList(); });

    /* 检索参数变化只更新配置状态；图谱生成统一由"提交"触发。 */
    for (const id of ['param-k', 'param-lambda', 'param-eta']) {
      document.getElementById(id).addEventListener('change', () => {
        Demo.state.replayData = null;
        Demo.playback?.load?.([]);
        Demo.graph.clear();
        document.getElementById('trace-panel').innerHTML = '';
        document.getElementById('answer-card').innerHTML = '';
        syncFinalConfigFromInputs();
        Demo.ui.setStatus('检索参数已变更，点击提交重新生成', 'info');
      });
    }
  });

  /* Task 7 复用的内部工具 */
  window.Demo.util = { debounce, escHtml, numOr, readRetrievalParams, syncFinalConfigFromInputs };
  window.Demo.actions = { retrieve: doRetrieve };

})();
