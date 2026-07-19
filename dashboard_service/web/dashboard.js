const API = "/api";
const REFRESH_MS = 600;

const state = {
  overview: null,
  selectedPairId: null,
  series: null,
  connected: false,
  busy: false,
};

const el = (id) => document.getElementById(id);

function safe(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function number(value, digits = 1, suffix = "") {
  if (value === null || value === undefined || value === "") return "--";
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return "--";
  const sign = parsed > 0 ? "+" : "";
  return `${sign}${parsed.toFixed(digits)}${suffix}`;
}

function probability(value) {
  if (value === null || value === undefined || value === "") return "--";
  const parsed = Number(value);
  return Number.isFinite(parsed) ? `${Math.round(parsed * 100)}%` : "--";
}

function timeOf(timestamp) {
  if (!timestamp) return "--:--:--";
  return new Date(Number(timestamp)).toLocaleTimeString("en-GB", {
    hour12: false,
  });
}

function duration(milliseconds) {
  const seconds = Math.max(0, Number(milliseconds) / 1000);
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}

function pnlClass(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed === 0) return "";
  return parsed > 0 ? "positive" : "negative";
}

function actionLabel(value) {
  if (value === null || value === undefined || value === "") return "--";
  return ({ 0: "WAIT", 1: "ENTER", 2: "HOLD", 3: "EXIT" })[Number(value)] ?? "--";
}

function stablePairOrder(left, right) {
  // Edge меняется каждый тик, поэтому сортируем по стабильным атрибутам.
  return String(left.base_ticker).localeCompare(String(right.base_ticker))
    || [left.leg1_exchange, left.leg2_exchange].sort().join("/").localeCompare(
      [right.leg1_exchange, right.leg2_exchange].sort().join("/"),
    )
    || Number(left.direction_code) - Number(right.direction_code)
    || String(left.pair_id).localeCompare(String(right.pair_id));
}

function rlStatusClass(status) {
  // Статус gate отделён от состояния самой recurrent policy.
  if (status === "ERROR" || status === "OFFLINE") return "error";
  if (status === "ENTER" || status === "HOLD" || status === "EXIT") return "live";
  if (status === "BLOCKED") return "blocked";
  if (status === "PAUSED") return "paused";
  return "waiting";
}

function warmupBlocks(progress) {
  const enabled = Math.round(Math.max(0, Math.min(1, Number(progress))) * 10);
  return `<span class="warmup" title="${Math.round(progress * 100)}% context ready">
    ${Array.from({ length: 10 }, (_, index) => `<i class="${index < enabled ? "on" : ""}"></i>`).join("")}
  </span>`;
}

function setConnection(connected, message = "") {
  state.connected = connected;
  const node = el("connection");
  node.className = `connection ${connected ? "online" : "offline"}`;
  node.querySelector("span").textContent = connected ? "LIVE FEED" : (message || "OFFLINE");
}

function renderSummary(data) {
  const summary = data.summary;
  el("sum-pairs").textContent = summary.monitored_pairs;
  el("sum-strategies").textContent = summary.active_strategies;
  el("sum-positions").textContent = summary.open_positions;
  el("sum-trades").textContent = summary.trades_in_memory;
  el("sum-store").textContent = String(data.paper_store.mode).toUpperCase();
  const pnl = el("sum-pnl");
  pnl.textContent = number(summary.total_net_pnl_bps, 2, " bps");
  pnl.parentElement.classList.remove("positive", "negative");
  const klass = pnlClass(summary.total_net_pnl_bps);
  if (klass) pnl.parentElement.classList.add(klass);
  el("session-clock").textContent = timeOf(data.server_ts);
  el("last-update").textContent = data.server_ts
    ? `virtual ${timeOf(data.server_ts)}`
    : "waiting for valid pair state";
}

function renderPairs(data) {
  // Повторная сортировка защищает UI даже при изменении порядка на backend.
  const pairs = [...data.pairs].sort(stablePairOrder);
  if (!state.selectedPairId && pairs.length) {
    state.selectedPairId = pairs[0].pair_id;
  }
  if (state.selectedPairId && !pairs.some((pair) => pair.pair_id === state.selectedPairId)) {
    state.selectedPairId = pairs[0]?.pair_id ?? null;
  }
  const rows = pairs.map((pair, index) => {
    const open = pair.strategies.some((strategy) => strategy.state === "open");
    const active = pair.strategies.filter((strategy) => strategy.active);
    const stateText = open
      ? "OPEN"
      : (active.find((strategy) => strategy.state === "pending") ? "PENDING" : "WATCH");
    const transformer = pair.warmup_progress < 1
      ? `WARM ${Math.round(pair.warmup_progress * 100)}%`
      : probability(pair.transformer_enter_probability);
    const rlGate = pair.rl_gate_active
      ? `GATE ${number(pair.rl_frozen_q35_bps, 1)}`
      : `≥ ${number(pair.rl_gate_threshold_bps, 0)}`;
    return `
      <tr data-pair-id="${safe(pair.pair_id)}" class="${pair.pair_id === state.selectedPairId ? "selected" : ""}">
        <td class="rank">${String(index + 1).padStart(2, "0")}</td>
        <td>
          <span class="pair-name">
            <strong>${safe(pair.base_ticker)}</strong>
            <small>${safe(pair.direction)}</small>
          </span>
        </td>
        <td class="metric bright">${number(pair.edge_bps, 1)}</td>
        <td class="metric">${number(pair.q35_watch_bps, 1)}</td>
        <td class="metric">${safe(transformer)}</td>
        <td class="metric">
          <span class="rl-status ${rlStatusClass(pair.rl_status)}"
                title="${safe(pair.rl_detail)}">
            <strong>${safe(pair.rl_status || actionLabel(pair.rl_action))}</strong>
            <small>${safe(rlGate)}</small>
          </span>
        </td>
        <td><span class="state-pill ${open ? "open" : ""}">${stateText}</span></td>
        <td>${warmupBlocks(pair.warmup_progress)}</td>
      </tr>`;
  }).join("");
  el("pair-rows").innerHTML = rows || `<tr><td colspan="8" class="empty">No registered pairs</td></tr>`;
  el("pair-rows").querySelectorAll("tr[data-pair-id]").forEach((row) => {
    row.addEventListener("click", () => {
      state.selectedPairId = row.dataset.pairId;
      state.series = null;
      renderPairs(state.overview);
      refreshSeries();
    });
  });
}

function renderModels(data) {
  el("model-list").innerHTML = data.models.map((model) => {
    const latency = Number(model.latency_p95_ms);
    const width = Number.isFinite(latency) ? Math.min(100, Math.max(3, latency / 0.5)) : 0;
    return `
      <article class="model-card" title="${safe(model.detail || "")}">
        <div class="model-card-top">
          <div>
            <strong>${safe(model.name)}</strong>
            <div class="strategy-models">${safe(model.kind)} · ${safe(model.device || "—")}</div>
          </div>
          <span class="model-state ${safe(model.state)}">${safe(model.state)}</span>
        </div>
        <div class="latency-row">
          <div class="latency-bar"><i style="width:${width}%"></i></div>
          <span>p95 ${number(model.latency_p95_ms, 2, " ms")}</span>
        </div>
        <div class="model-runtime">
          <span>${model.recent_inferences} recent inferences</span>
          <strong class="${model.recent_errors ? "negative" : ""}">
            ${model.recent_errors} errors
          </strong>
        </div>
      </article>`;
  }).join("");
}

function renderEvents(data) {
  el("event-count").textContent = `${data.events.length} EVENTS`;
  el("event-feed").innerHTML = data.events.map((event) => {
    const exit = event.action.includes("exit");
    return `
      <article class="event ${exit ? "exit" : ""}">
        <time>${timeOf(event.ts)} · ${safe(event.strategy)}</time>
        <div class="event-line">
          <strong>${safe(event.pair_id.split("_").slice(0, 2).join(" "))}</strong>
          <span class="event-action">${safe(event.action.replaceAll("_", " "))}</span>
        </div>
        <div class="event-reason">${safe(event.reason)}</div>
      </article>`;
  }).join("") || `<div class="empty">No strategy events yet</div>`;
}

async function toggleStrategy(name, active) {
  const action = active ? "pause" : "start";
  try {
    const response = await fetch(`${API}/v1/strategies/${encodeURIComponent(name)}/${action}`, {
      method: "POST",
    });
    if (!response.ok) throw new Error(await response.text());
    await refreshOverview();
  } catch (error) {
    setConnection(false, "CONTROL ERROR");
    console.error(error);
  }
}

function renderStrategies(data) {
  el("strategy-grid").innerHTML = data.strategies.map((strategy) => {
    const stats = strategy.stats;
    return `
      <article class="strategy-card">
        <div class="strategy-top">
          <div>
            <div class="strategy-name">${safe(strategy.name)}</div>
            <div class="strategy-models">${safe(strategy.models.join(" + ") || "rules only")}</div>
          </div>
          <button class="strategy-toggle ${strategy.active ? "active" : ""}"
                  data-strategy="${safe(strategy.name)}"
                  data-active="${strategy.active}">
            ${strategy.active ? "ACTIVE" : "PAUSED"}
          </button>
        </div>
        <div class="strategy-kpis">
          <div><span>TRADES</span><strong>${stats.trades}</strong></div>
          <div><span>WIN RATE</span><strong>${Math.round(stats.win_rate * 100)}%</strong></div>
          <div><span>NET</span><strong class="${pnlClass(stats.total_net_pnl_bps)}">${number(stats.total_net_pnl_bps, 1)}</strong></div>
          <div><span>MEAN</span><strong class="${pnlClass(stats.mean_trade_pnl_bps)}">${number(stats.mean_trade_pnl_bps, 1)}</strong></div>
          <div><span>HOLD</span><strong>${number(stats.mean_hold_seconds, 1, "s")}</strong></div>
          <div><span>DRAWDOWN</span><strong class="negative">${number(stats.max_drawdown_bps, 1)}</strong></div>
        </div>
      </article>`;
  }).join("");
  el("strategy-grid").querySelectorAll("button[data-strategy]").forEach((button) => {
    button.addEventListener("click", () => {
      toggleStrategy(button.dataset.strategy, button.dataset.active === "true");
    });
  });
}

function renderTrades(data) {
  el("trade-rows").innerHTML = data.trades.map((trade) => `
    <tr>
      <td>${timeOf(trade.closed_ts)}</td>
      <td>${safe(trade.pair_id.split("_").slice(0, 2).join(" "))}</td>
      <td>${safe(trade.strategy)}</td>
      <td>${duration(trade.hold_ms)}</td>
      <td><span class="positive">${number(trade.mfe_bps, 1)}</span> / <span class="negative">${number(trade.mae_bps, 1)}</span></td>
      <td>${safe(trade.exit_reason)}</td>
      <td class="${pnlClass(trade.net_pnl_bps)}">${number(trade.net_pnl_bps, 2, " bps")}</td>
    </tr>`).join("") || `<tr><td colspan="7" class="empty">No closed trades in this run</td></tr>`;
}

function selectedPair() {
  return state.overview?.pairs.find((pair) => pair.pair_id === state.selectedPairId);
}

function renderChartHeader() {
  const pair = selectedPair();
  if (!pair) {
    el("chart-title").textContent = "Choose a spread";
    el("chart-route").textContent = "The live edge will appear here.";
    el("chart-edge").textContent = "--";
    return;
  }
  el("chart-title").textContent = `${pair.base_ticker} · D${pair.direction_code}`;
  el("chart-route").textContent = pair.direction;
  el("chart-edge").textContent = number(pair.edge_bps, 2, " bps");
}

function drawChart() {
  // Canvas избегает тяжёлой chart-библиотеки и быстро перерисовывает 100-мс ряд.
  const canvas = el("spread-chart");
  const bounds = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(bounds.width * ratio));
  canvas.height = Math.max(1, Math.floor(bounds.height * ratio));
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const width = bounds.width;
  const height = bounds.height;
  ctx.clearRect(0, 0, width, height);

  const points = (state.series?.points || []).filter((point) => Number.isFinite(Number(point.edge_bps)));
  if (points.length < 2) {
    ctx.fillStyle = "#737a73";
    ctx.font = "11px monospace";
    ctx.fillText("Waiting for pair history…", 18, 30);
    return;
  }
  const padding = { left: 48, right: 18, top: 18, bottom: 25 };
  const values = points.map((point) => Number(point.edge_bps));
  let min = Math.min(...values, 0);
  let max = Math.max(...values, 0);
  const margin = Math.max(4, (max - min) * 0.12);
  min -= margin;
  max += margin;
  const start = Number(points[0].ts);
  const end = Number(points.at(-1).ts);
  const x = (ts) => padding.left + ((Number(ts) - start) / Math.max(1, end - start)) * (width - padding.left - padding.right);
  const y = (value) => padding.top + ((max - Number(value)) / Math.max(1e-9, max - min)) * (height - padding.top - padding.bottom);

  ctx.lineWidth = 1;
  ctx.strokeStyle = "#242824";
  ctx.fillStyle = "#747b74";
  ctx.font = "9px monospace";
  for (let index = 0; index <= 4; index += 1) {
    const value = min + ((max - min) * index) / 4;
    const yy = y(value);
    ctx.beginPath();
    ctx.moveTo(padding.left, yy);
    ctx.lineTo(width - padding.right, yy);
    ctx.stroke();
    ctx.fillText(`${value.toFixed(1)}`, 5, yy + 3);
  }
  if (min <= 0 && max >= 0) {
    ctx.strokeStyle = "#515751";
    ctx.setLineDash([4, 5]);
    ctx.beginPath();
    ctx.moveTo(padding.left, y(0));
    ctx.lineTo(width - padding.right, y(0));
    ctx.stroke();
    ctx.setLineDash([]);
  }

  const gradient = ctx.createLinearGradient(0, padding.top, 0, height - padding.bottom);
  gradient.addColorStop(0, "rgba(39, 228, 95, 0.24)");
  gradient.addColorStop(1, "rgba(39, 228, 95, 0)");
  ctx.beginPath();
  points.forEach((point, index) => {
    const xx = x(point.ts);
    const yy = y(point.edge_bps);
    if (index === 0) ctx.moveTo(xx, yy);
    else ctx.lineTo(xx, yy);
  });
  ctx.lineTo(x(points.at(-1).ts), height - padding.bottom);
  ctx.lineTo(x(points[0].ts), height - padding.bottom);
  ctx.closePath();
  ctx.fillStyle = gradient;
  ctx.fill();

  ctx.beginPath();
  points.forEach((point, index) => {
    if (index === 0) ctx.moveTo(x(point.ts), y(point.edge_bps));
    else ctx.lineTo(x(point.ts), y(point.edge_bps));
  });
  ctx.strokeStyle = "#27e45f";
  ctx.lineWidth = 1.6;
  ctx.stroke();

  for (const marker of state.series?.markers || []) {
    if (marker.ts < start || marker.ts > end) continue;
    const nearest = points.reduce((best, point) =>
      Math.abs(point.ts - marker.ts) < Math.abs(best.ts - marker.ts) ? point : best
    );
    const xx = x(marker.ts);
    const yy = y(nearest.edge_bps);
    ctx.save();
    ctx.translate(xx, yy);
    ctx.rotate(Math.PI / 4);
    ctx.fillStyle = marker.kind.startsWith("entry") ? "#f3bd32" : (marker.pnl_bps >= 0 ? "#51b7ff" : "#ff5362");
    ctx.fillRect(-4, -4, 8, 8);
    ctx.restore();
  }
}

function renderOverview(data) {
  state.overview = data;
  renderSummary(data);
  renderPairs(data);
  renderModels(data);
  renderEvents(data);
  renderStrategies(data);
  renderTrades(data);
  renderChartHeader();
  setConnection(true);
}

async function refreshOverview() {
  // Не запускаем второй overview-запрос, пока предыдущий ещё не завершён.
  if (state.busy) return;
  state.busy = true;
  try {
    const response = await fetch(`${API}/v1/monitor/overview`, { cache: "no-store" });
    if (!response.ok) throw new Error(`overview ${response.status}`);
    renderOverview(await response.json());
  } catch (error) {
    setConnection(false, "API OFFLINE");
    console.error(error);
  } finally {
    state.busy = false;
  }
}

async function refreshSeries() {
  if (!state.selectedPairId) {
    drawChart();
    return;
  }
  try {
    const response = await fetch(
      `${API}/v1/monitor/pairs/${encodeURIComponent(state.selectedPairId)}/series?limit=1800`,
      { cache: "no-store" },
    );
    if (!response.ok) throw new Error(`series ${response.status}`);
    state.series = await response.json();
    drawChart();
  } catch (error) {
    console.error(error);
  }
}

async function tick() {
  // setTimeout после await не накапливает параллельные polling-циклы.
  await refreshOverview();
  await refreshSeries();
  window.setTimeout(tick, REFRESH_MS);
}

window.addEventListener("resize", drawChart);
tick();
