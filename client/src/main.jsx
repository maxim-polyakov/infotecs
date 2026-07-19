import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";

import { getJson, postJson } from "./api";
import "./styles.css";

function App() {
  const [health, setHealth] = useState("checking");
  const [modelStatus, setModelStatus] = useState(null);
  const [reports, setReports] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function refresh() {
    try {
      const [healthData, modelData, reportData, jobData] = await Promise.all([
        getJson("/api/health"),
        getJson("/api/model/status"),
        getJson("/api/reports"),
        getJson("/api/jobs"),
      ]);
      setHealth(healthData.status);
      setModelStatus(modelData);
      setReports(reportData.reports ?? []);
      setJobs(jobData.jobs ?? []);
      setError("");
    } catch (err) {
      setHealth("offline");
      setError(err.message);
    }
  }

  async function runAction(label, action) {
    setBusy(true);
    setMessage("");
    setError("");
    try {
      const result = await action();
      setMessage(`${label}: ${summarize(result)}`);
      await refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, 5000);
    return () => window.clearInterval(timer);
  }, []);

  const latestReports = [...reports].reverse().slice(0, 5);

  return (
    <main className="page">
      <header className="hero">
        <div>
          <p className="eyebrow">UEBA Prototype</p>
          <h1>Мониторинг аномальной активности ПК</h1>
          <p>
            React-клиент управляет FastAPI backend: сбором телеметрии, обучением модели,
            запуском детектора и просмотром отчетов.
          </p>
        </div>
        <button onClick={refresh} disabled={busy}>Обновить</button>
      </header>

      {error && <div className="alert error">{error}</div>}
      {message && <div className="alert success">{message}</div>}

      <section className="grid">
        <Card title="API">
          <Metric label="Статус" value={health} />
        </Card>
        <Card title="Модель">
          <Metric label="Наличие" value={modelStatus?.exists ? "обучена" : "нет модели"} />
          <Metric label="Порог" value={formatNumber(modelStatus?.threshold)} />
          <Metric label="Признаков" value={modelStatus?.feature_count ?? "-"} />
        </Card>
        <Card title="Отчеты">
          <Metric label="Аномалий" value={reports.length} />
        </Card>
      </section>

      <section className="panel">
        <h2>Действия</h2>
        <div className="actions">
          <button disabled={busy} onClick={() => runAction("Demo", () => postJson("/api/demo", { output_dir: "reports/demo" }))}>
            Запустить demo
          </button>
          <button disabled={busy} onClick={() => runAction("Сбор", () => postJson("/api/collect", { duration_hours: 0.01, interval_seconds: 2, output_path: "data/raw.csv" }))}>
            Быстрый сбор
          </button>
          <button disabled={busy} onClick={() => runAction("Обучение", () => postJson("/api/train", { data_path: "data/raw.csv", model_dir: "models/default", epochs: 200 }))}>
            Обучить модель
          </button>
          <button disabled={busy} onClick={() => runAction("Проверка", () => postJson("/api/detect/once", { model_dir: "models/default", report_path: "reports/anomalies.jsonl" }))}>
            Проверить текущее окно
          </button>
          <button disabled={busy} onClick={() => runAction("Детектор", () => postJson("/api/detect/start", { model_dir: "models/default", report_path: "reports/anomalies.jsonl", interval_seconds: 5, max_samples: 12 }))}>
            Запустить детектор
          </button>
        </div>
        <p className="hint">
          Для итоговой обучающей выборки используйте сбор не менее 24 часов через backend API или CLI.
        </p>
      </section>

      <section className="grid two">
        <Card title="Последние jobs">
          <div className="list">
            {jobs.length === 0 && <p className="muted">Пока нет задач.</p>}
            {[...jobs].reverse().slice(0, 6).map((job) => (
              <div className="row" key={job.id}>
                <span>{job.name}</span>
                <strong>{job.status}</strong>
              </div>
            ))}
          </div>
        </Card>

        <Card title="Последние аномалии">
          <div className="list">
            {latestReports.length === 0 && <p className="muted">Аномалии еще не обнаружены.</p>}
            {latestReports.map((report, index) => (
              <article className="report" key={`${report.timestamp}-${index}`}>
                <div className="row">
                  <span>Score {formatNumber(report.anomaly_score)}</span>
                  <strong>Threshold {formatNumber(report.threshold)}</strong>
                </div>
                <p>{report.explanations?.[0] ?? "behavior differs from normal profile"}</p>
                <small>{(report.top_features ?? []).map((item) => item.feature).join(", ")}</small>
              </article>
            ))}
          </div>
        </Card>
      </section>
    </main>
  );
}

function Card({ title, children }) {
  return (
    <section className="card">
      <h2>{title}</h2>
      {children}
    </section>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function summarize(result) {
  if (result.detected_count !== undefined) return `обнаружено ${result.detected_count} demo-аномалий`;
  if (result.threshold !== undefined) return `порог ${formatNumber(result.threshold)}`;
  if (result.id) return `job ${result.status}`;
  if (result.is_anomaly !== undefined) return result.is_anomaly ? "аномалия обнаружена" : "аномалий нет";
  return "готово";
}

function formatNumber(value) {
  if (value === undefined || value === null) return "-";
  if (typeof value !== "number") return value;
  return value.toFixed(4);
}

createRoot(document.getElementById("root")).render(<App />);
