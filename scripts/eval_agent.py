"""Evaluation 2: does the agent reduce false alarms without missing real events?

Runs every camera in config/eval.yaml once per mode and compares:
  rules    the original SENTRA: every rule hit goes straight to the guard (--agent off)
  local    the offline policy: same OpenCV re-check tool, fixed thresholds (--agent local)
  bedrock  the agent: Claude Haiku 4.5 on Amazon Bedrock with OpenCV 5 tools (--agent bedrock)

Ground truth (real / benign) for each camera + rule is in eval/labels.yaml.

  python scripts/eval_agent.py --modes rules local bedrock
Writes eval/agent_results.json and eval/agent_cases.csv, and prints a Markdown table.
"""
import argparse
import csv
import json
import os
import sqlite3
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
AGENT_RULES = {"PERIMETER_BREACH", "AWAY_BREACH", "LOITERING", "VEHICLE_DWELL"}
FLAG = {"rules": "off", "local": "local", "bedrock": "bedrock"}


def run_mode(mode, cams, config):
    db = ROOT / "eval" / f"run_{mode}.db"
    if db.exists():
        db.unlink()
    env = {**os.environ, "SENTRA_DB": str(db), "PYTHONUNBUFFERED": "1"}
    env.pop("SENTRA_S3_BUCKET", None); env.pop("SENTRA_CLOUDWATCH", None)     # keep evaluation runs out of production
    for cam in cams:
        t = time.time()
        print(f"  [{mode}] camera {cam} ...", flush=True)
        subprocess.run([sys.executable, "scripts/run_sentra.py", "--camera", cam, "--config", config, "--agent", FLAG[mode],
                        "--no-blur"], cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  [{mode}] camera {cam} done in {time.time() - t:.0f}s", flush=True)
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    return [dict(r) for r in con.execute("SELECT * FROM events ORDER BY camera_id, video_time")]


def summarise(mode, events, labels):
    rows = []
    for e in events:
        if e["rule"] not in AGENT_RULES:
            continue                                            # camera-health alerts do not go through the agent
        label = (labels.get(e["camera_id"]) or {}).get(e["rule"], "unlabelled")
        trace = json.loads(e.get("agent_trace") or "[]") if e.get("agent_trace") else []
        calls = [s for s in trace if s.get("type") == "tool_call"]
        rechecks = [s for s in trace if s.get("type") == "tool_result" and s.get("name") == "recheck_region"]
        to_guard = e["status"] == "PENDING" if mode != "rules" else True
        rows.append({
            "mode": mode, "id": e["id"], "camera": e["camera_id"], "rule": e["rule"], "video_time": e["video_time"],
            "label": label, "sent_to_guard": to_guard, "status": e["status"],
            "decided_by": e.get("agent_decided_by") or ("rules" if mode == "rules" else ""),
            "agent_ms": int(e["agent_ms"]) if (e.get("agent_ms") or "").isdigit() else None,
            "tool_calls": len(calls), "rechecks": len(rechecks),
            "enhanced_recheck": any((s.get("output") or {}).get("enhanced") for s in rechecks),
            "guardrail": any(s.get("type") == "guardrail" for s in trace),
            "summary": (e.get("agent_summary") or e["message"])[:160],
        })
    real = [r for r in rows if r["label"] == "real"]
    benign = [r for r in rows if r["label"] == "benign"]
    sent = [r for r in rows if r["sent_to_guard"]]
    ms = [r["agent_ms"] for r in rows if r["agent_ms"] is not None]
    return rows, {
        "mode": mode,
        "cases": len(rows), "real": len(real), "benign": len(benign),
        "sent_to_guard": len(sent),
        "real_sent_to_guard": sum(r["sent_to_guard"] for r in real),
        "benign_sent_to_guard (false alarms)": sum(r["sent_to_guard"] for r in benign),
        "missed_real": sum(not r["sent_to_guard"] for r in real),
        "recall_real": round(sum(r["sent_to_guard"] for r in real) / len(real), 3) if real else None,
        "false_alarm_reduction": round(1 - sum(r["sent_to_guard"] for r in benign) / len(benign), 3) if benign else None,
        "guard_queue_precision": round(sum(r["sent_to_guard"] for r in real) / len(sent), 3) if sent else None,
        "median_agent_ms": int(statistics.median(ms)) if ms else None,
        "mean_tool_calls": round(statistics.mean(r["tool_calls"] for r in rows), 2) if rows and mode != "rules" else None,
        "cases_with_second_recheck": sum(r["rechecks"] >= 2 for r in rows),
        "guardrail_overrides": sum(r["guardrail"] for r in rows),
        "decided_by": {k: sum(r["decided_by"] == k for r in rows) for k in sorted({r["decided_by"] for r in rows})},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/eval.yaml")
    ap.add_argument("--labels", default="eval/labels.yaml")
    ap.add_argument("--modes", nargs="+", default=["rules", "local", "bedrock"], choices=list(FLAG))
    ap.add_argument("--cameras", nargs="*")
    args = ap.parse_args()

    os.chdir(ROOT)
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    labels = yaml.safe_load(open(args.labels, encoding="utf-8"))
    cams = args.cameras or list(cfg["cameras"])
    all_rows, summary = [], []
    for mode in args.modes:
        rows, s = summarise(mode, run_mode(mode, cams, args.config), labels)
        all_rows += rows; summary.append(s)

    out = ROOT / "eval"
    (out / "agent_results.json").write_text(json.dumps({"summary": summary, "cases": all_rows}, indent=2))
    with open(out / "agent_cases.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()) if all_rows else ["mode"])
        w.writeheader(); w.writerows(all_rows)
    keys = ["cases", "real", "benign", "sent_to_guard", "real_sent_to_guard", "benign_sent_to_guard (false alarms)",
            "missed_real", "recall_real", "false_alarm_reduction", "guard_queue_precision", "median_agent_ms",
            "mean_tool_calls", "cases_with_second_recheck", "guardrail_overrides"]
    print("\n| metric | " + " | ".join(s["mode"] for s in summary) + " |")
    print("|---|" + "---|" * len(summary))
    for k in keys:
        print(f"| {k} | " + " | ".join(str(s.get(k)) for s in summary) + " |")
    print("\nPer case:")
    for r in all_rows:
        print(f"  {r['mode']:<8} {r['camera']:<10} {r['rule']:<17} t={r['video_time']:>5}  {r['label']:<7} "
              f"-> {'GUARD' if r['sent_to_guard'] else 'logged'}  ({r['decided_by']}, {r['agent_ms']} ms, "
              f"{r['tool_calls']} tools, {r['rechecks']} rechecks{', guardrail' if r['guardrail'] else ''})")
