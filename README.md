# SENTRA
**Security Enhancement for Network Township Residential AI**

Existing township CCTV, turned into a guard that double-checks before it raises an alarm.
OpenCV 5 watches the cameras, an AI agent on Amazon Bedrock re-examines every alert with OpenCV tools,
and a human guard approves before any resident is alerted.

Team Codex, Universiti Teknologi PETRONAS · OpenCV AI Competition 2026, powered by AWS · Agentic Vision path

| | |
|---|---|
| Live web endpoint (AWS) | https://13-239-96-244.sslip.io |
| Demo video (≤ 5 min) | _link added at submission_ |
| Technical report | [docs/REPORT.md](docs/REPORT.md) |
| Judge login | guard and resident accounts are listed in the private submission notes |

![Architecture](docs/architecture.png)

## What it does
1. **Watch (OpenCV 5).** YOLOv8n runs in `cv.dnn` (OpenCV 5 new DNN engine, dynamic-shape ONNX). A small IoU tracker follows people and vehicles, and zone rules (`cv.pointPolygonTest`) fire on perimeter breach, loitering, vehicle dwell and Away Mode breach. Camera health checks catch blocked, frozen and offline cameras.
2. **Protect privacy first.** Faces are found with OpenCV's YuNet (`cv.FaceDetectorYN`) and pixelated, together with the head area of every person, before any frame is saved or shown. No facial recognition is used anywhere.
3. **Re-check with an agent (Amazon Bedrock).** Every rule hit opens a case. Claude Haiku 4.5 calls OpenCV 5 tools (`recheck_region`: crop, upscale, CLAHE, re-detect, Farneback optical flow) plus context tools, looks again when the evidence is weak, and then either **requests human approval** or **logs** the event. Guardrails in code stop it from logging a HIGH alert when OpenCV still sees a person.
4. **Human decides.** The guard gets a live pop-up with the clip, the agent's summary and its step-by-step trace, and presses **Approve alarm** or **Dismiss**. Only approved alerts reach the resident.

![Agent workflow](docs/agent_workflow.png)

## OpenCV 5 in SENTRA
| Where | OpenCV 5 API |
|---|---|
| Detection | `cv.dnn.readNetFromONNX` (new engine, dynamic input size), `blobFromImage`, `NMSBoxesBatched` |
| Zones | `cv.pointPolygonTest`, `fillPoly`, `polylines` |
| Privacy | `cv.FaceDetectorYN` (YuNet), pixelation with `resize` |
| Agent tool `recheck_region` | crop + `resize(INTER_CUBIC)`, `createCLAHE` on LAB, `cv.dnn` re-detection, `calcOpticalFlowFarneback`, HSV/colour checks for low light and infrared |
| Camera health | `Laplacian` variance (blocked lens), frame difference (frozen), brightness |
| Video | `VideoCapture` (files, webcams, RTSP), `VideoWriter`, drawing overlays |

## AWS in SENTRA
| Service | Use |
|---|---|
| Amazon EC2 (`m7i-flex.large`, Ubuntu 24.04) | runs the cameras, the agent and the web app as systemd services; HTTPS through Caddy |
| Amazon Bedrock | the agent: Converse API with tool use, `au.anthropic.claude-haiku-4-5` (Australian inference profile), Amazon Nova Lite as fallback |
| Amazon S3 | blurred snapshots and 10-second clips, server-side encrypted, deleted after 7 days by a lifecycle rule, served by short-lived presigned links |
| Amazon CloudWatch | custom metrics in namespace `SENTRA`: RuleFired, AgentDecision, AgentLatency, GuardrailOverride, GuardDecision, GuardResponseTime, CameraFPS |
| AWS IAM | instance role with a least-privilege policy ([deploy/iam-policy.json](deploy/iam-policy.json)); no access keys on the server |

## Results
| Test | Result |
|---|---|
| OpenCV 5 `cv.dnn` vs original PyTorch, COCO128 person AP50 | **0.768** vs 0.766 (no accuracy lost) |
| Agreement with PyTorch on 150 CCTV frames | F1 0.93, matched-box IoU 0.98 |
| Speed, 2 CPU cores | 103 ms per frame (9.7 FPS) vs 78 ms for PyTorch |
| Agent vs rules only, 11 labelled cases | false alarms to the guard **4 → 0**, real events kept **7 / 7**, median decision 11.2 s |

Method, tables and failure cases: [docs/REPORT.md](docs/REPORT.md).
Raw outputs: [eval/detector_results.json](eval/detector_results.json), [eval/agent_results_bedrock.json](eval/agent_results_bedrock.json),
[eval/bedrock_traces.jsonl](eval/bedrock_traces.jsonl), [eval/agent_results_rules_local.json](eval/agent_results_rules_local.json).

## Run it on a laptop (Windows, macOS or Linux, Python 3.12+)
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt   # pinned versions, OpenCV 5.0.0
python scripts/export_onnx.py --check
```
Put demo videos in `samples/` (see **Data** below), then:
```bash
python scripts/run_console.py                          # terminal 1: web app on http://127.0.0.1:8000
python scripts/run_cameras.py --cameras gate porch door # terminal 2: cameras + agent
```
Create a **Guard** account (invite code `codex-guard` locally, or set `SENTRA_GUARD_CODE`) and a **Resident** account (house `12`).
The agent uses Amazon Bedrock when AWS credentials are available (`aws configure`, region `ap-southeast-2`);
without them it falls back to the local policy automatically, or run with `--agent local`.

| Script | Purpose |
|---|---|
| `scripts/run_sentra.py --camera porch --config config/showcase.yaml --show` | one camera, with a preview window |
| `scripts/run_cameras.py` | several cameras, looped like live CCTV |
| `scripts/run_console.py` | web app (`--lan`, `--tunnel`, `--host`, `--port`) |
| `scripts/draw_zones.py` | draw zones on your own footage |
| `scripts/eval_detector.py` | evaluation 1: OpenCV 5 vs PyTorch accuracy, agreement and speed |
| `scripts/eval_agent.py` | evaluation 2: rules only vs local policy vs Bedrock agent |
| `scripts/export_onnx.py` | re-create the ONNX model (needs `pip install ultralytics onnx` once) |

## Deploy on AWS
1. Create an S3 bucket, add the 7-day lifecycle rule and upload the demo videos:
   ```bash
   aws s3 mb s3://<bucket> --region ap-southeast-2
   aws s3api put-bucket-lifecycle-configuration --bucket <bucket> --lifecycle-configuration file://deploy/s3-lifecycle.json
   aws s3 cp samples s3://<bucket>/samples/ --recursive
   ```
2. Create an IAM policy from [deploy/iam-policy.json](deploy/iam-policy.json) (replace `BUCKET_NAME`) and an EC2 role `sentra-ec2-role` with it.
3. Launch Ubuntu 24.04 on `m7i-flex.large` with that role, 20 GB disk, ports 22/80/443 open.
4. In EC2 Instance Connect:
   ```bash
   curl -sSL https://raw.githubusercontent.com/farah-aleya/SENTRA_CodeX/main/deploy/setup_ec2.sh -o setup.sh
   sudo bash setup.sh <bucket>
   ```
   It installs everything, starts `sentra-web`, `sentra-cameras` and Caddy, and prints the HTTPS address and a random guard invite code.
   Logs: `journalctl -u sentra-cameras -f`. Settings: `/etc/sentra.env`.

## Test
```bash
python scripts/eval_detector.py --images datasets/coco128 --videos samples   # needs: pip install ultralytics
python scripts/eval_agent.py --modes rules local bedrock                      # bedrock needs AWS credentials
```

## Responsible use
- **No facial recognition.** SENTRA only detects the classes person and vehicle. Faces and heads are pixelated before storage.
- **Human in control.** The agent can only log or ask; nothing reaches a resident without a guard's approval. Guardrails in code override the model for HIGH alerts.
- **Data minimisation.** Bedrock receives only numbers and labels (counts, confidences, zone names), never images. Raw frames for the re-check stay in memory for a few seconds.
- **Retention.** Evidence in S3 is encrypted and deleted after 7 days.
- **Access control.** Separate guard and resident roles, hashed passwords (PBKDF2), HttpOnly session cookies, clips served only to users allowed to see that alert, HTTPS.
- **Transparency.** Every agent decision is stored with its full tool trace and shown to the guard.

## Data
- Demo and evaluation clips are short, publicly posted CCTV and doorbell videos used for **non-commercial testing only**. They are not stored in this repository (`samples/` and `dataset/` are git-ignored), all rights remain with their owners, and faces are pixelated in every SENTRA output. They can be replaced with any footage by editing `config/*.yaml`.
- COCO128 (CC BY 4.0) is used for the detector accuracy test.
- No data from real township residents is used.

## Team Codex
| Name | Role |
|---|---|
| Nur Farah Aleya Binti Mohamad Noor'Azlan | Team representative, AI and computer vision, AWS |
| Ilhan Zikri Bin Zul Azlin | OpenCV pipeline, research |
| Anuar Afiq Bin Arfahairy | Evaluation, project management |
| Siti Zafirah Binti Zailan | Product, user experience, video |

## Third-party components
OpenCV 5 (Apache-2.0) · YOLOv8n weights by Ultralytics (AGPL-3.0) · YuNet face detector from OpenCV Zoo (MIT) ·
FastAPI, Uvicorn, boto3, NumPy (permissive licences) · Caddy (Apache-2.0).
SENTRA is a student prototype and not an official product of any company.
