"""SENTRA on AWS: evidence storage in Amazon S3 and metrics in Amazon CloudWatch.

Both are optional and switched on with environment variables, so the same code runs
on a laptop (no AWS) and on EC2 (credentials come from the instance IAM role):

  SENTRA_S3_BUCKET=sentra-evidence-xxxx   upload blurred snapshots and 10 s clips to S3
                                           (the bucket has a 7-day lifecycle rule, see deploy/)
  SENTRA_CLOUDWATCH=1                      publish metrics to the CloudWatch namespace "SENTRA"
  AWS_REGION=ap-southeast-2

Uploads and metrics run in a background thread and never slow down detection.
Only blurred media is ever uploaded: the privacy filter runs before frames are saved.
"""
import os
import queue
import threading
import time
from pathlib import Path

REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "ap-southeast-2"
BUCKET = os.environ.get("SENTRA_S3_BUCKET", "").strip()
METRICS = os.environ.get("SENTRA_CLOUDWATCH", "").strip() in ("1", "true", "yes")
NAMESPACE = "SENTRA"

_q: "queue.Queue" = queue.Queue(maxsize=1000)
_clients = {}
_warned = set()


def enabled() -> bool:
    return bool(BUCKET or METRICS)


def _client(name):
    if name not in _clients:
        import boto3
        _clients[name] = boto3.client(name, region_name=REGION)
    return _clients[name]


def _warn_once(key, msg):
    if key not in _warned:
        _warned.add(key)
        print(f"[cloud] {msg}", flush=True)


def s3_key(local_path: str) -> str:
    p = str(local_path).replace("\\", "/")
    return p[len("outputs/"):] if p.startswith("outputs/") else p


def upload(local_path: str):
    """Queue a blurred snapshot or clip for upload to S3 (no-op when no bucket is set)."""
    if BUCKET and local_path:
        _put(("s3", str(local_path)))


def metric(name: str, value: float = 1.0, unit: str = "Count", **dims):
    """Queue one CloudWatch metric data point (no-op unless SENTRA_CLOUDWATCH=1)."""
    if METRICS:
        _put(("cw", name, float(value), unit, {k: str(v) for k, v in dims.items() if v not in (None, "")}))


def presigned_url(local_path: str, seconds: int = 300):
    """Short-lived S3 link for media that is no longer on local disk."""
    if not BUCKET:
        return None
    try:
        return _client("s3").generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": s3_key(local_path)},
                                                    ExpiresIn=seconds)
    except Exception as e:                                                  # noqa: BLE001
        _warn_once("presign", f"cannot sign S3 URL: {e}")
        return None


def _put(item):
    try:
        _q.put_nowait(item)
    except queue.Full:
        _warn_once("full", "upload queue full, dropping items")


def _send(item):
    if item[0] == "s3":
        path = item[1]
        for _ in range(20):                                   # clips are written in the background: wait for them
            if Path(path).exists() and Path(path).stat().st_size > 0:
                break
            time.sleep(0.5)
        ctype = "video/mp4" if path.endswith(".mp4") else "image/jpeg"
        _client("s3").upload_file(path, BUCKET, s3_key(path),
                                  ExtraArgs={"ContentType": ctype, "ServerSideEncryption": "AES256"})
    else:
        _, name, value, unit, dims = item
        _client("cloudwatch").put_metric_data(Namespace=NAMESPACE, MetricData=[{
            "MetricName": name, "Value": value, "Unit": unit,
            "Dimensions": [{"Name": k, "Value": v} for k, v in dims.items()]}])


def _worker():
    while True:
        item = _q.get()
        try:
            _send(item)
        except Exception as e:                                              # noqa: BLE001
            _warn_once(item[0], f"{'S3 upload' if item[0] == 's3' else 'CloudWatch'} failed: {type(e).__name__}: {e}")


def flush(timeout=20):
    """Wait for queued uploads/metrics (call before the program exits)."""
    end = time.time() + timeout
    while not _q.empty() and time.time() < end:
        time.sleep(0.2)


if enabled():
    threading.Thread(target=_worker, daemon=True).start()
    print(f"[cloud] AWS on: S3 bucket={BUCKET or '-'} | CloudWatch metrics={'on' if METRICS else 'off'} | region={REGION}",
          flush=True)
