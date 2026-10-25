#!/usr/bin/env bash
# SENTRA one-command setup for an Ubuntu 24.04 EC2 instance (run in EC2 Instance Connect):
#
#   curl -sSL https://raw.githubusercontent.com/farah-aleya/SENTRA_CodeX/main/deploy/setup_ec2.sh -o setup.sh
#   sudo bash setup.sh <your-s3-bucket-name>
#
# What it does
#   1. installs Python, ffmpeg and Caddy (HTTPS reverse proxy with a free Let's Encrypt certificate)
#   2. clones SENTRA to /opt/sentra and installs the pinned Python packages
#   3. downloads the demo videos from s3://<bucket>/samples/ (uploaded from the laptop)
#   4. writes /etc/sentra.env (S3 bucket, CloudWatch on, random guard code and session secret)
#   5. starts two systemd services: sentra-web (the website) and sentra-cameras (OpenCV 5 + agent)
#   6. serves the site at https://<public-ip-with-dashes>.sslip.io
# AWS credentials come from the instance IAM role. No access keys are stored on the server.
set -euo pipefail

BUCKET="${1:?usage: sudo bash setup.sh <s3-bucket-name>}"
REPO="https://github.com/farah-aleya/SENTRA_CodeX.git"
APP=/opt/sentra
REGION=ap-southeast-2
CAMERAS="${CAMERAS:-gate porch door}"

echo "==> 1/6 system packages"
apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-pip git ffmpeg libgl1 libglib2.0-0t64 caddy curl

echo "==> 2/6 SENTRA code"
if [ -d "$APP/.git" ]; then git -C "$APP" pull --ff-only; else git clone --depth 1 "$REPO" "$APP"; fi
chown -R ubuntu:ubuntu "$APP"
sudo -u ubuntu python3 -m venv "$APP/.venv"
sudo -u ubuntu "$APP/.venv/bin/pip" install --upgrade pip -q
sudo -u ubuntu "$APP/.venv/bin/pip" install -r "$APP/requirements.txt" -q

echo "==> 3/6 demo videos from s3://$BUCKET/samples/"
sudo -u ubuntu AWS_REGION=$REGION "$APP/.venv/bin/python" - "$BUCKET" "$APP/samples" <<'PY'
import sys, pathlib, boto3
bucket, dest = sys.argv[1], pathlib.Path(sys.argv[2]); dest.mkdir(parents=True, exist_ok=True)
s3 = boto3.client("s3")
n = 0
for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="samples/"):
    for o in page.get("Contents", []):
        name = o["Key"].split("/", 1)[1]
        if name:
            s3.download_file(bucket, o["Key"], str(dest / name)); n += 1
print(f"downloaded {n} videos")
PY

echo "==> 4/6 settings (/etc/sentra.env)"
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
IP=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/public-ipv4)
DOMAIN="${IP//./-}.sslip.io"
if [ ! -f /etc/sentra.env ]; then
  cat > /etc/sentra.env <<ENV
AWS_REGION=$REGION
SENTRA_S3_BUCKET=$BUCKET
SENTRA_CLOUDWATCH=1
SENTRA_GUARD_CODE=guard-$(openssl rand -hex 3)
SENTRA_SECRET=$(openssl rand -hex 32)
SENTRA_DOMAIN=$DOMAIN
PYTHONUNBUFFERED=1
ENV
  chmod 600 /etc/sentra.env
else
  sed -i "s/^SENTRA_DOMAIN=.*/SENTRA_DOMAIN=$DOMAIN/" /etc/sentra.env
fi

echo "==> 5/6 services"
cat > /etc/systemd/system/sentra-web.service <<UNIT
[Unit]
Description=SENTRA web app
After=network-online.target
[Service]
User=ubuntu
WorkingDirectory=$APP
EnvironmentFile=/etc/sentra.env
ExecStart=$APP/.venv/bin/python scripts/run_console.py --host 127.0.0.1 --port 8000 --no-browser
Restart=always
RestartSec=3
[Install]
WantedBy=multi-user.target
UNIT
cat > /etc/systemd/system/sentra-cameras.service <<UNIT
[Unit]
Description=SENTRA cameras (OpenCV 5 detection, privacy blur, Bedrock agent)
After=sentra-web.service
[Service]
User=ubuntu
WorkingDirectory=$APP
EnvironmentFile=/etc/sentra.env
ExecStart=$APP/.venv/bin/python scripts/run_cameras.py --config config/showcase.yaml --cameras $CAMERAS
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
UNIT
cat > /etc/caddy/Caddyfile <<CADDY
$DOMAIN {
    encode gzip
    reverse_proxy 127.0.0.1:8000 {
        flush_interval -1
    }
    header {
        Strict-Transport-Security "max-age=31536000"
        X-Content-Type-Options nosniff
        Referrer-Policy same-origin
    }
}
CADDY
systemctl daemon-reload
systemctl enable --now sentra-web sentra-cameras
systemctl restart caddy

echo "==> 6/6 done"
sleep 5
systemctl --no-pager --lines=0 status sentra-web sentra-cameras caddy | grep -E "Active|●" || true
echo
echo "  Website:          https://$DOMAIN   (the HTTPS certificate can take up to a minute the first time)"
echo "  Guard invite code: $(grep SENTRA_GUARD_CODE /etc/sentra.env | cut -d= -f2)"
echo "  Logs:             journalctl -u sentra-cameras -f      journalctl -u sentra-web -f"
