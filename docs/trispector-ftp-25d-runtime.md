# TriSpector FTP 2.5D Runtime

This runbook starts the local system used by Studio and Operations for the
`mining_steel_ball_classification_25d` pipeline. It runs four long-lived
processes:

- FastAPI backend
- Vite frontend
- TriSpector FTP listener
- 2.5D processing worker

The FTP runtime listens for sensor uploads, registers each stable upload as a
take under `data/incoming/`, and the worker processes those takes with the
native 2.5D classification pipeline.

## One-Time Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd frontend
npm install
cd ..
```

Optional env files:

```bash
cp .env.example .env
cp frontend/.env.example frontend/.env
```

Current local config in your env files:

- API: `http://localhost:8380`
- Frontend: `http://localhost:5074`
- FTP: port `2121`
- FTP auth: anonymous
- FTP upload directory: `data/acquisition/trispector_ftp/incoming`

## Start Everything In Screen

Use the launcher:

```bash
scripts/start_25d_ftp_screens.sh start
```

The script creates these detached `screen` sessions:

- `sensor_api`
- `sensor_frontend`
- `sensor_ftp`
- `sensor_25d_worker`

Check status:

```bash
scripts/start_25d_ftp_screens.sh status
```

Attach to a process:

```bash
screen -r sensor_api
screen -r sensor_frontend
screen -r sensor_ftp
screen -r sensor_25d_worker
```

Detach from a screen session with `Ctrl-a` then `d`.

Stop everything:

```bash
scripts/start_25d_ftp_screens.sh stop
```

Restart everything:

```bash
scripts/start_25d_ftp_screens.sh restart
```

## Manual Commands

If you do not want to use `screen`, run each command in its own terminal.

Backend API:

```bash
source .venv/bin/activate
python scripts/run_api.py
```

Frontend:

```bash
cd frontend
npm run dev
```

FTP listener:

```bash
source .venv/bin/activate
python scripts/runtime.py start trispector_ftp --foreground
```

2.5D worker:

```bash
source .venv/bin/activate
python scripts/run_25d_worker.py \
  --data-dir data \
  --pipeline-id mining_steel_ball_classification_25d \
  --poll-interval-sec 1.0
```

## SOPAS / Sensor Upload Settings

Configure the TriSpector or SOPAS FTP destination to upload to this machine:

```text
Host: <this Mac's IP address>
Port: 2121
Username: anonymous
Password: empty
Upload mode: binary
```

Uploaded files should be image files supported by the TriSpector parser, such
as `.png`, `.tif`, `.tiff`, or `.bmp`.

## Checks

Backend health:

```bash
curl http://localhost:8380/api/health
```

FTP runtime self-test:

```bash
python scripts/runtime.py selftest trispector_ftp
```

Runtime status:

```bash
python scripts/runtime.py list
python scripts/runtime.py logs trispector_ftp --limit 100
curl http://localhost:8380/api/runtime/processes/trispector_ftp/ftp-status
```

Open the UI:

- `http://localhost:5074/studio`
- `http://localhost:5074/operations`
- `http://localhost:5074/superclass-hist`

## Notes

Use the supervised FTP runtime for the real FTP server path:

```bash
python scripts/runtime.py start trispector_ftp --foreground
```

Do not run it at the same time as `scripts/watch_trispector_folder.py` for the
same upload flow. The watcher is useful only when another process is already
writing files into a folder and you do not need this repo to host the FTP server.
