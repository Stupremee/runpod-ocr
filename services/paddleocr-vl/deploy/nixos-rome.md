# Deploy paddleocr-vl on rome (instructions for the nix-repo agent)

Goal: run the PaddleOCR-VL OCR service on `rome` from the `Stupremee/nix` flake
(`/home/stu/dev/nix`), as a `my.paddleocr-vl` module in the same style as
`modules/nixos/cliproxyapi`. The service updates itself: a timer pulls the
moving `:main` image tag and restarts the container when it changes.

## What you are deploying

- **Image:** `ghcr.io/stupremee/paddleocr-vl:main`, public, no registry login needed.
  Built by `.github/workflows/paddleocr-vl-image.yml` in `Stupremee/runpod-ocr` on every
  push to `main`. Each build is also tagged `sha-<commit>` if you ever want to pin one.
- **Container:** listens on `8080`, has a Docker `HEALTHCHECK` on `/health`, keeps its
  state (SQLite job queue) in `/data`, and runs as root inside the container.
  Layout detection runs on CPU. It needs about 2 to 8 GB of RAM depending on
  `WORKER_CONCURRENCY` (4 by default).
- **GPU inference:** happens remotely on a Runpod serverless endpoint, so the
  container needs outbound HTTPS only. No NVIDIA setup on rome.
- **API:** see `services/paddleocr-vl/README.md` in `Stupremee/runpod-ocr`
  (async `/jobs`, `/batches`, webhooks).

## Environment

| Var | Source | Notes |
| --- | --- | --- |
| `RUNPOD_API_KEY` | secret | Runpod API key |
| `PADDLEOCR_VL_ENDPOINT_ID` | secret file (not sensitive, but keep it with the key) | currently `4d2c541wzvlrzp` |
| `WEBHOOK_SECRET` | secret | `whsec_<base64>`, signs outgoing webhooks |
| `OCR_BACKEND` | module option | `runpod`, or `emulated` for a GPU-free smoke test |
| `WORKER_CONCURRENCY` | module option | jobs processed at once |

Stu has the current secret values in `/home/stu/dev/runpod/.env`. Never copy
them into the nix repo in plain text or print them in logs.

## Steps

### 1. Secret (needs Stu's YubiKey)

The repo uses agenix-rekey with a YubiKey master identity (`modules/nixos/secrets`).
Create `secrets/paddleocr-vl.env.age` holding a dotenv file:

```
RUNPOD_API_KEY=...
PADDLEOCR_VL_ENDPOINT_ID=4d2c541wzvlrzp
WEBHOOK_SECRET=whsec_...
```

From the nix repo's dev shell (`nix develop` or direnv), run
`agenix edit secrets/paddleocr-vl.env.age`, then `agenix rekey`. Both need the
YubiKey, so **hand these two commands to Stu** rather than attempting them. Don't
proceed to deploying `backend = "runpod"` until the rekeyed secret exists for rome.

### 2. Module: `modules/nixos/paddleocr-vl/default.nix`

Follow the repo's conventions (`my.*` options, `my.docker`, `my.persist`, tmpfiles,
loopback-only port, `age.secrets.*.rekeyFile`). Check how `modules/nixos/default.nix`
picks up modules and register the new one the same way.

```nix
{
  config,
  lib,
  ...
}:
with lib;
let
  cfg = config.my.paddleocr-vl;

  stateDirectory = "/var/lib/paddleocr-vl";
  image = "ghcr.io/stupremee/paddleocr-vl:main";
  docker = "${config.virtualisation.docker.package}/bin/docker";
in
{
  options.my.paddleocr-vl = {
    enable = mkEnableOption "PaddleOCR-VL OCR service (VLM inference on Runpod)";

    port = mkOption {
      type = types.port;
      default = 8330;
      description = "Host loopback port for the OCR API";
    };

    backend = mkOption {
      type = types.enum [ "runpod" "emulated" ];
      default = "runpod";
      description = "`emulated` answers with canned VLM output and needs no GPU or Runpod credentials";
    };

    workerConcurrency = mkOption {
      type = types.ints.positive;
      default = 4;
      description = "Jobs processed at once; each needs roughly 1-2 GB RAM";
    };

    autoUpdate = {
      enable = mkOption {
        type = types.bool;
        default = true;
        description = "Periodically pull the :main image and restart the container when it changed";
      };
      onCalendar = mkOption {
        type = types.str;
        default = "*:0/15";
        description = "systemd OnCalendar expression for the update check";
      };
    };
  };

  config = mkIf cfg.enable {
    my = {
      docker.enable = true;
      # the job queue must survive reboots (impermanence)
      persist.directories = [ stateDirectory ];
    };

    age.secrets.paddleocr-vl-env.rekeyFile = ../../../secrets/paddleocr-vl.env.age;

    systemd.tmpfiles.rules = [ "d ${stateDirectory} 0700 root root -" ];

    virtualisation.oci-containers.containers.paddleocr-vl = {
      inherit image;
      ports = [ "127.0.0.1:${toString cfg.port}:8080" ];
      environment = {
        OCR_BACKEND = cfg.backend;
        WORKER_CONCURRENCY = toString cfg.workerConcurrency;
      };
      environmentFiles = [ config.age.secrets.paddleocr-vl-env.path ];
      volumes = [ "${stateDirectory}:/data" ];
    };

    # Pull :main and restart only when the image actually changed. Jobs that are
    # running during a restart are requeued by the service on startup.
    systemd.services.paddleocr-vl-update = mkIf cfg.autoUpdate.enable {
      description = "Update the paddleocr-vl container image";
      after = [ "network-online.target" "docker.service" ];
      wants = [ "network-online.target" ];
      serviceConfig.Type = "oneshot";
      script = ''
        before="$(${docker} image inspect --format '{{.Id}}' ${image} 2>/dev/null || true)"
        ${docker} pull --quiet ${image}
        after="$(${docker} image inspect --format '{{.Id}}' ${image})"
        if [ "$before" != "$after" ]; then
          echo "new image $after, restarting paddleocr-vl"
          systemctl restart docker-paddleocr-vl.service
        fi
      '';
    };

    systemd.timers.paddleocr-vl-update = mkIf cfg.autoUpdate.enable {
      wantedBy = [ "timers.target" ];
      timerConfig = {
        OnCalendar = cfg.autoUpdate.onCalendar;
        RandomizedDelaySec = "2m";
        Persistent = true;
      };
    };
  };
}
```

Adjust anything that doesn't match the repo on inspection (e.g. how modules are
imported, the secret path, whether `my.backups` should include the state dir; it
holds only short-lived job data, so skipping the backup is fine).

### 3. Enable on rome, emulated first

In `configurations/nixos/rome/default.nix`, add under `my`:

```nix
paddleocr-vl = {
  enable = true;
  backend = "emulated"; # switch to "runpod" after the smoke test
};
```

Emulated mode exercises the real pipeline, job queue and webhooks without spending
GPU time, so the first deploy is free to verify. The secret file still has to exist
because `environmentFiles` references it. If step 1 isn't done yet, temporarily point
`environmentFiles` at an empty file created via tmpfiles, and remove that afterwards.

### 4. Deploy and verify

Deploy per the nix repo's `AGENTS.md` (`nix run .# -- rome` from the repo root),
then check:

```bash
systemctl status docker-paddleocr-vl.service
docker inspect -f '{{.State.Health.Status}}' paddleocr-vl   # healthy (allow ~2 min after start)
curl -s 127.0.0.1:8330/health                             # {"errorCode":0,"errorMsg":"Healthy",...}

# emulated smoke test: submit a job and poll it until "succeeded"
curl -s -X POST 127.0.0.1:8330/jobs -H 'content-type: application/json' \
  -d '{"request": {"file": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png", "fileType": 1, "visualize": false}}'
curl -s "127.0.0.1:8330/jobs/<id>?includeResult=false"

systemctl list-timers paddleocr-vl-update
systemctl start paddleocr-vl-update && journalctl -u paddleocr-vl-update -n 20
```

Then set `backend = "runpod"` (requires the secret from step 1), deploy again, and
repeat the health check. A real job costs a Runpod cold start (a few minutes of GPU
time, a few cents), so ask Stu before submitting one.

Report the deploy command output and the verification results. If anything
fails, include the failing command and its error output.

## Not in scope (yet)

- **Public exposure:** keep the port on loopback. Later access from Cloudflare (Lana)
  goes through the existing `my.cloudflare-tunnel` / `services.cloudflared` setup or
  Workers VPC. That's a separate task.
- **GPU on rome:** the RTX 2060 is on nouveau. Not needed for this service.

## Alternative: pinned digests instead of `:main`

For reproducible, rollback-able updates, pin `image` to
`ghcr.io/stupremee/paddleocr-vl@sha256:<digest>` (the CI job summary prints it),
drop the update timer, and bump the digest via Renovate or a CI commit to the nix repo.
Deploy with comin or `system.autoUpgrade`. This is more moving parts. Only do it if
Stu asks for it.
