#!/usr/bin/env bash
# Host-side deployment operations, always scoped to this project's private VM.
set -euo pipefail

deploy_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_dir="$(cd -- "$deploy_dir/../.." && pwd -P)"
runtime_dir="$repo_dir/data/x2_sim/gpu_probe/runtime-rebuild"
socket="$runtime_dir/docker.sock"
container="malecns-x2-gpu-trial"
image="malecns-lab-x2-gpu:local"

usage() {
    echo "Usage: bash deploy/x2/manage.sh config|build|create|ps|logs" >&2
    echo "config is offline. Other commands require the private GPU VM to be running." >&2
    echo "This script never starts the VM, Sim, MC or a brain process." >&2
}
if [[ $# != 1 ]]; then
    usage
    exit 2
fi
case "$1" in
    config|build|create|ps|logs) ;;
    *) usage; exit 2 ;;
esac

# Ignore inherited endpoints, TLS and remote build-service selection. An explicit
# private host and default builder keep Docker Desktop and its workloads separate.
unset DOCKER_HOST DOCKER_CONTEXT DOCKER_TLS DOCKER_TLS_VERIFY DOCKER_CERT_PATH
unset BUILDKIT_HOST COMPOSE_BAKE
export BUILDX_BUILDER=default
export LIGHTER_HOME="$runtime_dir"
docker_private=(docker --host "unix://$socket")
compose=("${docker_private[@]}" compose --env-file /dev/null
         --project-name malecns-x2-gpu --file "$deploy_dir/compose.yaml")

if [[ "$1" == config ]]; then
    exec "${compose[@]}" config
fi
if [[ ! -x "$runtime_dir/bin/lighter" || ! -S "$socket" ]]; then
    echo "Private GPU VM is not available at $runtime_dir; no resources were started." >&2
    exit 1
fi
"${docker_private[@]}" info --format '{{.ID}}' >/dev/null
case "$1" in
    build)
        exec "${compose[@]}" build x2
        ;;
    create)
        existing="$("${docker_private[@]}" container ls --all \
            --filter "name=^/${container}$" --format '{{.Names}}')"
        if [[ -n "$existing" ]]; then
            echo "$container already exists; refusing to replace the installed runtime container." >&2
            exit 1
        fi
        "${docker_private[@]}" image inspect "$image" >/dev/null
        exec "${compose[@]}" create --no-build --pull never x2
        ;;
    ps)
        exec "${docker_private[@]}" container ls --all \
            --filter "name=^/${container}$"
        ;;
    logs)
        exec "${docker_private[@]}" logs --tail 100 "$container"
        ;;
esac
