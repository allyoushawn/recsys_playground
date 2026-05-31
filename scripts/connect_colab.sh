#!/bin/bash
# Connect to a Colab runtime via cloudflared SSH tunnel.
# Usage: ./scripts/connect_colab.sh <cloudflared-hostname>
#
# The hostname is printed by the colab_ssh_bootstrap.ipynb notebook
# after running the SSH setup cell (e.g. "loud-turkey-foo.trycloudflare.com").

HOST=${1:?"Usage: $0 <cloudflared-hostname>"}

ssh -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o IdentityFile=~/.ssh/colab_key \
    -o ProxyCommand="cloudflared access ssh --hostname $HOST" \
    root@"$HOST"
