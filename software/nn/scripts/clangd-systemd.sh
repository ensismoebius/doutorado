systemd-run --user --scope \
  -p MemoryMax=10G \
  -p MemoryHigh=1500M \
  -p MemorySwapMax=1G \
  -p CPUQuota=200% \
  -p Nice=5 \
  /usr/bin/clangd "$@"