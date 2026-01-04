systemd-run --user --scope \
  -p MemoryMax=8G \
  -p MemoryHigh=1500M \
  -p MemorySwapMax=1G \
  -p CPUQuota=100% \
  /usr/bin/clangd "$@"