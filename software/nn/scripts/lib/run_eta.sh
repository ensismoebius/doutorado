#!/usr/bin/env bash
# run_eta.sh — work-weighted, EMA-smoothed ETA for multi-profile runs.
#
# Shared by the thesis (run_e05_profiles.sh) and the Guayaquil paper
# (01_e04_run_article_profiles.sh). `source` it, then per profile:
#
#   eta_reset                       # once, at the start of a run
#   w=$(profile_weight "$path")     # caller-provided cost estimate (see below)
#   ...run the profile, timing it...
#   eta_update "$w" "$seconds"      # fold the finished profile into the rate
#   remaining_s=$(eta_remaining "$remaining_weight")   # before the next one
#
# WHY WORK-WEIGHTED. The obvious `elapsed * (remaining/done)` assumes every profile
# costs the same. These runs violate that badly — E04 mixes one fast LSTM with three
# slow SNNs (~4-5x), E05 mixes fast handcrafted extraction (no training) with slow
# autoencoders. Counting profiles equally makes the ETA lurch at each boundary. Instead
# we track seconds *per unit of estimated work*: rate = sum(seconds)/sum(weight), and the
# remaining time is rate * remaining_weight. Even rough weights (2 vs 9) turn a wildly
# wrong count-based estimate into a usable one.
#
# WHY EMA. On top of the weighting, the per-unit rate is smoothed with an exponential
# moving average (alpha=0.3, the same factor tqdm uses for its speed estimate) so a single
# slow/fast profile nudges the estimate instead of yanking it. This is the combination the
# ETA literature recommends: meaningful work units + exponential smoothing of the rate.
#
# It is still an estimate. It is optimistic before the first heavy profile completes (the
# weights are a prior, not a measurement) and tightens as real timings arrive — as every
# honest ETA does.

_ETA_RATE=""          # EMA of seconds-per-unit-weight; empty until the first profile lands
_ETA_ALPHA="${ETA_ALPHA:-0.3}"

eta_reset() { _ETA_RATE=""; }

# eta_update <weight> <seconds> : fold one completed profile into the smoothed rate.
eta_update() {
    _ETA_RATE=$(awk -v w="$1" -v s="$2" -v r="${_ETA_RATE:-}" -v a="$_ETA_ALPHA" 'BEGIN{
        if (w+0 <= 0) { print r; exit }          # guard: no negative/zero-weight work
        nr = s / w;                               # this profile'"'"'s seconds-per-unit
        if (r == "") print nr;                    # seed on first sample
        else print a*nr + (1-a)*r;                # EMA
    }')
}

# eta_remaining <remaining_weight> : integer seconds left, or empty if no rate yet.
eta_remaining() {
    [ -n "${_ETA_RATE:-}" ] || return 0
    awk -v r="$_ETA_RATE" -v w="$1" 'BEGIN{ if (w+0 < 0) w=0; printf "%d", r*w }'
}

# fmt_hms <seconds> : compact H:MM:SS / M:SS for display.
fmt_hms() {
    local s=$1
    if (( s >= 3600 )); then printf '%d:%02d:%02d' $((s/3600)) $(((s%3600)/60)) $((s%60))
    else printf '%d:%02d' $((s/60)) $((s%60)); fi
}
