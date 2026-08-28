# Original User Request

## 2026-08-27T16:31:33Z

# Teamwork Project Prompt — Dynamic Multi-Platform Worker Pool & Auto-Capacity Balancing

Status: Launched
Rule: NO git commit & push without explicit user consent.
Goal: Continuous speed optimization + dynamic thread reallocation until all Meta videos are ingested.

Implement an intelligent dynamic scraper supervisor and worker pool that maximizes overall throughput and automatically shifts worker capacity when one platform completes.

Working directory: `/Users/LFH/code/leonhelfinger/project/social-media-optimizer`
Integrity mode: development

## Requirements

### R1. Continuous Throughput & Speed Optimization
- Constantly monitor individual stream download speeds and aggregate ingestion velocity (`Speed XX /min`).
- Auto-detect slow or stalled sockets and gracefully recycle them to prevent throughput dips.
- Ensure optimal format stream selection (`best[ext=mp4][height<=?720]`) and direct progressive downloads.

### R2. Dynamic Workload Rebalancing & Power Handoff
- Monitor remaining backlog for both Facebook and Instagram in real time.
- **Dynamic Thread Allocation**:
  - Balanced Mode: 5 threads Facebook + 5 threads Instagram (10 total concurrent workers).
  - Power Handoff Mode: If Instagram finishes earlier or runs out of pending items, instantly shift its 5 worker threads to Facebook (running Facebook at full 10x capacity), and vice versa if Facebook finishes first.
- Zero idle capacity: 100% of available bandwidth is always utilized.

### R3. Strict Git Policy
- All modifications and dynamic pool logic operate locally without automatic Git commit/push.
- No remote pushes occur without explicit user confirmation.

## Acceptance Criteria

### Speed & Monitoring
- [ ] Aggregate ingestion speed is monitored continuously with rolling velocity metrics.
- [ ] Failed or slow connections are automatically retried without dropping overall throughput.

### Capacity Transfer
- [ ] Automatic detection when a platform backlog reaches 0 or all videos are ingested.
- [ ] Complete transfer of idle worker slots to the remaining platform (e.g. Facebook scaling from 5 to 10 workers).

### Safety & Continuity
- [ ] No git commit or push executed without explicit user permission.
- [ ] Continuous synchronization to GCS bucket `sm-optimizer-processed`.
