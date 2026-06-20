# Lab 4 — Autoscale Under Load

**Book 3 · Part B, Chapter 9 (autoscaling, observability)**

---

## Hypothesis

When horizontal autoscaling is configured on the right signal (queue depth or requests-per-replica), replica count tracks load, and p99 latency stays bounded once replicas are warm. A visible cold-start lag appears when new replicas are added, because model weights must load before the replica can serve traffic.

## Background

**Wrong signal — GPU utilization:** decode is memory-bound, so GPU utilization sits at 20–40% even under heavy load. Autoscaling on GPU utilization will never trigger — the signal stays permanently low.

**Right signal — queue depth or pending requests per replica:** a growing queue is the direct indicator that supply (server capacity) is insufficient for demand. KEDA (Kubernetes Event-Driven Autoscaling) can scale on Prometheus metrics including vLLM's `vllm:num_requests_waiting`.

**Cold-start lag:** when a new replica is scheduled, it must: pull the container image (if not cached), load model weights into VRAM (10–60 seconds for a 7B model), and pass its readinessProbe before Kubernetes routes traffic to it. During this window, the existing replicas absorb all traffic. This is why keeping at least one "warm" replica at all times — even under zero load — is standard practice for latency-sensitive serving.

## Setup

```bash
# Prometheus + Grafana for metrics (install via Helm or apply manifests)
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install kube-prometheus prometheus-community/kube-prometheus-stack

# KEDA for event-driven autoscaling
helm repo add kedacore https://kedacore.github.io/charts
helm install keda kedacore/keda

# vLLM exposes Prometheus metrics at /metrics by default
```

## Procedure

### Part A — Deploy with HPA or KEDA
1. Deploy your model server from Lab 3 with `replicas: 1`.
2. Configure autoscaling. Option A — simple HPA on CPU (for demonstration):
   ```bash
   kubectl autoscale deployment llm --min=1 --max=5 --cpu-percent=50
   ```
   Option B — KEDA `ScaledObject` on `vllm:num_requests_waiting` Prometheus metric (more realistic for LLM serving):
   ```yaml
   triggers:
   - type: prometheus
     metadata:
       serverAddress: http://prometheus:9090
       metricName: vllm_num_requests_waiting
       threshold: "5"
   ```

### Part B — Drive a load ramp
1. Use an async load driver to ramp requests: start at 5 req/s, hold for 2 min, jump to 20 req/s, hold for 3 min, drop back to 5 req/s.
2. Record: replica count, queue depth, p50 TTFT, p99 TTFT — all on the same time axis.

### Part C — Observe cold-start lag
1. Ensure no Pods are pre-warmed (scale to 1 replica, let it idle).
2. Send a sudden burst of requests and start timing from the burst until p99 returns to the stable-region value.
3. The gap is the cold-start window: the autoscaler noticed the queue, scheduled new Pods, and waited for them to load weights.

## Expected Observations

- **Replica count rises with load** — with a lag (autoscaler polling interval + Pod scheduling + weight loading).
- **p99 latency stays bounded once warm replicas are ready** — the queue drains as replicas come online.
- **A visible latency spike occurs during cold start** — p99 rises sharply at the load ramp and recovers as new replicas warm up.
- **p99 recovers later than replica count** because Kubernetes starts routing traffic only after the readinessProbe passes (weight loading complete).
- On load decrease, replicas scale down slowly (cool-down period prevents oscillation).

## Why This Matters

Autoscaling configuration is one of the most common sources of production LLM serving problems. Teams that scale on GPU utilization never scale. Teams that scale on CPU usage scale for the wrong reasons. Teams without warm-replica headroom see latency spikes on every traffic surge. This lab makes the right and wrong choices observable rather than theoretical, and the cold-start lag measurement gives you the data to set appropriate `minReplicas` for your latency SLA.

## Variant Experiments

- Configure a `minReplicas: 2` (always-warm floor) and repeat the burst experiment — observe the cold-start lag shrink or disappear.
- Add a scale-down stabilization window (`scaleDown.stabilizationWindowSeconds: 300`) and observe that replica count doesn't oscillate during variable traffic.
- Plot the cost over time of the load ramp with and without autoscaling — autoscaling reduces idle-GPU cost during low-traffic periods at the expense of occasional cold-start latency.
