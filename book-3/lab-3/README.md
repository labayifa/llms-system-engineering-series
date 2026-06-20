# Lab 3 — Deploy on Kubernetes

**Book 3 · Part B, Chapter 7 (Kubernetes for inference, GPU scheduling)**

---

## Hypothesis

A Deployment + Service gives self-healing, load-balanced model serving: killed Pods respawn automatically (the Deployment reconciles desired state to actual state), and traffic distributes across replicas (the Service load-balances across healthy Pods).

## Background

Kubernetes manages containerized workloads by continuously reconciling *desired state* (declared in a Deployment manifest) with *actual state* (what is running on nodes). When a Pod dies, the Deployment controller notices the shortfall and schedules a replacement. A Service provides a stable virtual IP that round-robins incoming traffic to the healthy Pod IPs behind it, decoupling clients from individual Pod lifecycles. For GPU workloads, the NVIDIA device plugin exposes `nvidia.com/gpu` as a schedulable resource — the scheduler places Pods only on nodes that have available GPUs.

## Setup

```bash
# Install kind (local Kubernetes cluster in Docker)
brew install kind          # macOS
# or: https://kind.sigs.k8s.io/docs/user/quick-start/

# Create a cluster
kind create cluster --name llm-lab

# Install the NVIDIA device plugin (needed for GPU Pods; skip on CPU-only)
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Confirm kubectl is configured
kubectl get nodes
```

A CPU-only local cluster is fine for learning the Kubernetes mechanics. For GPU scheduling, use a cloud Kubernetes service (GKE, EKS, AKS) with GPU node pools, or a local machine with the NVIDIA device plugin installed.

## Procedure

### Part A — Deploy a model server
1. Build or pull a container image for a small model server (vLLM or a minimal FastAPI wrapper around a HuggingFace model).
2. Write a `deployment.yaml` defining a Deployment with `replicas: 1` and a Service of type `ClusterIP` (or `LoadBalancer` for cloud).
3. Apply and verify:
   ```bash
   kubectl apply -f deployment.yaml
   kubectl get pods -w          # watch Pod reach Running state
   kubectl get svc              # confirm Service is created
   ```

### Part B — Scale and observe self-healing
1. Scale to 3 replicas:
   ```bash
   kubectl scale deployment/llm --replicas=3
   kubectl get pods -w
   ```
2. Delete one Pod manually:
   ```bash
   kubectl delete pod <pod-name>
   kubectl get pods -w          # watch it respawn
   ```
3. Send a stream of requests through the Service while deleting Pods — confirm traffic is not interrupted (Kubernetes routes around the deleted Pod before a replacement is ready; brief 502s are expected during the transition window).

### Part C — Verify load balancing
1. Add a `/hostname` endpoint to your model server that returns the Pod name.
2. Send 20 requests to the Service and record which Pod each was routed to.

## Expected Observations

- **Deleted Pods respawn automatically** within ~10–30 seconds (time for the scheduler to assign, pull the image if not cached, and start the container).
- **Traffic distributes across replicas** — the `/hostname` endpoint responses should cycle through all three Pod names.
- **Model loading adds to startup latency** — the Pod reaches `Running` state seconds after container start, but the model server may not be ready to serve until weights are loaded (add a readinessProbe to your Deployment to block traffic until the server is warm).

## Why This Matters

Running a model server directly on a VM or bare-metal machine gives you no self-healing, no traffic balancing, and no declarative configuration. A single crash takes the service down until someone manually restarts it. Kubernetes turns that into a background operation. The difference between `kubectl scale --replicas=3` and SSH-ing into three machines to start the server manually is also the difference between an operator who can sleep at night and one who can't. This lab makes that difference concrete.

## Variant Experiments

- Add a `readinessProbe` that hits the model server's `/health` endpoint — observe that Kubernetes waits for it before routing traffic, eliminating the "serving errors during startup" problem.
- Configure a `PodDisruptionBudget` (`minAvailable: 2`) and attempt to drain a node with all three Pods — observe Kubernetes refusing to evict more than one at a time.
- Add a `resources.limits.nvidia.com/gpu: 1` resource request and observe the Pod staying in `Pending` state on a CPU-only cluster — then move to a GPU-enabled node and confirm it schedules.
