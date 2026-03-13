# Infrastructure — K8s, Terraform, Ansible

Read this for any task involving Kubernetes workloads, infrastructure-as-code, or environment provisioning.

---

## Kubernetes — Helm Always

All workloads use Helm charts — no raw manifests in production.

**Non-negotiables on every deployment:**
- Resource limits (CPU + memory) — required, not optional.
- Liveness and readiness probes — required, not optional.
- Horizontal Pod Autoscaler (`hpa.yaml`) with a sensible starting point.

**Default recommended structure per service:**
```
helm/
  Chart.yaml
  values.yaml           ← defaults
  values-dev.yaml       ← env overrides
  values-staging.yaml
  values-prod.yaml
  templates/
    deployment.yaml
    service.yaml
    ingress.yaml
    hpa.yaml
    _helpers.tpl
```

- Multi-platform target: AWS + Azure — avoid cloud-specific K8s features unless behind an abstraction.
- When suggesting K8s config, always include: resource limits, health probes, and a sensible HPA starting point.

---

## Terraform

- **Separate workspaces per environment**: `dev`, `staging`, `prod`.
- State stored in remote backend (S3 for AWS, Azure Storage for Azure) — never local state in shared repos.
- **All infra changes via PR** — no manual console or CLI changes, ever.
- `terraform plan` must work locally against the dev environment for fast feedback — use `.env` + `terraform.tfvars` pattern for local credentials.

**Module structure:**
```
infra/
  modules/
    <service-name>/     ← reusable module
  environments/
    dev/
    staging/
    prod/
```

- AWS and Azure get separate module implementations where needed; shared interface where possible.
- When proposing Terraform changes, always specify which workspace is targeted and include a note on plan/apply sequence.

---

## Ansible — Bootstrap Only

- Ansible is for **initial machine/cluster setup only** — never for ongoing config management.
- Ongoing state is owned by K8s + Helm + Terraform, not Ansible.
- Playbooks live in `infra/ansible/` and are treated as one-time setup scripts.
- If you find yourself writing an Ansible playbook for something that isn't bootstrap, stop and question whether it belongs in Helm or Terraform instead.
