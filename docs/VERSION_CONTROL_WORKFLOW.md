# Version Control Workflow — mlops-iris-classifier

## 1. Overview

This document describes the Git-based version control workflow used for this Machine Learning project.

- Repository: mlops-iris-classifier
- Primary language: Python
- Project: MLOps Laboratory

## 2. Branching Strategy

| Branch | Purpose |
|---|---|
| `main` | Stable project code |
| `develop` | Integration branch |
| `feature/<name>` | Individual features |
| `conflict-demo-*` | Merge conflict practice |

## 3. Commit Convention

Commits use a short type prefix:

- `feat:` — new functionality
- `fix:` — bug correction
- `docs:` — documentation
- `chore:` — configuration/tooling
- `refactor:` — code restructuring

Example:

`feat: add classification report to training script`

## 4. Standard Workflow

```bash
git switch develop
git pull origin develop
git switch -c feature/<short-description>
git add <files>
git commit -m "feat: <description>"
git push -u origin feature/<short-description>