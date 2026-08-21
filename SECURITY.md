# Security Policy

## Supported Version

Security fixes target the current `main` branch until versioned releases are
published. Release support will be documented here when the policy changes.

## Reporting a Vulnerability

Do not open a public issue for a suspected vulnerability. Use the repository's
[private security advisory form](https://github.com/Doithoo/pytorch-image-segmentation-lab/security/advisories/new)
and include the affected revision, reproduction, impact, and any suggested
mitigation. Remove private datasets, credentials, and local paths.

Datasets and checkpoints are external inputs. Prepared manifests are verified
against recorded SHA-256 values, and current checkpoints load through PyTorch's
`weights_only=True` mode. Report any path that bypasses those boundaries.
