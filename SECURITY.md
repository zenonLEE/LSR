# Security

Do not commit API keys, session archives, environment files, private keys, VPN profiles or raw
authentication output. The live provider delegates authentication to Codex CLI and never reads a
token from this repository.

## Runtime boundary

The live provider launches each Codex request from a fresh temporary directory with a least-
privilege permission profile, ephemeral session state, no repository rules, and no user
configuration. Sandboxed commands may read only Codex's minimal runtime paths and that empty
temporary workspace; command network access is disabled. Only a small allowlist of operating-system
and Codex login-location variables is passed to the child process, so credential-shaped environment
variables are excluded. Prompt sample blocks are treated as untrusted scientific data, and free-text
amine labels are replaced with numeric feature codes.

Permission profiles require Codex CLI 0.138.0 or later and are currently a beta Codex feature. The
provider uses strict config validation, so an incompatible CLI fails closed instead of silently
falling back to a broader sandbox.

Saved raw responses and provider errors are redacted for common credential shapes. Raw output is
off by default. When enabled, repository-local outputs must be placed under the ignored
`artifacts/` directory. These controls reduce accidental disclosure but do not make arbitrary
untrusted prompts safe for a privileged agent.

## Repository checks

Run `python scripts/check_secrets.py --history` before publication. It checks tracked files,
unignored candidate files, and every reachable commit. CI performs the same scan and uses
commit-pinned GitHub Actions. Findings report only location and type, never the suspected value.

If a credential is committed, revoke it first and then remove it from Git history.
