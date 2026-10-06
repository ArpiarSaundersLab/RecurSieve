# Publishing a new version

1. Bump `version` in `pyproject.toml` (e.g. `1.0.0` → `1.0.1`).
2. Commit, push, and merge into `main`.
3. Done. The [Publish to PyPI](.github/workflows/publish.yml) workflow tests the package, publishes it to PyPI, and creates the `v<version>` GitHub Release.

Check the run under **Actions**. Merges that don't change the version are tested but not published.
