# Publishing a new version

1. Bump `version` in `pyproject.toml` and `CITATION.cff` (e.g. `1.0.1` → `1.0.2`).
2. Commit, push, and merge into `main`.
3. Create a GitHub Release tagged `v<version>` on `main`, e.g.:
   `gh release create v1.0.2 --repo ArpiarSaundersLab/RecurSieve --target main --generate-notes`
4. Done. The [Publish to PyPI](.github/workflows/publish.yml) workflow tests the package and publishes it to PyPI, and Zenodo archives the release.

Check the run under **Actions**. Pushes to `main` without a release are tested but not published.
