# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-04

### Added

- Dynamic causality oracle: `check`, `check_label` and `assert_causal` with
  truncation and perturbation tests, leak horizon estimation and a text/JSON
  report.
- Static linter with thirteen rules (PP001 to PP013) for pandas, polars,
  NumPy and scikit-learn research code, with notebook support, inline
  suppression and `pyproject.toml` configuration.
- Command-line interface: `peekproof lint` and `peekproof check`.
- Pre-commit hook definition.
