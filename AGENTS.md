# Repository Guidelines

## Project Structure & Module Organization
This repository is currently minimal. The tracked files are [`README.md`](/home/ck/Code/dataVisPy/README.md) and Python-oriented ignore rules in `.gitignore`, so contributors should keep the layout simple and predictable as the codebase grows.

Recommended structure:
- `src/` for application modules
- `tests/` for automated tests
- `assets/` for sample data, images, or static resources
- `README.md` for setup and usage notes

Prefer small, focused modules. Use package-style paths such as `src/data_vis_py/loader.py` instead of loose top-level scripts.

## Build, Test, and Development Commands
No build system or test runner is configured yet. Until project tooling is added, use standard Python workflows and document any new commands in `README.md` and this file.

Common local setup examples:
- `python -m venv .venv` creates a virtual environment
- `source .venv/bin/activate` activates it on Unix shells
- `python -m pytest` runs the test suite once `pytest` is added
- `python -m pip install -r requirements.txt` installs dependencies if a requirements file is introduced

If you add tooling such as `ruff`, `pytest`, or `make`, include checked-in config files and keep command names stable.

## Coding Style & Naming Conventions
Target Python 3 with 4-space indentation and PEP 8 naming:
- `snake_case` for functions, variables, and modules
- `PascalCase` for classes
- `UPPER_CASE` for constants

Prefer type hints on public functions and short docstrings for non-obvious behavior. Keep files ASCII unless a clear reason exists otherwise.

## Testing Guidelines
Place tests under `tests/` and name them `test_*.py`. Match test files to the module they cover, for example `tests/test_loader.py` for `src/data_vis_py/loader.py`.

Add tests with each feature or bug fix. If coverage tooling is added later, document the target threshold in the repository.

## Commit & Pull Request Guidelines
Current history uses short, imperative commit messages such as `Initial commit` and `Update README.md`. Continue that pattern:
- `Add CSV parser`
- `Fix axis label formatting`

Pull requests should include a clear summary, testing notes, and linked issues when applicable. Include screenshots or generated output when a change affects visualizations or user-facing artifacts.
