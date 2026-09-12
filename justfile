test:
    uv run pytest

check:
    uv run ruff check .
    uv run ruff format --check .
    uv run ty check
    uv run pytest

fix:
    uv run ruff check --fix .
    uv run ruff format .

# 打包规则不逐个列举模块: 新增 .py 必须自动进包. 打完后校验 zip 根目录有 plugin.py.
pack:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p dist
    rm -f dist/sqzw.local.zip
    zip -X dist/sqzw.local.zip ./*.py
    zipinfo -1 dist/sqzw.local.zip | grep -qx 'plugin.py'
    unzip -l dist/sqzw.local.zip
