#!/usr/bin/env bash
# deploy.sh —— 服务器端：解包 + 逐文件校验 + 记录 DEPLOYED_COMMIT
#
# 三个保险中的第 2、3 个在这里：
#   - 任一文件 hash 不一致 -> 打印差异清单并以非零码退出（部署中止）
#   - 全部一致 -> 写 DEPLOYED_COMMIT
#
#   bash runs/scripts/deploy.sh --tar ~/deploy/deploy_xxx.tar.gz \
#        --manifest ~/deploy/deploy_manifest.sha256 --commit 1a2b3c4
set -uo pipefail

ROOT="${ROOT:-$HOME/Reca-Audio}"
TAR=""
MANIFEST=""
COMMIT="unknown"
DRY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tar) TAR="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --commit) COMMIT="$2"; shift 2 ;;
    --root) ROOT="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "未知参数: $1"; exit 2 ;;
  esac
done

[[ -f "$TAR" ]] || { echo "[deploy] 找不到 tar: $TAR"; exit 2; }
[[ -f "$MANIFEST" ]] || { echo "[deploy] 找不到 manifest: $MANIFEST"; exit 2; }
[[ -d "$ROOT" ]] || { echo "[deploy] 找不到仓库: $ROOT"; exit 2; }

echo "[deploy] 解包 -> $ROOT"
tar -xzf "$TAR" -C "$ROOT" || { echo "[deploy] 解包失败"; exit 1; }

FAILED=0
CHECKED=0
MISSING=()
MISMATCH=()

while read -r want rel; do
  [[ -z "${want:-}" ]] && continue
  rel="${rel%$'\r'}"
  CHECKED=$((CHECKED + 1))
  target="$ROOT/$rel"
  if [[ ! -f "$target" ]]; then
    MISSING+=("$rel"); FAILED=1; continue
  fi
  got=$(sha256sum "$target" | awk '{print $1}')
  if [[ "$got" != "$want" ]]; then
    MISMATCH+=("$rel"); FAILED=1
  fi
done < "$MANIFEST"

echo "[deploy] 校验 $CHECKED 个文件"

if [[ $FAILED -ne 0 ]]; then
  echo "❌ 校验失败，部署中止（远端未被标记为可用）" >&2
  [[ ${#MISSING[@]} -gt 0 ]] && { echo "缺失文件（${#MISSING[@]}）:" >&2; printf '  %s\n' "${MISSING[@]}" | head -20 >&2; }
  [[ ${#MISMATCH[@]} -gt 0 ]] && { echo "hash 不一致（${#MISMATCH[@]}）:" >&2; printf '  %s\n' "${MISMATCH[@]}" | head -20 >&2; }
  exit 1
fi

if [[ $DRY -eq 1 ]]; then
  echo "[deploy] dry-run：校验通过，不写 DEPLOYED_COMMIT"
  exit 0
fi

printf 'commit=%s\ndeployed_at=%s\nmanifest=%s\nfiles=%d\n' \
  "$COMMIT" "$(date -Iseconds)" "$(sha256sum "$MANIFEST" | awk '{print $1}')" "$CHECKED" \
  > "$ROOT/DEPLOYED_COMMIT"

echo "[deploy] ✅ 完成，DEPLOYED_COMMIT = $COMMIT（$CHECKED 个文件）"
