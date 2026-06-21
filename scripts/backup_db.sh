#!/usr/bin/env bash
# =============================================================
# scripts/backup_db.sh
# SQLite database backup and restore utility
# =============================================================
# Usage:
#   ./scripts/backup_db.sh backup          # create timestamped backup
#   ./scripts/backup_db.sh restore latest  # restore most recent backup
#   ./scripts/backup_db.sh restore FILE    # restore specific backup file
#   ./scripts/backup_db.sh list            # list all backups
#   ./scripts/backup_db.sh clean 7         # delete backups older than 7 days

set -euo pipefail

DB_PATH="data/processed/cmdb.sqlite"
BACKUP_DIR="data/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RETENTION_DAYS=${RETENTION_DAYS:-30}

mkdir -p "$BACKUP_DIR"

CMD="${1:-backup}"

# ─────────────────────────────────────────────────────────────
backup() {
    if [ ! -f "$DB_PATH" ]; then
        echo "✗ Database not found: $DB_PATH"
        echo "  Run 'python main.py' to initialise the pipeline first."
        exit 1
    fi

    BACKUP_FILE="$BACKUP_DIR/cmdb_${TIMESTAMP}.sqlite"

    # SQLite hot backup using .backup command
    sqlite3 "$DB_PATH" ".backup '$BACKUP_FILE'"

    SIZE=$(du -sh "$BACKUP_FILE" | cut -f1)
    echo "✓ Backup created: $BACKUP_FILE ($SIZE)"

    # Count rows for verification
    ROWS=$(sqlite3 "$BACKUP_FILE" "SELECT COUNT(*) FROM health_scores;" 2>/dev/null || echo "?")
    echo "  Health score records: $ROWS"

    # Compress
    gzip -f "$BACKUP_FILE"
    FINAL="${BACKUP_FILE}.gz"
    SIZE_GZ=$(du -sh "$FINAL" | cut -f1)
    echo "  Compressed: $FINAL ($SIZE_GZ)"
}

# ─────────────────────────────────────────────────────────────
restore() {
    TARGET="${2:-latest}"

    if [ "$TARGET" = "latest" ]; then
        BACKUP_FILE=$(ls -t "$BACKUP_DIR"/cmdb_*.sqlite.gz 2>/dev/null | head -1)
        if [ -z "$BACKUP_FILE" ]; then
            echo "✗ No backups found in $BACKUP_DIR"
            exit 1
        fi
    else
        BACKUP_FILE="$TARGET"
    fi

    if [ ! -f "$BACKUP_FILE" ]; then
        echo "✗ Backup file not found: $BACKUP_FILE"
        exit 1
    fi

    echo "→ Restoring from: $BACKUP_FILE"
    read -rp "  This will overwrite $DB_PATH. Continue? [y/N] " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "  Cancelled."
        exit 0
    fi

    # Backup current DB before restoring
    if [ -f "$DB_PATH" ]; then
        PRE_BACKUP="$BACKUP_DIR/pre_restore_${TIMESTAMP}.sqlite"
        sqlite3 "$DB_PATH" ".backup '$PRE_BACKUP'"
        gzip -f "$PRE_BACKUP"
        echo "  → Pre-restore backup saved: ${PRE_BACKUP}.gz"
    fi

    # Decompress and restore
    gunzip -c "$BACKUP_FILE" > "${DB_PATH}.tmp"
    mv "${DB_PATH}.tmp" "$DB_PATH"
    echo "✓ Restored: $DB_PATH"

    ROWS=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM health_scores;" 2>/dev/null || echo "?")
    echo "  Health score records: $ROWS"
}

# ─────────────────────────────────────────────────────────────
list_backups() {
    echo ""
    echo "Available backups in $BACKUP_DIR:"
    echo ""
    if ls "$BACKUP_DIR"/cmdb_*.sqlite.gz &>/dev/null; then
        printf "  %-40s %8s\n" "File" "Size"
        echo "  $(printf '%.0s─' {1..50})"
        for f in $(ls -t "$BACKUP_DIR"/cmdb_*.sqlite.gz); do
            SIZE=$(du -sh "$f" | cut -f1)
            printf "  %-40s %8s\n" "$(basename "$f")" "$SIZE"
        done
    else
        echo "  No backups found."
    fi
    echo ""

    TOTAL=$(ls "$BACKUP_DIR"/cmdb_*.sqlite.gz 2>/dev/null | wc -l || echo 0)
    echo "  Total: $TOTAL backup(s)"
}

# ─────────────────────────────────────────────────────────────
clean_old() {
    DAYS="${2:-$RETENTION_DAYS}"
    echo "→ Removing backups older than $DAYS days..."
    COUNT=$(find "$BACKUP_DIR" -name "cmdb_*.sqlite.gz" -mtime "+$DAYS" | wc -l)

    if [ "$COUNT" -eq 0 ]; then
        echo "  No old backups to remove."
    else
        find "$BACKUP_DIR" -name "cmdb_*.sqlite.gz" -mtime "+$DAYS" -delete
        echo "✓ Removed $COUNT old backup(s)"
    fi
}

# ─────────────────────────────────────────────────────────────
verify() {
    BACKUP_FILE="${2:-$(ls -t "$BACKUP_DIR"/cmdb_*.sqlite.gz 2>/dev/null | head -1)}"
    if [ -z "$BACKUP_FILE" ]; then
        echo "✗ No backup to verify."
        exit 1
    fi
    echo "→ Verifying: $BACKUP_FILE"
    TMP=$(mktemp --suffix=.sqlite)
    gunzip -c "$BACKUP_FILE" > "$TMP"
    sqlite3 "$TMP" "PRAGMA integrity_check;" | head -5
    TABLES=$(sqlite3 "$TMP" ".tables")
    echo "  Tables: $TABLES"
    rm -f "$TMP"
    echo "✓ Backup verified"
}

# ─────────────────────────────────────────────────────────────
case "$CMD" in
    backup)  backup ;;
    restore) restore "$@" ;;
    list)    list_backups ;;
    clean)   clean_old "$@" ;;
    verify)  verify "$@" ;;
    *)
        echo "Usage: $0 {backup|restore [latest|FILE]|list|clean [DAYS]|verify [FILE]}"
        exit 1 ;;
esac
