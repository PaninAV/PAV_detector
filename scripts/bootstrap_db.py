#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from urllib.parse import urlsplit, urlunsplit

import psycopg2
from dotenv import load_dotenv
from psycopg2 import sql

from pav_detector.db.postgres import PostgresStorage


def _as_bool(raw: str | None, default: bool = True) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _extract_db_name(dsn: str) -> str:
    parsed = urlsplit(dsn)
    db_name = parsed.path.lstrip("/")
    if not db_name:
        raise ValueError("POSTGRES_DSN must include a database name in the path.")
    return db_name


def _replace_db_name(dsn: str, db_name: str) -> str:
    parsed = urlsplit(dsn)
    return urlunsplit((parsed.scheme, parsed.netloc, f"/{db_name}", parsed.query, parsed.fragment))


def ensure_database_exists(target_dsn: str, admin_db: str = "postgres") -> None:
    target_db_name = _extract_db_name(target_dsn)
    admin_dsn = _replace_db_name(target_dsn, admin_db)

    with psycopg2.connect(admin_dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target_db_name,))
            exists = cur.fetchone() is not None
            if not exists:
                cur.execute(
                    sql.SQL("CREATE DATABASE {}").format(sql.Identifier(target_db_name))
                )
                print(f"[setup] Database created: {target_db_name}")
            else:
                print(f"[setup] Database already exists: {target_db_name}")


def main() -> int:
    load_dotenv(".env")

    if not _as_bool(os.getenv("ENABLE_DB", "true"), default=True):
        print("[setup] ENABLE_DB=false -> skipping PostgreSQL bootstrap.")
        return 0

    dsn = os.getenv(
        "POSTGRES_DSN",
        "postgresql://postgres:postgres@localhost:5432/pav_detector",
    )
    admin_db = os.getenv("POSTGRES_ADMIN_DB", "postgres")

    try:
        ensure_database_exists(dsn, admin_db=admin_db)
        PostgresStorage(dsn).init_schema()
        print("[setup] PostgreSQL schema initialized.")
    except Exception as exc:
        print(f"[setup] PostgreSQL bootstrap failed: {exc}", file=sys.stderr)
        print(
            "[setup] Check POSTGRES_DSN credentials, PostgreSQL availability, "
            "and CREATE DATABASE permissions.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
