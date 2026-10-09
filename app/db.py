from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

from app.utils import AppError, dumps, now

TABLES = {"imports", "examples", "style_profiles", "jobs", "feedback"}


class Store:
    """每次操作独立连接；复合写操作使用同一事务。"""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        with self.transaction() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise AppError("数据库版本不受支持", "schema_version")
            for table in TABLES:
                db.execute(f"CREATE TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY, body TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS settings (id TEXT PRIMARY KEY, body TEXT NOT NULL)")
            db.execute(
                "CREATE TABLE IF NOT EXISTS example_versions "
                "(id TEXT, version INTEGER, body TEXT, PRIMARY KEY(id,version))"
            )
            db.execute("PRAGMA user_version=1")

    @contextmanager
    def transaction(self):
        with self.lock:
            db = sqlite3.connect(self.path, timeout=20)
            try:
                db.execute("PRAGMA foreign_keys=ON")
                db.execute("BEGIN IMMEDIATE")
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise
            finally:
                db.close()

    def put(self, table: str, record: dict, db=None):
        assert table in TABLES | {"settings"}
        if db is None:
            with self.transaction() as connection:
                self.put(table, record, connection)
            return
        db.execute(
            f"INSERT INTO {table}(id,body) VALUES(?,?) ON CONFLICT(id) DO UPDATE SET body=excluded.body",
            (record["id"], dumps(record)),
        )
        if table == "examples":
            db.execute(
                "INSERT OR IGNORE INTO example_versions VALUES(?,?,?)", (record["id"], record["version"], dumps(record))
            )

    def get(self, table: str, record_id: str, db=None) -> dict:
        assert table in TABLES | {"settings"}
        if db is None:
            with self.transaction() as connection:
                return self.get(table, record_id, connection)
        row = db.execute(f"SELECT body FROM {table} WHERE id=?", (record_id,)).fetchone()
        if not row:
            raise AppError("记录不存在", "not_found", 404)
        return json.loads(row[0])

    def all(self, table: str, db=None) -> list[dict]:
        assert table in TABLES | {"settings"}
        if db is None:
            with self.transaction() as connection:
                return self.all(table, connection)
        return [json.loads(row[0]) for row in db.execute(f"SELECT body FROM {table} ORDER BY rowid")]

    def update(self, table: str, record_id: str, changes: dict) -> dict:
        with self.transaction() as db:
            item = self.get(table, record_id, db)
            item.update(changes)
            self.put(table, item, db)
            return item

    def edit_example(self, record_id: str, changes: dict, expected_version: int) -> dict:
        with self.transaction() as db:
            item = self.get("examples", record_id, db)
            if item["version"] != expected_version:
                raise AppError("资料已被修改，请刷新后重试", "version_conflict", 409)
            item.update(changes)
            item["version"] += 1
            item["updated_at"] = now()
            self.put("examples", item, db)
            return item

    def backup_to(self, destination: Path):
        with self.lock:
            source = sqlite3.connect(self.path)
            target = sqlite3.connect(destination)
            try:
                source.backup(target)
                if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise AppError("备份完整性检查失败", "backup_failed", 500)
            finally:
                target.close()
                source.close()
