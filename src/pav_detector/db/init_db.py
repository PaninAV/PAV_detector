from __future__ import annotations

from pav_detector.config import Settings
from pav_detector.db.postgres import PostgresStorage
from pav_detector.utils.logging_json import configure_logging


def main() -> None:
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    storage = PostgresStorage(settings.postgres_dsn)
    storage.init_schema()


if __name__ == "__main__":
    main()
