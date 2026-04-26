import ssl
import logging
from app.data.models import init_db
from app.data.repository import PortfolioRepository

logger = logging.getLogger(__name__)



def ensure_database_populated(progress_callback=None) -> bool:


    init_db()

    repo = PortfolioRepository()
    latest_date = repo.get_latest_quote_date()

    if latest_date is None:
        msg = "База даних порожня. Запускаємо початкове завантаження (S&P 500, 30 років історії)..."
        print(f"\n[BOOTSTRAP] {msg}")
        logger.info(msg)

        if progress_callback:
            progress_callback(0, "Початок ініціалізації бази даних...")

        from app.core.core import PortfolioCore
        core = PortfolioCore(repo=repo)
        core.sync_market_data(progress_callback=progress_callback)

        # Перевіряємо що реально щось збережено
        saved_count = len(repo.get_all_tickers())
        if saved_count == 0:
            raise RuntimeError(
                "Завантаження завершилось, але база даних залишилась порожньою. "
                "Перевірте з'єднання з інтернетом і спробуйте ще раз."
            )

        print(f"[BOOTSTRAP] Завантаження успішно завершено! Активів у БД: {saved_count}\n")
        return True

    else:
        msg = f"База даних вже містить дані (остання дата: {latest_date}). Початкове завантаження не потрібне."
        print(f"[BOOTSTRAP] {msg}")
        logger.info(msg)
        return False