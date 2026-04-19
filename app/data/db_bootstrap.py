import ssl
import logging
from app.data.models import init_db
from app.data.repository import PortfolioRepository

logger = logging.getLogger(__name__)


def _apply_macos_ssl_fix():
    """Дозволяє Python на Mac завантажувати сторінки Вікіпедії без помилок сертифікатів SSL."""
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context


def ensure_database_populated(progress_callback=None) -> bool:
    """
    Перевіряє стан бази даних. Якщо вона порожня (немає котирувань),
    запускає повне початкове завантаження даних (S&P 500 за 30 років).
    Повертає True, якщо було виконано завантаження.
    """
    _apply_macos_ssl_fix()

    # 1. Ініціалізуємо структури SQLite (створить порожній файл portfolio.db, якщо його немає)
    init_db()

    # 2. Перевіряємо, чи є в базі хоча б одна дата
    repo = PortfolioRepository()
    latest_date = repo.get_latest_quote_date()

    if latest_date is None:
        msg = "База даних порожня. Запускаємо початкове завантаження (S&P 500, 30 років історії)..."
        print(f"\n[BOOTSTRAP] {msg}")
        logger.info(msg)

        if progress_callback:
            progress_callback(0, "Початок ініціалізації бази даних...")

        # Імпортуємо PortfolioCore тут, щоб уникнути циклічних імпортів під час старту
        from app.core.core import PortfolioCore  # Або from app.core import PortfolioCore залежно від шляху

        core = PortfolioCore(repo=repo)

        # Викликаємо розумну синхронізацію. Вона сама побачить, що latest_date=None,
        # і викачає дані за 30 років.
        core.sync_market_data(progress_callback=progress_callback)

        print("[BOOTSTRAP] Завантаження успішно завершено!\n")
        return True

    else:
        msg = f"База даних вже містить дані (остання дата: {latest_date}). Початкове завантаження не потрібне."
        print(f"[BOOTSTRAP] {msg}")
        logger.info(msg)
        return False