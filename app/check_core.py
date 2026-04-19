import os
import logging
from app.core import PortfolioCore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    print("=== 🚀 Фінальна перевірка Core (Оркестратора) ===")

    # 1. ГАРАНТОВАНО ВКАЗУЄМО ШЛЯХ ДО ПЛАГІНІВ
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plugins_folder = os.path.join(current_dir, "plugins")

    print(f"🔍 [DEBUG] Шукаємо плагіни у: {plugins_folder}")

    # Ініціалізуємо Core з примусовим шляхом
    core = PortfolioCore(plugins_dir=plugins_folder)

    # Перевіряємо, чи бачить Core плагіни до запуску конвеєра
    found_plugins = core.get_plugins()
    print(f"🔍 [DEBUG] Знайдені плагіни: {list(found_plugins.keys())}")

    if not found_plugins:
        print("\n❌ Плагіни досі не знайдено! Перевір Крок 1.")
        return

    print("\n[Старт конвеєра] Запит даних -> Оптимізація -> Бектестинг...")

    try:
        opt_result, bt_report = core.run_and_backtest(
            method="plugin",
            plugin_name="EqualWeightOptimizer",
            tickers=["AAPL", "MSFT", "GOOGL", "AMZN"],
            start_date="2019-01-01",
            train_end="2021-01-01",
            end_date="2023-01-01",
            initial_capital=10000.0,
            rebalance_every=4,
            save=False
        )

        print("\n✅ Конвеєр відпрацював успішно!")
        print("=" * 45)
        print("⚖️ Сформовані ваги портфеля:")
        for ticker, weight in opt_result.weights.items():
            print(f"  - {ticker}: {weight * 100:.2f}%")

        r = bt_report.results[0]
        m = r.metrics
        print("\n📊 Результати ретроспективного тестування:")
        print(f"CAGR (Річна дох.):  {m.cagr * 100:+.2f}%")
        print(f"Макс. просадка:     {m.max_drawdown * 100:+.2f}%")
        print(f"Коефіцієнт Шарпа:   {m.sharpe_ratio:.2f}")

        if r.benchmark:
            bm = r.benchmark.metrics
            print(f"\n📈 Per-Portfolio Benchmark (EW):")
            print(f"CAGR: {bm.cagr * 100:+.2f}% | Шарп: {bm.sharpe_ratio:.2f}")
        print("=" * 45)

    except Exception as e:
        print(f"\n❌ Сталася помилка під час виконання: {e}")


if __name__ == "__main__":
    main()