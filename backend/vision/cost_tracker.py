"""Track Vision API costs and enforce budgets."""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class CostTracker:
    """Track Vision API costs with budget alerts."""

    def __init__(self, daily_budget: Optional[float] = None):
        """
        Initialize cost tracker.

        Args:
            daily_budget: Daily budget in USD (None = unlimited)
        """
        self.daily_budget = daily_budget
        self.costs: Dict[str, float] = {}  # date -> total_cost
        logger.info(f"CostTracker initialized (daily budget: ${daily_budget})")

    def add_cost(self, cost: float) -> None:
        """Add cost to today's total."""
        today = datetime.now().strftime("%Y-%m-%d")

        if today not in self.costs:
            self.costs[today] = 0.0

        self.costs[today] += cost

        logger.info(
            f"Added ${cost:.6f} to today's total "
            f"(total today: ${self.costs[today]:.4f})"
        )

        # Check budget
        if self.daily_budget and self.costs[today] >= self.daily_budget:
            logger.warning(
                f"⚠️ Daily budget exceeded! "
                f"${self.costs[today]:.4f} / ${self.daily_budget:.2f}"
            )

    def get_today_cost(self) -> float:
        """Get total cost for today."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.costs.get(today, 0.0)

    def get_total_cost(self, days: int = 7) -> float:
        """Get total cost for last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        total = 0.0

        for date_str, cost in self.costs.items():
            date = datetime.strptime(date_str, "%Y-%m-%d")
            if date >= cutoff:
                total += cost

        return total

    def is_over_budget(self) -> bool:
        """Check if today's cost exceeds budget."""
        if not self.daily_budget:
            return False

        return self.get_today_cost() >= self.daily_budget

    def get_stats(self) -> Dict[str, any]:
        """Get cost statistics."""
        return {
            "today_cost": self.get_today_cost(),
            "week_cost": self.get_total_cost(days=7),
            "month_cost": self.get_total_cost(days=30),
            "daily_budget": self.daily_budget,
            "over_budget": self.is_over_budget()
        }
