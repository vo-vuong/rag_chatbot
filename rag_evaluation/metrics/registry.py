"""
Metric registry for automatic discovery and registration.

Enables easy extension of the evaluation framework with new metrics
using a decorator-based registration pattern.
"""

import logging
from typing import Dict, List, Optional, Type

from rag_evaluation.base.metric_interface import RetrievalMetric

logger = logging.getLogger(__name__)


class MetricRegistry:
    """
    Registry for retrieval metric implementations.

    Provides automatic discovery and instantiation of registered metrics.
    Metrics can be registered using the @register_metric decorator.

    Example:
        @register_metric("mrr")
        class MRRMetric(RetrievalMetric):
            name = "Mean Reciprocal Rank"
            short_name = "mrr"
            ...
    """

    _metrics: Dict[str, Type[RetrievalMetric]] = {}

    @classmethod
    def register(cls, short_name: str, metric_class: Type[RetrievalMetric]) -> None:
        """
        Register a metric class.

        Args:
            short_name: Short identifier for the metric (e.g., "hit", "recall")
            metric_class: The metric class to register
        """
        cls._metrics[short_name.lower()] = metric_class
        logger.debug(f"Registered metric: {short_name} -> {metric_class.__name__}")

    @classmethod
    def get(cls, short_name: str) -> Optional[Type[RetrievalMetric]]:
        """
        Get a metric class by short name.

        Args:
            short_name: Short identifier for the metric

        Returns:
            The metric class, or None if not found
        """
        return cls._metrics.get(short_name.lower())

    @classmethod
    def get_instance(cls, short_name: str) -> Optional[RetrievalMetric]:
        """
        Get an instantiated metric by short name.

        Args:
            short_name: Short identifier for the metric

        Returns:
            An instance of the metric, or None if not found
        """
        metric_class = cls.get(short_name)
        if metric_class:
            return metric_class()
        return None

    @classmethod
    def list_metrics(cls) -> List[str]:
        """
        List all registered metric short names.

        Returns:
            List of registered metric short names
        """
        return list(cls._metrics.keys())

    @classmethod
    def get_all_instances(cls) -> List[RetrievalMetric]:
        """
        Get instances of all registered metrics.

        Returns:
            List of instantiated metrics
        """
        return [metric_class() for metric_class in cls._metrics.values()]


def register_metric(short_name: str):
    """
    Decorator to register a metric class with the registry.

    Args:
        short_name: Short identifier for the metric

    Example:
        @register_metric("ndcg")
        class NDCGMetric(RetrievalMetric):
            name = "Normalized Discounted Cumulative Gain"
            short_name = "ndcg"
            ...
    """
    def decorator(cls: Type[RetrievalMetric]) -> Type[RetrievalMetric]:
        MetricRegistry.register(short_name, cls)
        return cls
    return decorator
