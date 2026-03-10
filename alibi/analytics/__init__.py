"""Analytics module for spending analysis and pattern detection."""

from alibi.analytics.patterns import (
    MonthlyTrend,
    CategoryTrend,
    SpendingInsights,
    analyze_spending_patterns,
    compare_periods,
)
from alibi.analytics.anomalies import (
    SpendingAnomaly,
    detect_anomalies,
)
from alibi.analytics.spending import (
    VendorSpend,
    MonthlySpend,
    ItemFrequency,
    SeasonalPattern,
    spending_by_vendor,
    spending_by_month,
    item_frequency,
    seasonal_patterns,
)
from alibi.analytics.vendors import (
    VendorAlias,
    VendorDeduplicationReport,
    vendor_deduplication_report,
)
from alibi.analytics.subscriptions import (
    SubscriptionPattern,
    detect_subscriptions,
    mark_subscriptions,
    get_upcoming_subscriptions,
)

__all__ = [
    # Patterns
    "MonthlyTrend",
    "CategoryTrend",
    "SpendingInsights",
    "analyze_spending_patterns",
    "compare_periods",
    # Anomalies
    "SpendingAnomaly",
    "detect_anomalies",
    # V2 spending analytics
    "VendorSpend",
    "MonthlySpend",
    "ItemFrequency",
    "SeasonalPattern",
    "spending_by_vendor",
    "spending_by_month",
    "item_frequency",
    "seasonal_patterns",
    # Vendor deduplication
    "VendorAlias",
    "VendorDeduplicationReport",
    "vendor_deduplication_report",
    # Subscriptions
    "SubscriptionPattern",
    "detect_subscriptions",
    "mark_subscriptions",
    "get_upcoming_subscriptions",
]
