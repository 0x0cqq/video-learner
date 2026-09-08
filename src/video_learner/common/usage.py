"""按服务端返回值汇总本次转换用量，并使用用户单价估算费用。"""

from datetime import datetime, timedelta, timezone

from video_learner.common.config import Config, ModelPrice


def request_cost(usage: dict, price: ModelPrice) -> float | None:
    """按已知用量估价；缓存细分未知按普通输入价，DeepSeek 时段按请求开始时间。"""
    if price.audio_per_second is not None:
        seconds = usage.get("audio_seconds")
        return None if seconds is None else seconds * price.audio_per_second
    incoming, outgoing = usage.get("input_tokens"), usage.get("output_tokens")
    if incoming is None or outgoing is None:
        return None
    if price.input_per_million is None or price.output_per_million is None:
        return None
    cached = 0
    if price.cached_input_per_million is not None:
        cached = usage.get("cached_input_tokens") or 0
    cost = (
        (incoming - cached) * price.input_per_million
        + cached * (price.cached_input_per_million or 0)
        + outgoing * price.output_per_million
    ) / 1_000_000
    if price.deepseek_off_peak_multiplier is not None and usage.get("started_at"):
        local = datetime.fromisoformat(usage["started_at"]).astimezone(timezone(timedelta(hours=8)))
        peak = local.weekday() < 5 and (9 <= local.hour < 12 or 14 <= local.hour < 18)
        if not peak:
            cost *= price.deepseek_off_peak_multiplier
    return cost


def summarize_usage(records: list[dict], config: Config) -> dict:
    """汇总一次运行的所有尝试，包含修复重试；缺失用量与缺失报价分别记录。

    思考与缓存 token 是输出/输入的子项，不再加到总 token。费用总计仅在全部
    请求可计价时提供；部分已知费用单列，避免误读为整次转换价格。
    """
    requests = {}
    for record in records:
        asr = record["stage"].startswith("asr_")
        identity = (asr, record["call"])
        item = requests.setdefault(identity, {})
        if record["status"] == "running":
            item["started_at"] = record.get("timestamp_utc")
        item.update(record)
    groups = {}
    for (asr, _), usage in requests.items():
        provider = "qwen" if asr else config.provider
        model = usage.get("model", config.asr_qwen_model if asr else config.model)
        key = f"{provider}:{model}"
        row = groups.setdefault(
            key,
            {
                "provider": provider,
                "model": model,
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_input_tokens": 0,
                "audio_seconds": 0.0,
                "missing_token_usage": 0,
                "unpriced_requests": 0,
                "unknown_cache_requests": 0,
                "estimated_known_cost": None,
                "currency": config.prices[key].currency if key in config.prices else None,
            },
        )
        row["requests"] += 1
        for field in ("input_tokens", "output_tokens", "cached_input_tokens", "audio_seconds"):
            if usage.get(field) is not None:
                row[field] += usage[field]
        if usage.get("input_tokens") is None or usage.get("output_tokens") is None:
            row["missing_token_usage"] += 1
        price = config.prices.get(key)
        if price and price.cached_input_per_million is not None:
            if usage.get("cached_input_tokens") is None:
                row["unknown_cache_requests"] += 1
        cost = request_cost(usage, price) if price else None
        if cost is None:
            row["unpriced_requests"] += 1
        else:
            row["estimated_known_cost"] = (row["estimated_known_cost"] or 0) + cost
    totals = {}
    for row in groups.values():
        row["total_tokens"] = row["input_tokens"] + row["output_tokens"]
        if row["estimated_known_cost"] is not None:
            currency = row["currency"]
            totals[currency] = totals.get(currency, 0) + row["estimated_known_cost"]
    return {
        "models": list(groups.values()),
        "input_tokens": sum(r["input_tokens"] for r in groups.values()),
        "output_tokens": sum(r["output_tokens"] for r in groups.values()),
        "total_tokens": sum(r["total_tokens"] for r in groups.values()),
        "missing_token_usage": sum(r["missing_token_usage"] for r in groups.values()),
        "estimated_known_cost": totals,
        "estimate_complete": bool(requests)
        and all(r["unpriced_requests"] == 0 for r in groups.values()),
        "prices": {
            key: price.model_dump() for key, price in config.prices.items() if key in groups
        },
    }
