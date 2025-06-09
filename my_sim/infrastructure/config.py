"""
Чтение config.yaml → строго типизированные dataclass-ы.
Сторонние слои получают уже проверенные значения, никакого dict-фри-стайла.
"""

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import yaml
from dateutil.parser import isoparse


# ── dataclass-ы низкого уровня ────────────────────────────────────────────────
@dataclass(frozen=True)
class TeamConfig:
    n_devs: int
    wip_limit: int


@dataclass(frozen=True)
class ArrivalConfig:
    lambda0: float
    tau: int
    amplitude_quarter_end: float
    holiday_drop: float
    delta_qtr: Dict[str, float]


@dataclass(frozen=True)
class DeveloperDefaults:
    s0: float
    delta: float
    alpha: float
    beta_weekday: float
    beta_holiday: float
    w_thr: float
    tp0: float
    kappa: float
    theta_recover: float
    lognorm_sigma: float


@dataclass(frozen=True)
class StoryPointsConfig:
    zipf_gamma: float


# ── конфиг верхнего уровня ────────────────────────────────────────────────────
@dataclass(frozen=True)
class SimulationConfig:
    start_date: datetime
    horizon_days: int
    tick_minutes: int
    team: TeamConfig
    arrival: ArrivalConfig
    dev_defaults: DeveloperDefaults
    story_points: StoryPointsConfig


# ── loader ────────────────────────────────────────────────────────────────────
def _subdict(src: Dict[str, Any], key: str) -> Dict[str, Any]:
    try:
        return src[key]
    except KeyError as exc:
        raise KeyError(f"Missing '{key}' section in config.yaml") from exc


def load_conf(raw: Dict[str, Any] | str | Path) -> SimulationConfig:
    """
    Принимает либо уже-считанный dict, либо путь/yaml-строку.
    Возвращает SimulationConfig c проверенными типами.
    """
    if isinstance(raw, (str, Path)):
        raw = yaml.safe_load(Path(raw).read_text() if isinstance(raw, Path) else raw)

    cfg = SimulationConfig(
        start_date=isoparse(raw["start_date"]),
        horizon_days=int(raw["horizon_days"]),
        tick_minutes=int(raw["tick_minutes"]),
        team=TeamConfig(**_subdict(raw, "team")),
        arrival=ArrivalConfig(**_subdict(raw, "arrival")),
        dev_defaults=DeveloperDefaults(**_subdict(raw, "developer_defaults")),
        story_points=StoryPointsConfig(**_subdict(raw, "story_points")),
    )
    _validate(cfg)
    return cfg


# ── простая валидация (можно расширить pydantic, но dataclass-ам хватает) ─────
def _validate(cfg: SimulationConfig) -> None:
    assert cfg.team.n_devs > 0, "n_devs must be > 0"
    assert cfg.tick_minutes > 0, "tick_minutes must be positive"
    assert 0 <= cfg.arrival.holiday_drop <= 1, "holiday_drop must be 0…1"
    # ... добавляйте правила по мере необходимости
