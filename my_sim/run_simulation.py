#!/usr/bin/env python
"""
Запуск полной симуляции.

Алгоритм:
1. Конфиг → dataclass (infrastructure.config)
2. Environment + очереди
3. Arrival-процесс (задачи), Planner-процесс (распределение/WIP)
4. Developers (domain.Developer)
5. ParquetLogger – сохраняет события
6. После run() → pandas → визуализация
"""

from pathlib import Path
import simpy

from infrastructure.config import load_conf
from infrastructure.logger import ParquetLogger
from simulation import (make_environment,
                        task_arrival_process,
                        sprint_planner_process)
from domain import Developer, DevParams, FatigueParams
from visual.plots import plot_velocity, plot_tp_hist


# ──────────────────────────────────────────────────────────────────────────────
def _build_devs(conf) -> list[Developer]:
    """Создаём dev-объекты по шаблону из YAML."""
    devs = []
    for pid in range(conf.team.n_devs):
        f_params = FatigueParams(
            alpha=conf.dev_defaults.alpha,
            beta_work=conf.dev_defaults.beta_weekday,
            beta_rest=conf.dev_defaults.beta_holiday,
            w_thr=conf.dev_defaults.w_thr,
            tp0=conf.dev_defaults.tp0,
            delta=conf.dev_defaults.delta,
        )
        d_params = DevParams(
            s0=conf.dev_defaults.s0,
            fatigue=f_params,
            lognorm_sigma=conf.dev_defaults.lognorm_sigma,
            kappa=conf.dev_defaults.kappa,
            theta_recover=conf.dev_defaults.theta_recover,
        )
        devs.append(Developer(pid, d_params, seed=pid))
    return devs


# ──────────────────────────────────────────────────────────────────────────────
def main(cfg_path: str | Path = "config.yaml") -> None:
    conf = load_conf(Path(cfg_path))

    # 1. SimPy-мир + очереди
    env = make_environment(conf)
    backlog = simpy.Store(env)
    done_q = simpy.Store(env)

    # 2. Доменные акторы
    devs = _build_devs(conf)

    # 3. Процессы SimPy
    env.process(task_arrival_process(env, conf, backlog))
    env.process(sprint_planner_process(env, conf, backlog, done_q, devs))

    # 4. Logger на краю
    logger = ParquetLogger(Path("simlog"))
    env.process(logger.collect(done_q))

    # 5. Run!
    horizon_sec = conf.horizon_days * 24 * 3600
    print(f"⏳  Running {conf.horizon_days} days ({horizon_sec:,} sec)…")
    env.run(until=horizon_sec)
    print("✅  Simulation finished.")

    # 6. Post-processing
    df = logger.to_dataframe()
    if df.empty:
        print("❗  Empty log — что-то пошло не так.")
        return

    plot_velocity(df)
    df_year = df[df["sim_time"] <= 365 * 24 * 3600]
    plot_velocity(df_year)
    plot_tp_hist(df)


if __name__ == "__main__":
    main()
