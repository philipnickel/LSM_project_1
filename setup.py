#!/usr/bin/env python3
"""One-time Databricks credential bootstrap for MLflow."""

from __future__ import annotations

import mlflow


def main() -> None:
    print("[setup] Launching interactive mlflow.login() – follow the prompts.")
    mlflow.login(backend="databricks")
    print("[setup] Credentials stored. Attempting a quick verification run...")
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(experiment_id="3399934008965459")
    with mlflow.start_run():
        mlflow.log_param("setup_ping", "ok")
    print("[setup] Verification succeeded. You're ready to run experiments.")


if __name__ == "__main__":
    main()

