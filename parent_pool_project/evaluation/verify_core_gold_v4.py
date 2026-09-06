#!/usr/bin/env python3
"""Run the independent verifier with v4 namespaces active."""

import filter_core_gold_v4  # noqa: F401
import verify_core_gold


if __name__ == "__main__":
    raise SystemExit(verify_core_gold.main())
