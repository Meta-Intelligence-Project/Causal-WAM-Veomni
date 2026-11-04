"""Serve native causal Cosmos3 with streaming observations and optional RTC."""

import logging

from veomni.models.diffusers.causal_cosmos3.serving.server import main


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
