"""Compatibility module for OpenEnv local validator expectations."""

import os

import uvicorn
from fastapi import FastAPI

app = FastAPI(title="Email Triage OpenEnv Compatibility")


def main() -> None:
	port = int(os.environ.get("PORT", "7860"))
	uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
	main()
