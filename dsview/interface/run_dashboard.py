from pathlib import Path

import marimo
from fastapi import FastAPI
import uvicorn

INTERFACE_DIR = Path("dsview/interface/")
HOST = "0.0.0.0"
HOST = "localhost"
PORT = 8080

# Create a marimo ASGI app
server = (
    marimo.create_asgi_app()
    .with_app(path="", root=INTERFACE_DIR / "dashboard" / "content_dashboard.py")
    .with_app(path="/searchVault", root=INTERFACE_DIR / "dashboard" / "search_vault.py")
    .with_app(
        path="/uploadDashboard",
        root=INTERFACE_DIR / "dashboard" / "upload_dashboard.py",
    )
    .with_app(
        path="/extractionDashboard",
        root=INTERFACE_DIR / "dashboard" / "extraction_dashboard.py",
    )
    .with_app(
        path="/embeddingDashboard",
        root=INTERFACE_DIR / "dashboard" / "embedding_dashboard.py",
    )
    .with_app(
        path="/databaseExplorer",
        root=INTERFACE_DIR / "dashboard" / "db_explorer.py",
    )
)

# Create a FastAPI app
app = FastAPI(
    title="DSView dashboard and analytics platform",
    description="A comprehensive data analytics platform built with marimo.",
    version="1.0.0",
)

# Mount the marimo server
app.mount("", server.build())


def main():
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


# Run the server
if __name__ == "__main__":
    main()
