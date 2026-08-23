from pathlib import Path

import marimo
import typer
import uvicorn
from fastapi import FastAPI

INTERFACE_DIR = Path("dsview/interface/")

# Create typer app
app_cli = typer.Typer(help="DSView Dashboard Server")

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


@app_cli.command()
def run_dashboard(
    host: str = typer.Option("0.0.0.0", help="Host to bind the server to"),
    port: int = typer.Option(2718, help="Port to bind the server to"),
):
    """Start the DSView dashboard server."""
    uvicorn.run(app, host=host, port=port, log_level="info")


def main():
    app_cli()


# Run the server
if __name__ == "__main__":
    main()
