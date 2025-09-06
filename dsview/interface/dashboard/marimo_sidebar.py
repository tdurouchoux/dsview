import marimo as mo


def get_sidebar():
    return mo.sidebar(
        [
            mo.md("# DSView"),
            mo.nav_menu(
                {
                    "/": "Content dashboard",
                    "/searchVault": "Search vault",
                    "/uploadDashboard": "Upload dashboard",
                    "/extractionDashboard": "Extraction dashboard",
                    "/embeddingDashboard": "Embedding dashboard",
                    "/databaseExplorer": "Database explorer",
                },
                orientation="vertical",
            ),
        ]
        # TODO add link to labelling interface
    )
