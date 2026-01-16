import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--hybrid-db",
        action="store",
        default="chroma",
        help="Hybrid DB backend to test (chroma or qdrant).",
    )


@pytest.fixture(scope="session")
def hybrid_db_backend(request: pytest.FixtureRequest) -> str:
    return str(request.config.getoption("--hybrid-db") or "chroma").lower()
