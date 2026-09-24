import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from kalshi_quant.data.kalshi_client import KalshiCredentials


@pytest.fixture
def throwaway_pem() -> str:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    ).decode()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no stray .env picked up by load_dotenv
    for var in ("KALSHI_API_KEY_ID", "KALSHI_PRIVATE_KEY", "KALSHI_PRIVATE_KEY_PATH"):
        monkeypatch.delenv(var, raising=False)


def test_loads_key_from_env_text_with_real_newlines(monkeypatch, throwaway_pem):
    monkeypatch.setenv("KALSHI_API_KEY_ID", "test-id")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY", throwaway_pem)
    creds = KalshiCredentials.from_env()
    assert creds.key_id == "test-id"


def test_loads_key_from_env_text_with_escaped_newlines(monkeypatch, throwaway_pem):
    monkeypatch.setenv("KALSHI_API_KEY_ID", "test-id")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY", throwaway_pem.replace("\n", "\\n"))
    assert KalshiCredentials.from_env().key_id == "test-id"


def test_loads_key_from_file_path(monkeypatch, throwaway_pem, tmp_path):
    key_file = tmp_path / "key.pem"
    key_file.write_text(throwaway_pem)
    monkeypatch.setenv("KALSHI_API_KEY_ID", "test-id")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", str(key_file))
    assert KalshiCredentials.from_env().key_id == "test-id"


def test_missing_credentials_raise(monkeypatch):
    monkeypatch.setenv("KALSHI_API_KEY_ID", "test-id")
    with pytest.raises(RuntimeError):
        KalshiCredentials.from_env()
