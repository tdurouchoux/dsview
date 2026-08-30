from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from dsview.notification import email_sender


@dataclass
class FakeNotificationConfig:
    smtp_host: str = "smtp.example.com"
    smtp_port: int = 587
    sender_name: str = "DSView Weekly Digest"
    smtp_user: str = "digest@example.com"
    smtp_password: str = "secret"
    email_to: str = "reader@example.com"


@pytest.fixture
def fake_smtp(monkeypatch):
    monkeypatch.setattr(email_sender, "notification_config", FakeNotificationConfig())

    smtp_instance = MagicMock()
    smtp_instance.__enter__.return_value = smtp_instance
    smtp_class = MagicMock(return_value=smtp_instance)
    monkeypatch.setattr(email_sender.smtplib, "SMTP", smtp_class)

    return smtp_class, smtp_instance


def test_send_email_connects_authenticates_and_sends(fake_smtp):
    smtp_class, smtp_instance = fake_smtp

    email_sender.send_email("Weekly digest", "<html>body</html>")

    smtp_class.assert_called_once_with("smtp.example.com", 587)
    smtp_instance.starttls.assert_called_once()
    smtp_instance.login.assert_called_once_with("digest@example.com", "secret")
    smtp_instance.send_message.assert_called_once()

    sent_message = smtp_instance.send_message.call_args[0][0]
    assert sent_message["Subject"] == "Weekly digest"
    assert sent_message["To"] == "reader@example.com"
    assert "digest@example.com" in sent_message["From"]
    assert "DSView Weekly Digest" in sent_message["From"]
    assert "<html>body</html>" in sent_message.get_payload()[0].get_payload()
