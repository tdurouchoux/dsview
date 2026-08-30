import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import logfire

from dsview.config import notification_config

logger = logging.getLogger(__name__)


@logfire.instrument("Send email", extract_args=["subject"])
def send_email(subject: str, html_body: str) -> None:
    config = notification_config

    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = f"{config.sender_name} <{config.smtp_user}>"
    message["To"] = config.email_to
    message.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP(config.smtp_host, config.smtp_port) as smtp:
        smtp.starttls()
        smtp.login(config.smtp_user, config.smtp_password)
        smtp.send_message(message)

    logger.info("Weekly digest email sent to %s", config.email_to)
