import logging
import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from app import telegram_bot


class ParseDefaultChatIdTests(unittest.TestCase):
    def test_returns_int_for_valid_value(self):
        self.assertEqual(telegram_bot._parse_default_chat_id("12345"), 12345)
        self.assertEqual(telegram_bot._parse_default_chat_id(" 999 "), 999)

    def test_returns_none_and_logs_warning_for_invalid_value(self):
        logger_name = telegram_bot.logger.name
        with self.assertLogs(logger_name, level="WARNING") as cm:
            result = telegram_bot._parse_default_chat_id("abc")
        self.assertIsNone(result)
        joined_logs = "\n".join(cm.output)
        self.assertIn("TELEGRAM_DEFAULT_CHAT_ID", joined_logs)

    def test_returns_none_for_missing_value(self):
        self.assertIsNone(telegram_bot._parse_default_chat_id(None))
        self.assertIsNone(telegram_bot._parse_default_chat_id(""))


if __name__ == "__main__":
    unittest.main()
