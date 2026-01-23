import unittest
from types import SimpleNamespace
from unittest.mock import patch

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


class ScheduleRemindersTests(unittest.TestCase):
    def test_skips_when_job_queue_missing(self):
        app = type("DummyApp", (), {"job_queue": None})()
        with self.assertLogs(telegram_bot.logger.name, level="WARNING") as cm:
            scheduled = telegram_bot._schedule_reminders(app)
        self.assertFalse(scheduled)
        self.assertIn("Job queue unavailable", "\n".join(cm.output))

    def test_runs_repeating_when_job_queue_present(self):
        class DummyJobQueue:
            def __init__(self):
                self.called_with = None

            def run_repeating(self, cb, interval, first):
                self.called_with = (cb, interval, first)

        jq = DummyJobQueue()
        app = type("DummyApp", (), {"job_queue": jq})()

        scheduled = telegram_bot._schedule_reminders(app)

        self.assertTrue(scheduled)
        self.assertEqual(jq.called_with, (telegram_bot.reminder_job, 30, 10))


class HandleMessageTests(unittest.IsolatedAsyncioTestCase):
    async def test_handle_message_uses_default_debug_flag(self):
        sent = []

        class DummyChat:
            def __init__(self):
                self.id = 42

            async def send_message(self, text, disable_web_page_preview=True):
                sent.append(text)

        class DummyMessage:
            def __init__(self, text):
                self.text = text
                self.caption = None

        class DummyUser:
            def __init__(self, user_id):
                self.id = user_id

        update = SimpleNamespace(
            effective_message=DummyMessage("hello"),
            effective_chat=DummyChat(),
            effective_user=DummyUser("u1"),
        )
        bot_data = {}
        context = SimpleNamespace(application=SimpleNamespace(bot_data=bot_data))

        called = {}

        async def fake_run_in_executor(executor, fn, *args):
            called["executor"] = executor
            called["fn"] = fn
            called["args"] = args
            return "ok"

        fake_loop = SimpleNamespace(run_in_executor=fake_run_in_executor)

        with patch("asyncio.get_running_loop", return_value=fake_loop):
            await telegram_bot.handle_message(update, context)

        self.assertEqual(called["args"], ("u1", "hello"))
        self.assertEqual(bot_data.get("user_chat_ids", {}).get("u1"), update.effective_chat.id)
        self.assertEqual(sent, ["ok"])


if __name__ == "__main__":
    unittest.main()

