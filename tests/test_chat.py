import unittest

from app.chat import split_prefixed_requests


class SplitPrefixedRequestsTests(unittest.TestCase):
    def test_defaults_to_life_when_no_prefix(self):
        self.assertEqual(
            split_prefixed_requests("plan my week"),
            [("life", "plan my week")],
        )

    def test_override_default_agent(self):
        self.assertEqual(
            split_prefixed_requests("plan my week", default_agent="general"),
            [("general", "plan my week")],
        )

    def test_splits_multiple_prefixed_chunks(self):
        text = "life: list reminders; ds: next lesson"
        self.assertEqual(
            split_prefixed_requests(text),
            [("life", "list reminders"), ("ds", "next lesson")],
        )


if __name__ == "__main__":
    unittest.main()
