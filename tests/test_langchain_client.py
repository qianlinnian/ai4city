from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from utils import llm_client


ROOT = Path(__file__).resolve().parents[1]


class LangChainClientTestCase(unittest.TestCase):
    def test_text_chat_uses_langchain_runnable_chain(self) -> None:
        fake_model = RunnableLambda(lambda _: AIMessage(content='{"ok": true}'))
        with (
            patch("utils.llm_client.use_mock_llm", return_value=False),
            patch("utils.llm_client._build_model", return_value=fake_model),
        ):
            result = llm_client.chat("system", "user")

        self.assertEqual(result, '{"ok": true}')

    def test_multimodal_chat_builds_langchain_messages(self) -> None:
        fake_model = Mock()
        fake_model.invoke.return_value = AIMessage(content="multimodal-ok")
        with (
            patch("utils.llm_client.use_mock_llm", return_value=False),
            patch("utils.llm_client._build_model", return_value=fake_model),
        ):
            result = llm_client.chat_with_image(
                "system",
                "inspect panorama",
                ROOT / "data" / "p1.jpg",
            )

        self.assertEqual(result, "multimodal-ok")
        messages = fake_model.invoke.call_args.args[0]
        self.assertEqual(messages[0].content, "system")
        self.assertEqual(messages[1].content[0]["type"], "text")
        self.assertEqual(messages[1].content[1]["type"], "image_url")
        self.assertTrue(
            messages[1].content[1]["image_url"]["url"].startswith(
                "data:image/jpeg;base64,"
            )
        )

    def test_multi_image_message_includes_each_view_metadata(self) -> None:
        views = [
            {
                "view_id": "overview",
                "output_path": str(ROOT / "data" / "p1.jpg"),
                "width": 2048,
                "height": 1024,
                "is_overview": True,
            },
            {
                "view_id": "yaw_090",
                "output_path": str(ROOT / "data" / "p2.jpg"),
                "yaw": 90,
                "pitch": 0,
                "fov": 90,
                "width": 1024,
                "height": 1024,
                "is_overview": False,
            },
        ]
        messages = llm_client.build_multi_image_messages("system", "inventory", views)
        content = messages[1].content
        image_blocks = [item for item in content if item["type"] == "image_url"]
        text = "\n".join(
            item["text"] for item in content if item["type"] == "text"
        )
        self.assertEqual(len(image_blocks), 2)
        self.assertIn("overview", text)
        self.assertIn("yaw=90", text)


if __name__ == "__main__":
    unittest.main()
