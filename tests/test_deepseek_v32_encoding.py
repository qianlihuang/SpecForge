import importlib.machinery
import importlib.util
import os
import unittest

# Load sglang encoding module directly
ENC_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "sglang", "python", "sglang", "srt", "entrypoints", "openai", "encoding_dsv32.py")
)
loader = importlib.machinery.SourceFileLoader("encoding_dsv32", ENC_PATH)
spec = importlib.util.spec_from_loader(loader.name, loader)
enc = importlib.util.module_from_spec(spec)
loader.exec_module(enc)


class TestDeepSeekV32Encoding(unittest.TestCase):
    def setUp(self):
        # sample conversations (inputs): must END with user (input side)
        self.non_thinking = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's 2+2?"},
        ]
        self.thinking = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain your reasoning for answering this."},
        ]

    def test_render_non_thinking_chat(self):
        out = enc.encode_messages(self.non_thinking, thinking_mode="chat")
        # Should include the special user/assistant markers and not the thinking start token
        self.assertIn("<｜User｜>", out)
        self.assertIn("<｜Assistant｜>", out)
        self.assertNotIn("<think>", out)
        self.assertIn("</think>", out)  # chat path appends thinking end token

    def test_render_thinking_mode(self):
        out = enc.encode_messages(self.thinking, thinking_mode="thinking")
        # thinking mode for input should append the thinking start token
        self.assertIn("<think>", out)
        self.assertIn("<｜User｜>", out)
        self.assertIn("<｜Assistant｜>", out)


if __name__ == "__main__":
    unittest.main()
