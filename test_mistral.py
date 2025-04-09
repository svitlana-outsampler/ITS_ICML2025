import unittest
from mistral import Mistral  # Assuming Mistral is defined in mistral.py

class TestMistral(unittest.TestCase):
    def setUp(self):
        self.chat = Mistral()

    def test_ask_method(self):
        response = self.chat.ask(
            "ou_process.png",
            "Describe the time series in three sentences. "
            "First sentence: describe increasing/decreasing/flat pattern. "
            "Second sentence: describe the overall trend and the noise. "
            "Third sentence: describe local and globe extrema."
        )
        print(response)  # For debugging purposes
        content = response['choices'][0]['message']['content']
        print(len(content))
        self.assertTrue(len(content) > 10)
    
if __name__ == '__main__':
    unittest.main()