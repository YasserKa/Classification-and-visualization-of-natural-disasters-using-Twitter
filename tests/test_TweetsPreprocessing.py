from src.data.text_processing import TextProcessing


def test_clean_text():
    text = "CaPitaL, words https://example.com @this #that \n 1923 something?"
    expected_text = "capital words that something"

    text_processing = TextProcessing()

    assert text_processing.clean_text(text) == [expected_text]
