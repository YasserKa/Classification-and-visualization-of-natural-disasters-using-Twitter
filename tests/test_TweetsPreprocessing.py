from flood_detection.data.text_processing import TextProcessing


def test_clean_text():
    text = "CaPitaL, words https://example.com @this #that \n something?"
    expected_text = "capital words something"

    text_processing = TextProcessing()

    assert text_processing.clean_text(text) == [expected_text]
