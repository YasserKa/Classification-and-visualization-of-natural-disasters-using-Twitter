from src.data.preprocess import clean_text


def test_clean_text():
    text = "CaPitaL, words https://example.com @this #that \n 1923 something?"
    expected_text = "capital words that something"

    assert clean_text(text) == [expected_text]
