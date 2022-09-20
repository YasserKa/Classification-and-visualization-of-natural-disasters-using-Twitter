from flood_detection.twitter.utils import clean_text


def test_clean_text():
    text = "CaPitaL, words https://example.com @this #that \n something?"
    expected_text = "capital words something"

    assert clean_text(text) == expected_text
