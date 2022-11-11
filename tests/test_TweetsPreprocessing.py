from flood_detection.data.preprocess import Preprocess


def test_clean_text():
    text = "CaPitaL, words https://example.com @this #that \n 1923 something?"
    expected_text = "capital words that something"
    preprocess = Preprocess()
    assert preprocess.clean_text(text) == [expected_text]
