from code.twitter.utils import clean_text

# distilbert-base-uncased 
# Accuracy	F1
# 0.954733	0.954659

# lbert-base-uncased 
# Accuracy	F1
# distilbert-base-uncased Accuracy	F1
def test_clean_text():
    text = "CaPitaL, words https://example.com @this #that \n something?"
    expected_text = "capital words something"

    assert clean_text(text) == expected_text
