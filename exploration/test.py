from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

# Two lists of sentences
sentences1 = [
    "The cat sits outside",
    "blabla",
    "SMHI: Bälten: Varning klass 1, kuling.",
]

sentences2 = [
    "The dog plays in the garden",
    "A woman watches TV",
    "SMHI: Skagerack: Varning klass 1, kuling. https://goo.gl/fb/aSp3I0",
]

# Compute embedding for both lists
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)

# Compute cosine-similarities
cosine_scores = util.cos_sim(embeddings1, embeddings2)

# Output the pairs with their score
for i in range(len(sentences1)):
    print(
        "{} \t\t {} \t\t Score: {:.4f}".format(
            sentences1[i], sentences2[i], cosine_scores[i][i]
        )
    )
