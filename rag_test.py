from pypdf import PdfReader 
# from fastembed import TextEmbedding
import numpy as np
from openai import OpenAI
import re


client = OpenAI()

pdf_path = "final_kyc_verification_policy.pdf"


#read pdf 
reader = PdfReader(pdf_path) 

#extract text 

text = ""

for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text 
        text += "\n"

# print(len(text))

# display the text 
# print("="*60)
# print("Extracted KYC text") 
# print(text) 
# print("="*60)

# ============================================================
# SPLIT POLICY INTO SECTION-BASED CHUNKS
# ============================================================

section_headings = [
    "1. Purpose",
    "2. ID Information Requirements",
    "3. Face Verification",
    "4. Liveness Verification",
    "5. Overall Decision Rules",
    "6. Outcome Definitions",
    "7. Example Evidence",
    "8. Scope of This Policy"
]

collection_chunks = []

current_chunk = ""

for line in text.splitlines():

    line = line.strip()

    if not line:
        continue

    # Check whether this line starts a new policy section
    if any(line.startswith(heading) for heading in section_headings):

        # Save the previous section
        if current_chunk:
            collection_chunks.append(
                current_chunk.strip()
            )

        # Start the new section
        current_chunk = line

    else:

        # Add the line to the current section
        current_chunk += "\n" + line


# Save the final section
if current_chunk:
    collection_chunks.append(
        current_chunk.strip()
    )

# ============================================================
# DISPLAY RESULT
# ============================================================

# print("\n")
# print("=" * 60)
# print("SECTION-BASED POLICY CHUNKS")
# print("=" * 60)

# for i, chunk in enumerate(collection_chunks):

#     print(f"\n--- CHUNK {i + 1} ---")
#     print(chunk)

# print("=" * 60)

# load the embedding model
# embedding_model = TextEmbedding(
#     model_name="BAAI/bge-small-en-v1.5"
# )


# ============================================================
# CREATE EMBEDDINGS FOR POLICY CHUNKS
# ============================================================

# chunk_embeddings = list(
#     embedding_model.embed(collection_chunks)
# )



# ============================================================
# CREATE OPENAI EMBEDDINGS FOR POLICY CHUNKS
# ============================================================

embedding_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=collection_chunks
)

chunk_embeddings = [
    item.embedding
    for item in embedding_response.data
]



# ============================================================
# DISPLAY EMBEDDING RESULT
# ============================================================

# print("\n")
# print("=" * 60)
# print("EMBEDDINGS")
# print("=" * 60)

# print(
#     f"Number of chunks: "
#     f"{len(collection_chunks)}"
# )

# print(
#     f"Embedding dimensions: "
#     f"{len(chunk_embeddings[0])}"
# )

# print("=" * 60)

# ============================================================
# RETRIEVE TOP 3 MOST RELEVANT POLICY CHUNKS
# ============================================================

# ============================================================
# HYBRID RETRIEVAL
# SEMANTIC SIMILARITY + KEYWORD MATCHING
# ============================================================

query = """
The face distance is 0.41, the blink count is 0, lip movement is
greater than zero, and one or more required ID fields could not
be extracted from the Photo ID.
"""



# Convert the query into an embedding
# query_embedding = list(
#     embedding_model.embed([query])
# )[0]

query_response = client.embeddings.create(
    model = "text-embedding-3-small",
    input = query
)

query_embedding = query_response.data[0].embedding


# ============================================================
# CALCULATE SIMILARITY FOR EACH POLICY CHUNK
# ============================================================
#set the keywords
query_words = set(
    re.findall(
        r"\b[a-zA-Z0-9.]+\b",
        query.lower()
    )
)


# ============================================================
# CALCULATE COMBINED SCORE FOR EACH CHUNK
# ============================================================

retrieval_results = []

for i, chunk_embedding in enumerate(chunk_embeddings):

    # --------------------------------------------------------
    # 1. SEMANTIC SIMILARITY
    # --------------------------------------------------------

    semantic_similarity = np.dot(
        query_embedding,
        chunk_embedding
    ) / (
        np.linalg.norm(query_embedding)
        * np.linalg.norm(chunk_embedding)
    )

    # --------------------------------------------------------
    # 2. KEYWORD MATCHING
    # --------------------------------------------------------

    chunk_words = set(
        re.findall(
            r"\b[a-zA-Z0-9.]+\b",
            collection_chunks[i].lower()
        )
    )

    matched_words = query_words.intersection(
        chunk_words
    )

    if len(query_words) > 0:

        keyword_score = (
            len(matched_words)
            / len(query_words)
        )

    else:

        keyword_score = 0 

    # --------------------------------------------------------
    # 3. COMBINED SCORE
    # --------------------------------------------------------

    combined_score = (
        0.80 * semantic_similarity
        + 0.20 * keyword_score
    )


    retrieval_results.append(
        (
            i,
            semantic_similarity,
            keyword_score,
            combined_score
        )
    )


# ============================================================
# SORT BY COMBINED SCORE
# ============================================================

retrieval_results.sort(
    key=lambda x: x[3],
    reverse=True
)


# ============================================================
# TAKE TOP 3
# ============================================================

top_chunks = retrieval_results[:3]


# ============================================================
# COMBINE RETRIEVED POLICY
# ============================================================

retrieved_policy = "\n\n".join(
    collection_chunks[result[0]]
    for result in top_chunks
)


# ============================================================
# DISPLAY RETRIEVAL RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("HYBRID RETRIEVAL RESULT")
print("=" * 60)

print(
    f"Query: {query}"
)

for rank, (
    chunk_index,
    semantic_similarity,
    keyword_score,
    combined_score
) in enumerate(
    top_chunks,
    start=1
):

    print(
        f"\n--- RESULT {rank} ---"
    )

    print(
        f"Chunk: {chunk_index + 1}"
    )

    print(
        f"Semantic similarity: "
        f"{semantic_similarity:.4f}"
    )

    print(
        f"Keyword score: "
        f"{keyword_score:.4f}"
    )

    print(
        f"Combined score: "
        f"{combined_score:.4f}"
    )

    print("\nRetrieved policy:")

    print(
        collection_chunks[chunk_index]
    )

print("=" * 60)



# ============================================================
# SORT FROM HIGHEST TO LOWEST SIMILARITY
# ============================================================

# similarities.sort(
#     key=lambda x: x[1],
#     reverse=True
# )


# ============================================================
# TAKE TOP 3 CHUNKS
# ============================================================

# top_chunks = similarities[:3]

# # ============================================================
# # COMBINE RETRIEVED POLICY CHUNKS
# # ============================================================

# retrieved_policy = "\n\n".join(
#     collection_chunks[chunk_index]
#     for chunk_index, similarity in top_chunks
# )


# ============================================================
# TEST KYC EVIDENCE
# ============================================================

# test case 1
# kyc_evidence = """
# Name: Rahul Sharma
# DOB: 15-06-1998
# ID Number: DEMO123456

# Face distance: 0.72

# Blink count: 1
# Lip movement count: 1

# Liveness status: PASS
# """

# test case 2 

kyc_evidence = """
Name: Rahul Sharma
DOB: Not detected
ID Number: DEMO123456

Face distance: 0.41

Blink count: 0
Lip movement count: 5

Liveness status: REVIEW
Face detected in video: True
Multiple faces detected in video: False
"""

# ============================================================
# CREATE LLM PROMPT
# ============================================================

prompt = f"""
You are an AI assistant supporting a bank's KYC verification team.

Evaluate the provided KYC evidence strictly according to the retrieved
KYC policy.

IMPORTANT INSTRUCTIONS:

1. Use only the KYC evidence and retrieved policy provided below.
2. Do not invent facts, rules, thresholds, or verification results.
3. Do not change or reinterpret the policy rules.
4. If the policy says PASS, return PASS.
5. If the policy says REVIEW, return REVIEW.
6. If the policy says FAIL, return FAIL.
7. Do not treat REVIEW as FAIL.
8. Do not treat FAIL as REVIEW or PASS.
9. Clearly distinguish between a confirmed failure and an inconclusive case.
10. The explanation must be concise, professional, and suitable for a bank
    verification team.
11. Do not unnecessarily mention embeddings, vectors, chunks, cosine
    similarity, model names, or other internal technical details.
12. Base the reason directly on the supplied evidence and policy.
13. Do not make assumptions about information that is not provided.
14. Do not introduce specific verification procedures, corrective actions,
    or requirements unless they are supported by the provided KYC policy.
15. When multiple conditions are present, select FAIL if any
    conclusive FAIL condition applies. REVIEW conditions must not
    override a conclusive FAIL condition.

KYC EVIDENCE:
{kyc_evidence}

RETRIEVED KYC POLICY:
{retrieved_policy}

Return the result exactly in this format:

Decision: PASS / REVIEW / FAIL

Reason:
Give a concise explanation of why this decision follows from
the KYC evidence and the retrieved policy.

Recommended Action:
State the appropriate next action for the verification team.
"""


# ============================================================
# SEND TO QWEN3 --> tested initally wid this
# ============================================================

# ============================================================
# SEND PROMPT TO OPENAI
# ============================================================

response = client.responses.create(
    model="gpt-5-mini",
    input=prompt
)


# ============================================================
# DISPLAY LLM RESULT
# ============================================================

print("\n")
print("=" * 60)
print("LLM KYC ASSESSMENT")
print("=" * 60)

print(response.output_text)

print("=" * 60)


# ============================================================
# DISPLAY RETRIEVED CHUNKS
# ============================================================

# print("\n")
# print("=" * 60)
# print("RETRIEVAL RESULT")
# print("=" * 60)

# print(
#     f"Query: {query}"
# )

# for rank, (chunk_index, similarity) in enumerate(
#     top_chunks,
#     start=1
# ):

#     print(
#         f"\n--- RESULT {rank} ---"
#     )

#     print(
#         f"Chunk: {chunk_index + 1}"
#     )

#     print(
#         f"Similarity: {similarity:.4f}"
#     )

#     print("\nRetrieved policy:")

#     print(
#         collection_chunks[chunk_index]
#     )

# print("=" * 60)