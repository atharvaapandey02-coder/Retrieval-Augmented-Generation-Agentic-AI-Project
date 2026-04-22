import os
import json
import faiss
import torch
import numpy as np
from PIL import Image
import open_clip
from google import genai # we will change this as we have genai already installed using requirements.txt just alias here when using

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INDEX_PATH = "faiss.index"
META_PATH  = "metadata.json"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SIMILARITY_THRESHOLD = 0.75   # cosine similarity cut-off

# ─────────────────────────────────────────────
# KNOWLEDGE BASE  (Augmentation layer)
# Each enrolled person_id maps to a rich profile
# used to ground the LLM generation step.
# ─────────────────────────────────────────────
PERSON_PROFILES = {
    "akshay": {
        "full_name":    "Akshay Kumar",
        "profession":   "Actor, Film Producer",
        "nationality":  "Canadian-Indian",
        "known_for":    "Action & comedy films; philanthropy; National Award winner",
        "access_level": "VIP",
        "department":   "Entertainment"
    },
    "alia": {
        "full_name":    "Alia Bhatt",
        "profession":   "Actress, Film Producer",
        "nationality":  "Indian-British",
        "known_for":    "Versatile dramatic roles; Gangubai Kathiawadi, Raazi; Filmfare Award winner",
        "access_level": "VIP",
        "department":   "Entertainment"
    },
    "deepika": {
        "full_name":    "Deepika Padukone",
        "profession":   "Actress",
        "nationality":  "Indian",
        "known_for":    "Piku, Padmaavat; TIME100 list; mental-health advocate",
        "access_level": "VIP",
        "department":   "Entertainment"
    },
    "kohli": {
        "full_name":    "Virat Kohli",
        "profession":   "Cricketer (Batsman)",
        "nationality":  "Indian",
        "known_for":    "Former Indian captain; one of cricket's greatest batsmen; 70+ international centuries",
        "access_level": "VIP",
        "department":   "Sports"
    },
    "anirkhan": {
        "full_name":    "Aamir Khan",
        "profession":   "Actor, Director, Producer",
        "nationality":  "Indian",
        "known_for":    "Perfectionist of Bollywood; Dangal, 3 Idiots, Lagaan; Padma Bhushan recipient",
        "access_level": "VIP",
        "department":   "Entertainment"
    },
    "vaibhav": {
        "full_name":    "Vaibhav Suryavanshi",
        "profession":   "Cricketer",
        "nationality":  "Indian",
        "known_for":    "High run scoring opener",
        "access_level": "VIP",
        "department":   "Sports"
    },
    "shraddha": {
        "full_name":    "Shraddha Kapoor",
        "profession":   "Actress, Singer",
        "nationality":  "Indian",
        "known_for":    "Aashiqui 2, Stree franchise; one of the most-followed Indian celebrities online",
        "access_level": "VIP",
        "department":   "Entertainment"
    },
    "rohitsharma": {
        "full_name":    "Rohit Sharma",
        "profession":   "Cricketer (Batsman, Captain)",
        "nationality":  "Indian",
        "known_for":    "Indian cricket captain; highest ODI individual score (264); 5 IPL titles",
        "access_level": "VIP",
        "department":   "Sports"
    },
    "ranveer": {
        "full_name":    "Ranveer Singh",
        "profession":   "Actor",
        "nationality":  "Indian",
        "known_for":    "High-energy performances; Bajirao Mastani, Gully Boy, 83; Filmfare Award winner",
        "access_level": "VIP",
        "department":   "Entertainment"
    },
    "kartikaaryan": {
        "full_name":    "Kartik Aaryan",
        "profession":   "Actor",
        "nationality":  "Indian",
        "known_for":    "Pyaar Ka Punchnama, Bhool Bhulaiyaa franchise; top-grossing actor of his generation",
        "access_level": "VIP",
        "department":   "Entertainment"
    },
    "bachachan": {
        "full_name":    "Amitabh Bachchan",
        "profession":   "Actor, Film Producer, Television Host",
        "nationality":  "Indian",
        "known_for":    "'Shehenshah of Bollywood'; Padma Vibhushan; host of Kaun Banega Crorepati",
        "access_level": "VIP",
        "department":   "Entertainment"
    },
}

# ─────────────────────────────────────────────
# LOAD CLIP MODEL  (Retrieval embedding model)
# ─────────────────────────────────────────────
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name="RN50",
    pretrained="openai"
)
model = model.to(DEVICE)
model.eval()

EMBED_DIM = model.visual.output_dim

# ─────────────────────────────────────────────
# VECTOR STORE  (FAISS)
# ─────────────────────────────────────────────
def load_vector_store():
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r") as f:
            metadata = json.load(f)
        print("Loaded existing FAISS index and metadata")
    else:
        index = faiss.IndexFlatIP(EMBED_DIM)   # inner product = cosine on unit vectors
        metadata = []
        print("Created new FAISS index")
    return index, metadata

index, metadata = load_vector_store()

def save_vector_store():
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print("Vector store saved")

# ─────────────────────────────────────────────
# STEP 1 – RETRIEVAL
# ─────────────────────────────────────────────
def image_to_embedding(image_path: str) -> np.ndarray:
    """Encode an image into a normalised CLIP embedding."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model.encode_image(image)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")[0]


def enroll_person(person_id: str, image_path: str):
    """Enroll a person by adding their embedding + metadata to FAISS."""
    emb = image_to_embedding(image_path)
    index.add(np.array([emb]))
    metadata.append({"person_id": person_id, "image_path": image_path})
    save_vector_store()
    print(f"Enrolled {person_id}")


def retrieve(query_image_path: str) -> dict:
    """
    RETRIEVAL STAGE
    Returns the closest enrolled person (or a not-matched dict)
    based on cosine similarity in the FAISS index.
    """
    if index.ntotal == 0:
        return {
            "match": "Not Matched",
            "reason": "No enrolled embeddings found",
        }

    q_emb = image_to_embedding(query_image_path)
    scores, ids = index.search(np.array([q_emb]), k=1)
    similarity  = float(scores[0][0])
    match_idx   = int(ids[0][0])

    if similarity < SIMILARITY_THRESHOLD:
        return {
            "match":      "Not Matched",
            "similarity": similarity,
            "reason":     f"Best similarity {similarity:.3f} is below threshold {SIMILARITY_THRESHOLD}",
        }

    return {
        "match":         "Matched",
        "person_id":     metadata[match_idx]["person_id"],
        "similarity":    similarity,
        "matched_image": metadata[match_idx]["image_path"],
    }

# Keep the old function name working so enroll.py / legacy callers still work
def identify_person(query_image_path: str) -> dict:
    return retrieve(query_image_path)


# ─────────────────────────────────────────────
# STEP 2 – AUGMENTATION
# ─────────────────────────────────────────────
def augment(retrieval_result: dict) -> dict:
    """
    AUGMENTATION STAGE
    Enriches the retrieval result with structured knowledge from
    PERSON_PROFILES so the generation step has rich context.
    Returns a new dict that merges retrieval output + profile data.
    """
    result = dict(retrieval_result)   # don't mutate original

    if result["match"] == "Matched":
        person_id = result["person_id"]
        profile   = PERSON_PROFILES.get(person_id, {})
        result["profile"] = profile   # attach full profile
    else:
        result["profile"] = {}

    return result


# ─────────────────────────────────────────────
# STEP 3 – GENERATION
# ─────────────────────────────────────────────
def build_prompt(augmented_result: dict) -> str:
    """Build the LLM prompt from the augmented retrieval context."""
    if augmented_result["match"] == "Matched":
        p   = augmented_result["profile"]
        sim = augmented_result["similarity"]

        context_block = f"""
Face Recognition Result
-----------------------
Status          : MATCHED
Person ID       : {augmented_result['person_id']}
Full Name       : {p.get('full_name', 'Unknown')}
Profession      : {p.get('profession', 'Unknown')}
Known For       : {p.get('known_for', 'N/A')}
Nationality     : {p.get('nationality', 'Unknown')}
Access Level    : {p.get('access_level', 'Standard')}
Department      : {p.get('department', 'General')}
Similarity Score: {sim:.4f}  (threshold: {SIMILARITY_THRESHOLD})
"""
        prompt = f"""You are the AI component of a face-recognition access-control system.
Based on the structured recognition data below, write a concise, professional
identification report in 6-7 sentences that:
  1. Confirms who has been identified and at what confidence level.
  2. Provides a brief, relevant description of the person using the context provided, internet data, for actors mention films other than the ones already there in PERSON_PROFILES and also give good achievements for a person with profession other than actor
  3. States the access decision clearly (granted / denied) but don't mention VIP access.

{context_block}

Write only the report. Do not include headings or bullet points."""

    else:
        sim_str = f"{augmented_result.get('similarity', 0):.4f}" if augmented_result.get("similarity") else "N/A"
        prompt = f"""You are the AI component of a face-recognition access-control system.
The system could not match the uploaded face to any enrolled person.

Recognition Result
------------------
Status          : NOT MATCHED
Best Similarity : {sim_str}  (threshold: {SIMILARITY_THRESHOLD})
Reason          : {augmented_result.get('reason', 'Unknown person')}

Write a concise 2-sentence security response that:
  1. States the person was not recognised.
  2. Denies access and recommends escalation to human security.

Write only the response. Do not include headings."""

    return prompt


def generate(augmented_result: dict) -> str:
    """
    GENERATION STAGE
    Calls the Google Gemini Flash 2.5 API with the augmented context as a prompt
    and returns a natural-language identification report.
    Falls back to a template response when GEMINI_API_KEY is not set.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        # Graceful fallback when no API key is configured
        if augmented_result["match"] == "Matched":
            p    = augmented_result.get("profile", {})
            name = p.get("full_name", augmented_result["person_id"])
            sim  = augmented_result.get("similarity", 0)
            lvl  = p.get("access_level", "Standard")
            return (f"Identity confirmed: {name} — recognised with {sim:.1%} confidence. ")
                
        else:
            return ("The uploaded face does not match any enrolled person in the database. "
                    "Access denied. Please contact security for manual verification.")

    client = genai.Client(api_key=api_key)
    prompt = build_prompt(augmented_result)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "temperature": 0.5,
            "max_output_tokens": 1024
        }
    )

    return response.text.strip()


# ─────────────────────────────────────────────
# FULL RAG PIPELINE  (public entry point)
# ─────────────────────────────────────────────
def full_rag_pipeline(query_image_path: str) -> dict:
    """
    Runs the complete RAG pipeline:
        Retrieval -> Augmentation -> Generation

    Returns a dict with keys:
        match, similarity?, person_id?, matched_image?,
        profile?, generated_response, description
    """
    # Stage 1 - Retrieval: CLIP embedding + FAISS nearest-neighbour search
    retrieval_result = retrieve(query_image_path)

    # Stage 2 - Augmentation: enrich with structured profile from knowledge base
    augmented_result = augment(retrieval_result)

    # Stage 3 - Generation: LLM produces a natural-language identification report
    generated_response = generate(augmented_result)
    augmented_result["generated_response"] = generated_response

    # Keep legacy 'description' field so old result.html still works
    augmented_result["description"] = generated_response

    return augmented_result