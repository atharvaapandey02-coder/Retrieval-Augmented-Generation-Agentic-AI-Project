from flask import Flask, render_template, request
import os
from rag_engine import full_rag_pipeline, load_vector_store

# Ensure vector store is loaded on startup
index, metadata = load_vector_store()

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/identify", methods=["POST"])
def identify():
    if "image" not in request.files:
        return "No file uploaded", 400

    file = request.files["image"]
    if file.filename == "":
        return "No file selected", 400

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    # Full RAG pipeline: Retrieval -> Augmentation -> Generation
    result = full_rag_pipeline(save_path)

    return render_template("result.html",
                           result=result,
                           uploaded_image=save_path)

if __name__ == "__main__":
    app.run(debug=True)
