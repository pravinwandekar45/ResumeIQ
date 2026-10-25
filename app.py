
import os
import json
from datetime import datetime, timezone
from io import BytesIO

import fitz
from docx import Document
from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename

app = Flask(__name__, instance_relative_config=True)

# Use an environment variable in production.
app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY", "dev-only-change-this-key"
)
app.config["SQLALCHEMY_DATABASE_URI"] = (
    "sqlite:///" + os.path.join(app.instance_path, "resumeiq.db")
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB

os.makedirs(app.instance_path, exist_ok=True)

db = SQLAlchemy(app)

ALLOWED_EXTENSIONS = {"pdf", "docx"}

SKILLS = [
    "python", "java", "flask", "django", "fastapi",
    "sql", "mysql", "sqlite", "mongodb",
    "html", "css", "javascript", "react",
    "rest api", "git", "github", "docker",
    "machine learning", "pandas", "numpy",
    "scikit-learn", "data analysis", "communication",
    "problem solving", "oops", "data structures",
    "algorithms", "pytest", "sqlalchemy"
]


class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    resume_filename = db.Column(db.String(255), nullable=False)
    job_title = db.Column(db.String(150), nullable=False)
    match_score = db.Column(db.Float, nullable=False)
    matched_skills = db.Column(db.Text, nullable=False)
    missing_skills = db.Column(db.Text, nullable=False)
    created_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc),
        nullable=False
    )


def extract_text(file_storage):
    """Extract text from an uploaded PDF or DOCX file."""
    data = file_storage.read()
    filename = file_storage.filename.lower()

    if filename.endswith(".pdf"):
        with fitz.open(stream=data, filetype="pdf") as pdf:
            return "\n".join(page.get_text() for page in pdf)

    if filename.endswith(".docx"):
        document = Document(BytesIO(data))
        return "\n".join(p.text for p in document.paragraphs)

    raise ValueError("Only PDF and DOCX files are supported.")


def analyze_resume(resume_text, job_description):
    """Calculate text similarity and identify skills."""
    resume_text = resume_text.strip()
    job_description = job_description.strip()

    if not resume_text or not job_description:
        raise ValueError(
            "Could not extract text. Try a text-based PDF or DOCX file."
        )

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        vectors = vectorizer.fit_transform(
            [resume_text, job_description]
        )
        score = float(
            cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        ) * 100
    except ValueError:
        score = 0.0

    resume_lower = resume_text.lower()
    job_lower = job_description.lower()

    required_skills = [
        skill for skill in SKILLS if skill in job_lower
    ]

    matched = [
        skill for skill in required_skills if skill in resume_lower
    ]
    missing = [
        skill for skill in required_skills if skill not in resume_lower
    ]

    return round(score, 2), matched, missing


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        uploaded_file = request.files.get("resume")
        job_title = request.form.get("job_title", "").strip()
        job_description = request.form.get(
            "job_description", ""
        ).strip()

        if not uploaded_file or not uploaded_file.filename:
            error = "Please select a resume file."
        elif not job_title or len(job_title) > 150:
            error = "Enter a job title (up to 150 characters)."
        elif not job_description:
            error = "Please enter the job description."
        elif len(job_description) > 30000:
            error = "Job description is too long (maximum 30,000 characters)."
        elif (
            "." not in uploaded_file.filename
            or uploaded_file.filename.rsplit(".", 1)[1].lower()
            not in ALLOWED_EXTENSIONS
        ):
            error = "Upload a PDF or DOCX file."
        else:
            try:
                resume_text = extract_text(uploaded_file)

                if len(resume_text) > 200000:
                    raise ValueError("Extracted resume text is too long.")

                score, matched, missing = analyze_resume(
                    resume_text, job_description
                )

                safe_filename = secure_filename(
                    uploaded_file.filename
                ) or "resume"

                analysis = Analysis(
                    resume_filename=safe_filename,
                    job_title=job_title,
                    match_score=score,
                    matched_skills=json.dumps(matched),
                    missing_skills=json.dumps(missing)
                )
                db.session.add(analysis)
                db.session.commit()

                result = {
                    "id": analysis.id,
                    "score": score,
                    "matched": matched,
                    "missing": missing
                }

            except (ValueError, RuntimeError) as exc:
                db.session.rollback()
                error = str(exc)
            except Exception:
                db.session.rollback()
                app.logger.exception("Resume analysis failed")
                error = "Analysis failed. Please check the file and try again."

    return render_template("index.html", result=result, error=error)



@app.route("/history")
def history():
    analyses = Analysis.query.order_by(
        Analysis.created_at.desc()
    ).limit(100).all()

    highest_score = max(
        (item.match_score for item in analyses),
        default=0
    )

    average_score = (
        sum(item.match_score for item in analyses) / len(analyses)
        if analyses else 0
    )

    return render_template(
        "history.html",
        analyses=analyses,
        highest_score=highest_score,
        average_score=average_score
    )



with app.app_context():
    db.create_all()


if __name__ == "__main__":
    app.run(debug=True)

