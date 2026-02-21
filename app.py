import os
import hashlib
import datetime
import base64
import requests as http_requests
from functools import wraps

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from flask import (Flask, render_template, request, redirect,
                   url_for, flash, session)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, UserMixin, login_user,
                         logout_user, login_required, current_user)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from authlib.integrations.flask_client import OAuth
from PIL import Image

PLANT_ID_API_KEY = os.getenv("PLANT_ID_API_KEY", "")

# ── App & config ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "plant-pulse-dev-secret-change-in-production")
app.config["UPLOAD_FOLDER"]               = os.path.join(BASE_DIR, "uploads")
app.config["MAX_CONTENT_LENGTH"]          = 16 * 1024 * 1024
app.config["SQLALCHEMY_DATABASE_URI"]     = os.getenv(
    "DATABASE_URL",
    "sqlite:///" + os.path.join(BASE_DIR, "plant_pulse.db")
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "gif"}

# Ensure uploads folder always exists (including on Render)
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

db           = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view     = "login"
login_manager.login_message  = "Please log in to access this page."

# ── Google OAuth ───────────────────────────────────────────────────────────────
oauth = OAuth(app)
google = oauth.register(
    name="google",
    client_id     = os.getenv("GOOGLE_CLIENT_ID", ""),
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET", ""),
    server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs = {"scope": "openid email profile"},
)

# ── Models ─────────────────────────────────────────────────────────────────────
class User(UserMixin, db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    email         = db.Column(db.String(200), unique=True, nullable=False, index=True)
    name          = db.Column(db.String(120))
    password_hash = db.Column(db.String(256))          # None for Google-only accounts
    google_id     = db.Column(db.String(120), unique=True)
    avatar_url    = db.Column(db.String(500))
    created_at    = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    scans         = db.relationship("ScanHistory", backref="user", lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return self.password_hash and check_password_hash(self.password_hash, password)


class ScanHistory(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True, index=True)
    image_hash = db.Column(db.String(32), index=True)
    filename    = db.Column(db.String(200))
    plant       = db.Column(db.String(100))
    disease     = db.Column(db.String(200))
    raw_class   = db.Column(db.String(500))   # stores description
    treatment   = db.Column(db.Text)
    confidence  = db.Column(db.Float)
    category    = db.Column(db.String(100))
    is_healthy  = db.Column(db.Boolean, default=False)
    scanned_at  = db.Column(db.DateTime, default=datetime.datetime.utcnow)


with app.app_context():
    db.create_all()


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ── Plant.id API ──────────────────────────────────────────────────────────────
def _guess_plant_name(disease_name):
    """Extract host plant from a disease name like 'Tomato Bacterial Spot' → 'Tomato'."""
    if not disease_name or disease_name in ("Healthy", "Unknown Disease"):
        return "Plant"
    # Plant.id disease names almost always start with the plant host
    parts = disease_name.split()
    return parts[0] if len(parts) >= 2 else disease_name


def call_plant_id(img_path):
    """
    Calls Plant.id v3 Health Assessment API.
    Returns a dict with keys: plant, disease, confidence, is_healthy,
    category, description, treatment_dict, similar_images.
    'treatment_dict' has optional keys: biological, chemical, prevention (each a list).
    """
    import json as _json
    from io import BytesIO

    # Resize to max 800px to keep payload small & avoid 400 errors
    pil_img = Image.open(img_path).convert("RGB")
    pil_img.thumbnail((800, 800))
    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    payload = {
        "images": [f"data:image/jpeg;base64,{img_b64}"],
        "health": "all",
        "similar_images": True,
    }
    headers = {
        "Api-Key": PLANT_ID_API_KEY,
        "Content-Type": "application/json",
    }
    params = {"details": "description,treatment,classification,common_names"}

    resp = http_requests.post(
        "https://plant.id/api/v3/health_assessment",
        json=payload, headers=headers, params=params, timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    # ── Parse response ──────────────────────────────────────────────────────
    result     = data.get("result", {})
    is_healthy = result.get("is_healthy", {}).get("binary", True)
    confidence = round(result.get("is_healthy", {}).get("probability", 0) * 100, 1)

    # Plant name — from API classification, or guessed from the disease name
    plant_suggestions = result.get("classification", {}).get("suggestions", [])
    plant_name = plant_suggestions[0]["name"] if plant_suggestions else None

    # Disease info
    disease_suggestions = result.get("disease", {}).get("suggestions", [])
    if disease_suggestions and not is_healthy:
        top_disease  = disease_suggestions[0]
        disease_name = top_disease.get("name", "Unknown Disease")
        confidence   = round(top_disease.get("probability", 0) * 100, 1)
        details      = top_disease.get("details", {})

        description = details.get("description") or "No description available."

        # Keep treatment as a structured dict  {biological:[...], chemical:[...], prevention:[...]}
        raw_treat = details.get("treatment") or {}
        treatment_dict = {
            "biological": [str(t) for t in raw_treat.get("biological", [])[:3]],
            "chemical":   [str(t) for t in raw_treat.get("chemical", [])[:3]],
            "prevention": [str(t) for t in raw_treat.get("prevention", [])[:3]],
        }

        classification = details.get("classification") or []
        category = classification[0] if classification else "Plant Disease"

        similar_imgs = [
            s["url"] for s in top_disease.get("similar_images", [])[:3] if s.get("url")
        ]
    else:
        disease_name   = "Healthy"
        description    = "No disease detected. Your plant looks healthy!"
        treatment_dict = {}
        category       = "Healthy"
        similar_imgs   = []

    if not plant_name:
        plant_name = _guess_plant_name(disease_name)

    return {
        "plant":          plant_name,
        "disease":        disease_name,
        "confidence":     confidence,
        "is_healthy":     is_healthy,
        "category":       category,
        "description":    description,
        "treatment_dict": treatment_dict,            # structured dict
        "similar_images": similar_imgs,
    }




def PLACEHOLDER_CLASS_NAMES_STUB(): pass  # kept so line references stay stable


# ── Helpers ────────────────────────────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXTENSIONS

def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

# ── Dummy/fallback label set (REMOVED — using Plant.id API now) ────────────────
if False:  # noqa — kept as dead code so other imports don't break
    CLASS_NAMES = [
    "Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust",
    "Apple___healthy","Blueberry___healthy","Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy","Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_","Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy","Grape___Black_rot","Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)","Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)","Peach___Bacterial_spot",
    "Peach___healthy","Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy",
    "Potato___Early_blight","Potato___Late_blight","Potato___healthy",
    "Raspberry___healthy","Soybean___healthy","Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch","Strawberry___healthy","Tomato___Bacterial_spot",
    "Tomato___Early_blight","Tomato___Late_blight","Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot","Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot","Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus","Tomato___healthy",
]

DISEASE_INFO = {
    "Apple___Apple_scab":           {"category":"Fungal disease",   "spread":"Wind-dispersed spores from infected leaves and fruit.","part":"Leaves and fruit"},
    "Apple___Black_rot":            {"category":"Fungal disease",   "spread":"Rain splash and infected debris.","part":"Fruit, leaves, bark"},
    "Apple___Cedar_apple_rust":     {"category":"Fungal disease",   "spread":"Wind-carried spores from nearby cedar trees.","part":"Leaves and fruit"},
    "Apple___healthy":              {"category":"Healthy",          "spread":"N/A","part":"N/A"},
    "Blueberry___healthy":          {"category":"Healthy",          "spread":"N/A","part":"N/A"},
    "Cherry_(including_sour)___Powdery_mildew":{"category":"Fungal disease","spread":"Airborne spores in warm, dry weather.","part":"Leaves and shoots"},
    "Cherry_(including_sour)___healthy":{"category":"Healthy","spread":"N/A","part":"N/A"},
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":{"category":"Fungal disease","spread":"Wind and rain splash.","part":"Leaves"},
    "Corn_(maize)___Common_rust_":  {"category":"Fungal disease",   "spread":"Wind-blown urediniospores.","part":"Leaves"},
    "Corn_(maize)___Northern_Leaf_Blight":{"category":"Fungal disease","spread":"Wind-dispersed conidia.","part":"Leaves"},
    "Corn_(maize)___healthy":       {"category":"Healthy",          "spread":"N/A","part":"N/A"},
    "Grape___Black_rot":            {"category":"Fungal disease",   "spread":"Rain splash from infected mummies.","part":"Fruit, leaves, shoots"},
    "Grape___Esca_(Black_Measles)": {"category":"Fungal disease",   "spread":"Through pruning wounds.","part":"Wood, leaves, fruit"},
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":{"category":"Fungal disease","spread":"Wind and rain.","part":"Leaves"},
    "Grape___healthy":              {"category":"Healthy",          "spread":"N/A","part":"N/A"},
    "Orange___Haunglongbing_(Citrus_greening)":{"category":"Bacterial disease","spread":"Asian citrus psyllid insect.","part":"Leaves, fruit, root"},
    "Peach___Bacterial_spot":       {"category":"Bacterial disease","spread":"Rain and wind from infected tissue.","part":"Leaves and fruit"},
    "Peach___healthy":              {"category":"Healthy",          "spread":"N/A","part":"N/A"},
    "Pepper,_bell___Bacterial_spot":{"category":"Bacterial disease","spread":"Rain splash and contaminated tools.","part":"Leaves and fruit"},
    "Pepper,_bell___healthy":       {"category":"Healthy",          "spread":"N/A","part":"N/A"},
    "Potato___Early_blight":        {"category":"Fungal disease",   "spread":"Wind and rain splash.","part":"Leaves and tubers"},
    "Potato___Late_blight":         {"category":"Oomycete disease", "spread":"Wind-dispersed sporangia in cool wet weather.","part":"Leaves, stems, tubers"},
    "Potato___healthy":             {"category":"Healthy",          "spread":"N/A","part":"N/A"},
    "Raspberry___healthy":          {"category":"Healthy",          "spread":"N/A","part":"N/A"},
    "Soybean___healthy":            {"category":"Healthy",          "spread":"N/A","part":"N/A"},
    "Squash___Powdery_mildew":      {"category":"Fungal disease",   "spread":"Airborne spores.","part":"Leaves"},
    "Strawberry___Leaf_scorch":     {"category":"Fungal disease",   "spread":"Rain splash.","part":"Leaves"},
    "Strawberry___healthy":         {"category":"Healthy",          "spread":"N/A","part":"N/A"},
    "Tomato___Bacterial_spot":      {"category":"Bacterial disease","spread":"Rain splash and infected seeds.","part":"Leaves and fruit"},
    "Tomato___Early_blight":        {"category":"Fungal disease",   "spread":"Wind and rain.","part":"Leaves, stems, fruit"},
    "Tomato___Late_blight":         {"category":"Oomycete disease", "spread":"Wind-dispersed sporangia.","part":"Leaves, stems, fruit"},
    "Tomato___Leaf_Mold":           {"category":"Fungal disease",   "spread":"Airborne conidia in humid conditions.","part":"Leaves"},
    "Tomato___Septoria_leaf_spot":  {"category":"Fungal disease",   "spread":"Windblown water and rain splash.","part":"Leaf, stem, fruit"},
    "Tomato___Spider_mites Two-spotted_spider_mite":{"category":"Pest damage","spread":"Wind and physical contact.","part":"Leaves"},
    "Tomato___Target_Spot":         {"category":"Fungal disease",   "spread":"Wind and rain.","part":"Leaves and fruit"},
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":{"category":"Viral disease","spread":"Whitefly (Bemisia tabaci).","part":"Whole plant"},
    "Tomato___Tomato_mosaic_virus": {"category":"Viral disease",    "spread":"Mechanical contact and infected seed.","part":"Whole plant"},
    "Tomato___healthy":             {"category":"Healthy",          "spread":"N/A","part":"N/A"},
}

def is_plant_image(img_path, threshold=0.08):
    """
    Returns True if the image likely contains a plant / leaf.
    Uses HSV color analysis:
      - Hue 60-165° covers yellow-green → green → cyan-green (leaf spectrum)
      - Saturation > 25% (rules out grey/white backgrounds)
      - Value 15-95% (rules out near-black and blown-out whites)
    If at least `threshold` (8%) of pixels match, we accept it as a plant image.
    """
    try:
        img   = Image.open(img_path).convert("RGB").resize((224, 224))
        arr   = np.array(img, dtype=np.float32) / 255.0      # H×W×3  [0,1]
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

        # Convert RGB → HSV (vectorised, no extra library needed)
        cmax  = np.maximum(np.maximum(r, g), b)
        cmin  = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin

        # Hue (0-360)
        hue = np.zeros_like(cmax)
        mask_r = (cmax == r) & (delta > 0)
        mask_g = (cmax == g) & (delta > 0)
        mask_b = (cmax == b) & (delta > 0)
        hue[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / delta[mask_r])) % 360
        hue[mask_g] = 60 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120
        hue[mask_b] = 60 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240

        sat = np.where(cmax > 0, delta / cmax, 0)   # saturation [0,1]
        val = cmax                                    # value [0,1]

        # Leaf-green pixels: hue in [55°,165°], sat>0.20, val in [0.10,0.95]
        green_mask = (
            (hue >= 55) & (hue <= 165) &
            (sat > 0.20) &
            (val > 0.10) & (val < 0.95)
        )
        green_ratio = green_mask.sum() / green_mask.size
        return green_ratio >= threshold
    except Exception:
        return True   # if analysis fails, don't block the user


# ── Auth routes ────────────────────────────────────────────────────────────────
@app.route("/register", methods=["GET","POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("home"))
    if request.method == "POST":
        name     = request.form.get("name","").strip()
        email    = request.form.get("email","").strip().lower()
        password = request.form.get("password","")
        confirm  = request.form.get("confirm","")
        if not name or not email or not password:
            flash("All fields are required.", "error")
        elif password != confirm:
            flash("Passwords do not match.", "error")
        elif len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
        elif User.query.filter_by(email=email).first():
            flash("An account with this email already exists.", "error")
        else:
            user = User(name=name, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            login_user(user)
            flash(f"Welcome, {user.name}! Your account has been created.", "success")
            return redirect(url_for("home"))
    return render_template("register.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("home"))
    if request.method == "POST":
        email    = request.form.get("email","").strip().lower()
        password = request.form.get("password","")
        user     = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user, remember=request.form.get("remember"))
            flash(f"Welcome back, {user.name}!", "success")
            next_page = request.args.get("next")
            return redirect(next_page or url_for("home"))
        flash("Invalid email or password.", "error")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))

# ── Google OAuth ───────────────────────────────────────────────────────────────
@app.route("/auth/google")
def google_login():
    if not os.getenv("GOOGLE_CLIENT_ID"):
        flash("Google login is not configured yet. Please use email/password.", "error")
        return redirect(url_for("login"))
    redirect_uri = url_for("google_callback", _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route("/auth/google/callback")
def google_callback():
    try:
        token    = google.authorize_access_token()
        userinfo = token.get("userinfo") or {}
        email    = userinfo.get("email","").lower()
        if not email:
            flash("Google login failed: no email returned.", "error")
            return redirect(url_for("login"))

        user = User.query.filter_by(email=email).first()
        if not user:
            user = User(
                email      = email,
                name       = userinfo.get("name", email.split("@")[0]),
                google_id  = userinfo.get("sub"),
                avatar_url = userinfo.get("picture"),
            )
            db.session.add(user)
            db.session.commit()
        else:
            # Update Google info if they previously registered with email
            if not user.google_id:
                user.google_id  = userinfo.get("sub")
                user.avatar_url = userinfo.get("picture")
                db.session.commit()

        login_user(user)
        flash(f"Welcome, {user.name}!", "success")
        return redirect(url_for("home"))
    except Exception as e:
        flash(f"Google login error: {str(e)}", "error")
        return redirect(url_for("login"))

# ── App routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/upload", methods=["GET","POST"])
@login_required
def upload():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        f = request.files["file"]
        if f.filename == "" or not allowed_file(f.filename):
            return redirect(request.url)
        filename  = secure_filename(f.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        f.save(save_path)
        return redirect(url_for("result", filename=filename))
    return render_template("upload.html")

@app.route("/result/<filename>")
@login_required
def result(filename):
    img_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(img_path):
        return redirect(url_for("upload"))

    # ── Plant / leaf validation ────────────────────────────────────────────
    if not is_plant_image(img_path):
        return render_template("invalid_image.html", filename=filename)

    img_hash = file_md5(img_path)

    # Check if this user already scanned this image
    cached = ScanHistory.query.filter_by(
        image_hash=img_hash, user_id=current_user.id
    ).first()

    import json as _json
    if cached:
        plant          = cached.plant
        disease        = cached.disease
        confidence     = cached.confidence
        is_healthy     = cached.is_healthy
        category       = cached.category
        description    = cached.raw_class or ""
        try:
            treatment_dict = _json.loads(cached.treatment or "{}")
        except Exception:
            treatment_dict = {}
        similar_images = []
    else:
        if not PLANT_ID_API_KEY:
            flash("Plant.id API key not configured. Add PLANT_ID_API_KEY to your .env file.", "error")
            return redirect(url_for("upload"))
        try:
            res            = call_plant_id(img_path)
            plant          = res["plant"]
            disease        = res["disease"]
            confidence     = res["confidence"]
            is_healthy     = res["is_healthy"]
            category       = res["category"]
            description    = res["description"]
            treatment_dict = res["treatment_dict"]
            similar_images = res["similar_images"]
        except Exception as e:
            flash(f"Plant.id API error: {str(e)}", "error")
            return redirect(url_for("upload"))

        db.session.add(ScanHistory(
            user_id    = current_user.id,
            image_hash = img_hash,
            filename   = filename,
            plant      = plant,
            disease    = disease,
            raw_class  = description,
            treatment  = _json.dumps(treatment_dict),
            confidence = confidence,
            category   = category,
            is_healthy = is_healthy,
        ))
        db.session.commit()

    return render_template("result.html",
        filename=filename, plant=plant, disease=disease,
        confidence=confidence, is_healthy=is_healthy,
        category=category, description=description,
        treatment=treatment_dict, similar_images=similar_images)


@app.route("/history")
@login_required
def history():
    scans = ScanHistory.query.filter_by(user_id=current_user.id)\
                             .order_by(ScanHistory.scanned_at.desc()).all()
    return render_template("history.html", scans=scans)

@app.route("/history/clear", methods=["POST"])
@login_required
def history_clear():
    ScanHistory.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    return redirect(url_for("history"))


# ── PDF Download ───────────────────────────────────────────────────────────────
@app.route("/result/<filename>/pdf")
@login_required
def download_pdf(filename):
    """Generate and stream a PDF report for a scan result."""
    import json as _json, datetime as _dt
    from io import BytesIO
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, Image as RLImage,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER

    cached = ScanHistory.query.filter_by(
        filename=filename, user_id=current_user.id
    ).first()
    if not cached:
        return redirect(url_for("upload"))

    try:
        treatment_dict = _json.loads(cached.treatment or "{}")
    except Exception:
        treatment_dict = {}
    description = cached.raw_class or "No description available."

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()
    GREEN  = colors.HexColor("#4A6D4B")
    LGREY  = colors.HexColor("#f4f7f4")
    DGREY  = colors.HexColor("#555")
    RED    = colors.HexColor("#b91c1c")

    h1  = ParagraphStyle("h1",  fontSize=20, leading=24, textColor=GREEN,  spaceAfter=4,  fontName="Helvetica-Bold")
    h2  = ParagraphStyle("h2",  fontSize=13, leading=16, textColor=GREEN,  spaceAfter=4,  fontName="Helvetica-Bold")
    sub = ParagraphStyle("sub", fontSize=9,  leading=12, textColor=DGREY,  spaceAfter=2,  fontName="Helvetica")
    bod = ParagraphStyle("bod", fontSize=10, leading=14, textColor=colors.HexColor("#333"), spaceAfter=3, fontName="Helvetica")
    bul = ParagraphStyle("bul", fontSize=10, leading=14, textColor=colors.HexColor("#333"), fontName="Helvetica", leftIndent=12)

    story = []

    # ── Header ────────────────────────────────────────────────────────────
    story.append(Paragraph("🌿 Plant Pulse", h1))
    story.append(Paragraph("AI Plant Disease Analysis Report", sub))
    story.append(Paragraph(
        f"Generated: {_dt.datetime.utcnow().strftime('%d %B %Y, %H:%M')} UTC  |  User: {current_user.name}",
        sub
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=GREEN, spaceAfter=12))

    # ── Scanned image ─────────────────────────────────────────────────────
    img_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if os.path.exists(img_path):
        try:
            from PIL import Image as PILImage
            pil = PILImage.open(img_path).convert("RGB")
            pil.thumbnail((400, 400))
            ibuf = BytesIO()
            pil.save(ibuf, format="JPEG", quality=85)
            ibuf.seek(0)
            rl_img = RLImage(ibuf, width=7*cm, height=7*cm)
            rl_img.hAlign = "CENTER"
            story.append(rl_img)
            story.append(Spacer(1, 0.4*cm))
        except Exception:
            pass

    # ── Summary table ─────────────────────────────────────────────────────
    status_lbl = "✅ Healthy" if cached.is_healthy else "⚠️ Disease Detected"
    status_clr = GREEN if cached.is_healthy else RED
    table_data = [
        [Paragraph("Field", ParagraphStyle("th", fontSize=10, fontName="Helvetica-Bold", textColor=colors.white)),
         Paragraph("Value", ParagraphStyle("th", fontSize=10, fontName="Helvetica-Bold", textColor=colors.white))],
        ["Plant",          cached.plant or "Unknown"],
        ["Disease / Status", cached.disease or "N/A"],
        ["Category",       cached.category or "N/A"],
        ["Confidence",     f"{cached.confidence}%"],
        ["Status",         status_lbl],
        ["Scan Date",      cached.scanned_at.strftime("%d %b %Y, %H:%M")],
    ]
    tbl = Table(table_data, colWidths=[4.5*cm, 12*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,0), GREEN),
        ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
        ("FONTNAME",     (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",     (0,0), (-1,-1), 10),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [LGREY, colors.white]),
        ("GRID",         (0,0), (-1,-1), 0.5, colors.HexColor("#ccc")),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING",  (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING",   (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.5*cm))

    # ── Description ───────────────────────────────────────────────────────
    if description and description != "N/A":
        story.append(Paragraph("Description", h2))
        story.append(Paragraph(description, bod))
        story.append(Spacer(1, 0.4*cm))

    # ── Treatment ─────────────────────────────────────────────────────────
    if not cached.is_healthy and treatment_dict:
        story.append(Paragraph("Treatment & Prevention", h2))
        sections = [
            ("biological", "🌿 Biological Treatment"),
            ("chemical",   "🧪 Chemical Treatment"),
            ("prevention", "🛡️ Prevention"),
        ]
        for key, label in sections:
            items = treatment_dict.get(key, [])
            if items:
                story.append(Paragraph(f"<b>{label}</b>", bod))
                for item in items:
                    story.append(Paragraph(f"• {item}", bul))
                story.append(Spacer(1, 0.25*cm))

    # ── Footer ────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=DGREY, spaceBefore=12))
    story.append(Paragraph(
        "This report is generated by Plant Pulse AI. Always consult a local agronomist for professional advice.",
        ParagraphStyle("foot", fontSize=8, textColor=DGREY, fontName="Helvetica-Oblique", alignment=TA_CENTER)
    ))

    doc.build(story)
    buf.seek(0)
    safe_name = f"PlantPulse_{cached.disease or 'report'}_{_dt.datetime.utcnow().strftime('%Y%m%d')}.pdf".replace(" ", "_")
    from flask import send_file
    return send_file(
        buf,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=safe_name,
    )

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    app.run(debug=debug)
