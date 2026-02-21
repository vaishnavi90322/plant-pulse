# 🌿 Plant Pulse — AI Plant Disease Detection

**Plant Pulse** is a Flask web application that detects plant diseases from leaf images using the [Plant.id v3 API](https://plant.id/). Upload or capture a photo of a plant leaf and get instant diagnosis with treatment advice.

---

## ✨ Features

- 🔍 **AI Disease Detection** — powered by Plant.id API v3
- 📋 **Detailed Analysis** — disease description, category, and confidence score
- 💊 **Structured Treatment** — Biological, Chemical, and Prevention sections
- 📄 **PDF Download** — export your scan report with image included
- 🕒 **Scan History** — view all past scans with treatment notes
- 📷 **Camera Capture** — scan directly from your device camera
- 🔐 **Authentication** — Email/password + Google OAuth login
- 🏥 **Healthy Detection** — identifies healthy plants too

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- A [Plant.id API key](https://www.kindwise.com/plant-id) (free tier available)
- (Optional) Google OAuth credentials for Google login

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/vaishnavi90322/plant-pulse.git
cd plant-pulse

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and fill in your keys (see below)

# 5. Run the app
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## ⚙️ Environment Variables

Create a `.env` file in the project root:

```env
SECRET_KEY=your-random-secret-key-here
PLANT_ID_API_KEY=your-plant-id-api-key

# Optional — only needed for Google OAuth login
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
```

> ⚠️ **Never commit your `.env` file.** It is already excluded via `.gitignore`.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python, Flask, SQLAlchemy |
| AI / Disease API | [Plant.id v3](https://plant.id/) |
| Authentication | Flask-Login, Authlib (Google OAuth) |
| PDF Generation | ReportLab |
| Frontend | HTML5, CSS3, Vanilla JS |
| Database | SQLite |

---

## 📁 Project Structure

```
plant-pulse/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── .env                    # Secrets (not committed)
├── .gitignore
├── static/
│   └── style.css           # App styles
├── templates/
│   ├── home.html
│   ├── upload.html
│   ├── result.html         # Scan results + PDF download
│   ├── history.html        # Scan history
│   ├── login.html
│   ├── register.html
│   └── ...
└── uploads/                # Uploaded images (not committed)
```

---

## 📸 How It Works

1. **Register / Log in** (email or Google)
2. **Upload or capture** a plant leaf photo
3. **Plant.id API** analyses the image for diseases
4. **Results page** shows:
   - Plant name & disease name
   - Confidence score
   - Description of the disease
   - Treatment & Prevention (Biological / Chemical / Prevention)
   - Similar disease images
5. **Download PDF** — full report with image embedded
6. **History** — all past scans saved to your account

---

## 📦 Dependencies

```
Flask
Flask-Login
Flask-SQLAlchemy
Authlib
requests
python-dotenv
Pillow
numpy
reportlab
```

Install with: `pip install -r requirements.txt`

---

## 📜 License

MIT License — free to use and modify.

---

## 🙋 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

> Built with ❤️ using Flask + Plant.id AI
