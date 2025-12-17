# frontend/streamlit_app.py
# Complete merged Streamlit app for Nutri-Chef (Home, Guest, Login, BMI, Recipe flows, Profile)
import streamlit as st
import sqlite3
import hashlib
import os
import pandas as pd
import json
import base64
import re
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
import sys, os

# Go from: frontend → Nutrichef (root)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
from models.prediction_ml import estimate_days_to_goal, estimate_weight_loss_from_meals


# If you use Gemini / GenAI, keep this; otherwise comment out
from google import genai
client = genai.Client(api_key="AIzaSyCNqnklazaTD6kYtWaLbax6RCCRp24T_tE")  # replace with actual key or comment out if not used

# ---------------------- PATHS & DIRS ----------------------
BASE_DIR = os.path.dirname(__file__)  # frontend
REPO_ROOT = os.path.join(BASE_DIR, "..")
NOSQL_DIR = os.path.join(REPO_ROOT, "nosql")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
DB_PATH = os.path.join(r"C:\Users\BHAVITHAK\Documents\Nutrichef", "Receipes.db")  # keep your path

os.makedirs(NOSQL_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "assets"), exist_ok=True)

# ---------------------- JSON (NoSQL-replacement helpers) ----------------------
def json_path(name: str) -> str:
    return os.path.join(NOSQL_DIR, f"{name}.json")

def json_read(name: str):
    p = json_path(name)
    if not os.path.exists(p):
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def json_write(name: str, data):
    p = json_path(name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str)

# ---------------------- SQLITE HELPERS ----------------------
def get_db_connection():
    db_parent = os.path.join(BASE_DIR, "..", "db")
    os.makedirs(db_parent, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS users (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               email TEXT UNIQUE NOT NULL,
               password_hash TEXT NOT NULL
           )"""
    )
    conn.commit()
    return conn

def hash_password(pwd: str) -> str:
    return hashlib.sha256(pwd.encode("utf-8")).hexdigest()

def register_user(email: str, password: str):
    try:
        conn = get_db_connection()
        conn.execute(
            "INSERT INTO users (email, password_hash) VALUES (?, ?)",
            (email, hash_password(password))
        )
        conn.commit()
        return True, "Registration successful! Please login now."
    except sqlite3.IntegrityError:
        return False, "Email already registered!"
    except Exception as e:
        return False, str(e)

def login_user(email: str, password: str):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT password_hash FROM users WHERE email = ?", (email,))
    row = cur.fetchone()
    if row and row[0] == hash_password(password):
        return True, "Login successful!"
    return False, "Invalid email or password."

def update_user_email(old_email: str, new_email: str):
    """
    Update email in the auth table while keeping hashes intact.
    """
    try:
        conn = get_db_connection()
        conn.execute("UPDATE users SET email = ? WHERE email = ?", (new_email, old_email))
        conn.commit()
        return True, None
    except sqlite3.IntegrityError as e:
        return False, "Email already exists. Choose another."
    except Exception as e:
        return False, str(e)

# ---------------------- USER HISTORY TABLE ----------------------
def create_user_history_table():
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT,
            date TEXT,
            weight_kg REAL,
            calorie_intake REAL,
            protein_g REAL,
            carbs_g REAL,
            fats_g REAL
        )
    """)
    conn.commit()
create_user_history_table()

def insert_history_entry(user_email, date, weight_kg, calorie_intake, protein_g=None, carbs_g=None, fats_g=None):
    conn = get_db_connection()
    conn.execute("""
        INSERT INTO user_history (user_email, date, weight_kg, calorie_intake, protein_g, carbs_g, fats_g)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (user_email, date, weight_kg, calorie_intake, protein_g, carbs_g, fats_g))
    conn.commit()

def load_history_df():
    conn = get_db_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM user_history", conn)
    except Exception:
        df = pd.DataFrame(columns=['id', 'user_email', 'date', 'weight_kg', 'calorie_intake', 'protein_g', 'carbs_g', 'fats_g'])
    return df

# ---------------------- Recipe utilities ----------------------
def load_recipes():
    path = os.path.join(REPO_ROOT, "db", "recipes.csv")
    return pd.read_csv(path)

def debug_table_columns(table_name):
    conn = get_db_connection()
    table_name_quoted = f'"{table_name}"'
    st.write(f"Checking columns in: {table_name}")
    try:
        cursor = conn.execute(f"PRAGMA table_info({table_name_quoted})")
        cols = [row[1] for row in cursor.fetchall()]
        st.write("COLUMNS:", cols)
    except Exception as e:
        st.error(f"Error reading columns: {e}")

def load_ingredients(recipe_name):
    conn = get_db_connection()
    table_name = recipe_name.strip()
    table_name_quoted = f'"{table_name}"'
    try:
        cursor = conn.execute(f"PRAGMA table_info({table_name_quoted})")
        cols = [row[1] for row in cursor.fetchall()]
        if "ingredient" in cols:
            col = "ingredient"
        elif "ingredients" in cols:
            col = "ingredients"
        else:
            st.error(f"⚠ Ingredient column not found in table: {table_name}")
            st.info(f"Columns available: {cols}")
            return []
        df = pd.read_sql_query(f"SELECT {col} FROM {table_name_quoted}", conn)
        return df[col].tolist()
    except Exception as e:
        st.error(f"Error loading ingredients from '{table_name}': {e}")
        return []

# ---------------------- NoSQL replacement functions ----------------------
def save_preferences_nosql(email, preferences, allergies):
    data = json_read("preferences")
    data = [d for d in data if d.get("email") != email]
    data.append({"email": email, "preferences": preferences, "allergies": allergies})
    json_write("preferences", data)
    return True, "Preferences saved (JSON)"

def save_weight_log_nosql(email, weight, bmi, min_weight=None, max_weight=None):
    data = json_read("weight_logs")
    data.append({
        "email": email,
        "weight": float(weight),
        "bmi": float(bmi),
        "target_range": {"min": min_weight, "max": max_weight},
        "date": str(pd.Timestamp.now())
    })
    json_write("weight_logs", data)
    return True, "Weight log saved (JSON)"

def save_meal_log_nosql(email, recipe_name, ingredients, generated_text, nutrition):
    data = json_read("meal_logs")
    data.append({
        "email": email,
        "recipe_name": recipe_name,
        "ingredients": ingredients,
        "generated_recipe": generated_text,
        "nutrition": nutrition,
        "date": str(pd.Timestamp.now())
    })
    json_write("meal_logs", data)
    return True, "Meal log saved (JSON)"

def load_meal_history_nosql(email, limit=50):
    data = json_read("meal_logs")
    return [d for d in data if d.get("email") == email][-limit:]

def load_weight_history_nosql(email, limit=100):
    data = json_read("weight_logs")
    return [d for d in data if d.get("email") == email][-limit:]

# ---------------------- Nutrition extractor & helpers ----------------------
def extract_nutrition_from_text(text):
    import re

    nutrition = {
        "calories": 0.0,
        "protein": 0.0,
        "carbs": 0.0,
        "fats": 0.0,
        "vitamins": 0.0
    }

    if not text:
        return nutrition

    t = text.lower()

    # ---------------- CALORIES ----------------
    # Matches: 250 kcal, calories: 250, calorie 300
    cal_match = re.search(r'(\d+\.?\d*)\s*kcal', t)
    if not cal_match:
        cal_match = re.search(r'calories?\s*[:\-]?\s*(\d+\.?\d*)', t)

    if cal_match:
        val = float(cal_match.group(1))
        if 10 <= val <= 2000:      # safe range
            nutrition["calories"] = val

    # ---------------- PROTEIN ----------------
    p_match = re.search(r'protein\s*[:\-]?\s*(\d+\.?\d*)\s*g', t)
    if p_match:
        val = float(p_match.group(1))
        if 0 <= val <= 150:
            nutrition["protein"] = val

    # ---------------- CARBS ----------------
    c_match = re.search(r'carbs?\s*[:\-]?\s*(\d+\.?\d*)\s*g', t)
    if c_match:
        val = float(c_match.group(1))
        if 0 <= val <= 300:
            nutrition["carbs"] = val

    # ---------------- FATS ----------------
    f_match = re.search(r'fats?\s*[:\-]?\s*(\d+\.?\d*)\s*g', t)
    if f_match:
        val = float(f_match.group(1))
        if 0 <= val <= 150:
            nutrition["fats"] = val

    return nutrition
def show_nutrition_piechart(nutrition_dict):
    labels = []
    values = []
    for key, val in nutrition_dict.items():
        try:
            if val is not None and float(val) > 0:
                labels.append(key.capitalize())
                values.append(float(val))
        except:
            continue
    if not values:
        st.warning("No nutrition values available to plot.")
        return
    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

def extract_quantity(ingredient, text):
    pattern = fr"{re.escape(ingredient)}.*?(\d+\.?\d*)\s*(g|ml|tbsp|tsp|cup|pcs|piece|kg|mg)?"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        num = match.group(1)
        unit = match.group(2) if match.group(2) else ""
        return f"{num} {unit}".strip()
    return "-"

def _load_image_as_base64(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""

# ---------------------- HOME PAGE (Guest / Login) ----------------------
def home_page():
    st.set_page_config(page_title="Nutri-Chef", page_icon="🍽️", layout="wide")

    # Load background image as base64
    bg_path = os.path.join(BASE_DIR, "assets", "home_bg.jpg")
    try:
        with open(bg_path, "rb") as f:
            bg_base64 = base64.b64encode(f.read()).decode()
        background_css = f"background-image: url('data:image/png;base64,{bg_base64}');"
    except:
        background_css = "background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);"  # fallback

    st.markdown(
    f"""
    <style>
        .home-hero {{
            height: 100vh;
            width: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            {background_css}
            background-size: cover;
            background-position: center;
            color: white;
            text-align: center;
            padding: 0;
            margin: 0;
        }}

        .home-title {{
            font-size: 56px;
            font-weight: 900;
            margin-bottom: 6px;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.5);
        }}

        .home-sub {{
            font-size: 20px;
            margin-bottom: 30px;
            text-shadow: 2px 2px 6px rgba(0,0,0,0.5);
        }}

        .home-cta {{
            width: 100%;
            display: flex;
            justify-content: center;
            margin-top: 28px;
            padding: 18px 0 48px 0;
        }}

        .home-cta-inner {{
            width: 100%;
            max-width: 820px;
            display: flex;
            gap: 20px;
            justify-content: center;
            align-items: center;
        }}

        /* enlarge buttons */
        .stButton>button, .stButton>div>button {{
            font-size: 18px !important;
            padding: 14px 28px !important;
            border-radius: 12px !important;
        }}

        /* ⭐ ADD BUTTON BORDER HERE */
        .stButton>button, .stButton>div>button {{
            
border: 3px solid #FFD700 !important; /* thick yellow border */
    background-color: rgba(255, 255, 224, 0.85) !important; /* light yellow background */
    color: black !important;
    font-weight: 600 !important;
    transition: 0.25s ease-in-out;

        }}

        .stButton>button:hover, .stButton>div>button:hover {{
           
background-color: rgba(152, 251, 152, 0.85) !important; /* light green background */
    border-color: #228B22 !important; /* thick green border */
    color: black !important;
    transform: scale(1.03);

        }}

        @media (max-width: 700px) {{
            .home-cta-inner {{ flex-direction: column; gap: 12px; padding: 0 16px; }}
        }}
    </style>
    """,
    unsafe_allow_html=True
)

    st.markdown(
        """
        <div class="home-hero">
            <div class="home-title">🍽️ Nutri-Chef</div>
            <div class="home-sub">Your Personal AI-Powered Chef — choose a mode to continue</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='home-cta'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("👤 Continue as Guest", key="guest_mode_btn", use_container_width=True):
            st.session_state["guest"] = True
            st.session_state["authenticated"] = False
            st.session_state["page"] = "food_pref"
            st.rerun()
    with col2:
        if st.button("🔐 Login / Create Account", key="go_login_btn", use_container_width=True):
            st.session_state["guest"] = False
            st.session_state["page"] = "login"
            st.rerun()

def login_page():
    import re
    st.set_page_config(page_title="Nutri-Chef - Login", page_icon="🍽️", layout="wide")

    # ------------------ CSS ------------------
    st.markdown("""
        <style>
            .block-container {
                padding-left: 0 !important;
                padding-right: 0 !important;
            }


            .main {
                background: linear-gradient(135deg,#fdfbfb 0%, #ebedee 100%);
            }
            
        /* Make the tabs 50% each */
        .stTabs [role="tab"] {
            flex: 1 !important;
            width: 50% !important;
            text-align: center !important;
            justify-content: center !important;
            font-size: 18px !important;
            padding: 10px 0 !important;
        }

            /* Center text input fields */
            input[type="text"], input[type="password"] {
                width: 100% !important;
                border-radius: 8px;
                padding: 10px;
            }

            .login-title {
                text-align: center;
                font-size: 26px;
                font-weight: 800;
                margin-bottom: 18px;
            }

            .input-error {
                color: red;
                font-size: 13px;
                margin-top: -8px;
                margin-bottom: 6px;
            }

            .rule-bad  { color: red; font-size: 14px; }
            .rule-good { color: green; font-size: 14px; }

        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <h1 style="text-align:center; color:#333; margin-top:10px;">🍽️ Nutri-Chef</h1>
        <h3 style="text-align:center; color:#555;">Your Personal AI-Powered Chef</h3>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


    tab_reg, tab_login = st.tabs(["📝 Register", "🔐 Login"])


    with tab_reg:

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:

            st.markdown('<div class="login-card">', unsafe_allow_html=True)
            st.markdown('<div class="login-title">Create Your Account</div>', unsafe_allow_html=True)

            # FIRST NAME
            fname = st.text_input("First Name", key="reg_fname_inp")
            if fname and not fname.isalpha():
                st.markdown('<div class="input-error">❌ Only alphabets allowed</div>', unsafe_allow_html=True)

            # LAST NAME
            lname = st.text_input("Last Name", key="reg_lname_inp")
            if lname and not lname.isalpha():
                st.markdown('<div class="input-error">❌ Only alphabets allowed</div>', unsafe_allow_html=True)

            # EMAIL
            email = st.text_input("Email", key="reg_email_inp")
            email_valid = ("@" in email) and email.endswith(".com")
            if email:
                st.markdown(
                    f"<div class='{ 'rule-good' if email_valid else 'rule-bad' }'>"
                    f"{'✔ Valid Email' if email_valid else '✖ Must contain @ and end with .com'}</div>",
                    unsafe_allow_html=True
                )

            # PHONE
            raw_phone = st.text_input("Phone Number (10 digits)", key="reg_phone_input")
            filtered_phone = "".join([c for c in raw_phone if c.isdigit()])
            phone_valid = filtered_phone.isdigit() and len(filtered_phone) == 10

            if raw_phone:
                if not raw_phone.isdigit():
                    st.markdown('<div class="input-error">❌ Only digits allowed</div>', unsafe_allow_html=True)
                elif len(raw_phone) != 10:
                    st.markdown('<div class="input-error">❌ Enter exactly 10 digits</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="color:green; font-size:14px;">✔ Valid phone number</div>', unsafe_allow_html=True)

            # PASSWORD
            pwd = st.text_input("Password", type="password", key="reg_pwd_inp")
            if pwd:
                st.markdown(
                    f"""
                    <div class="{ 'rule-good' if len(pwd)>=8 else 'rule-bad' }">• Min 8 characters</div>
                    <div class="{ 'rule-good' if any(c.isupper() for c in pwd) else 'rule-bad' }">• At least 1 uppercase letter</div>
                    <div class="{ 'rule-good' if any(c.isdigit() for c in pwd) else 'rule-bad' }">• At least 1 number</div>
                    <div class="{ 'rule-good' if any(c in '@$!#%&*?' for c in pwd) else 'rule-bad' }">• At least 1 special character</div>
                    """,
                    unsafe_allow_html=True
                )

            # RE-TYPE PASSWORD
            pwd2 = st.text_input("Retype Password", type="password", key="reg_pwd2_inp")
            if pwd2 and pwd != pwd2:
                st.markdown('<div class="input-error">❌ Passwords do not match</div>', unsafe_allow_html=True)

            # SUBMIT
            if st.button("Register", use_container_width=True):
                all_valid = (
                    fname.isalpha() and lname.isalpha() and
                    email_valid and phone_valid and
                    len(pwd) >= 8 and
                    any(c.isupper() for c in pwd) and
                    any(c.isdigit() for c in pwd) and
                    any(c in "@$!#%&*?" for c in pwd) and
                    pwd == pwd2
                )

                if not all_valid:
                    st.error("⚠ Please fix all errors before submitting.")
                else:
                    ok, msg = register_user(email.lower().strip(), pwd)
                    if ok:
                        meta = json_read("users_meta")
                        meta = [m for m in meta if m.get("email") != email.lower()]
                        meta.append({
                            "email": email.lower(),
                            "first_name": fname,
                            "last_name": lname,
                            "phone": filtered_phone,
                            "bmi_completed": False,
                            "plan": None
                        })
                        json_write("users_meta", meta)

                        st.success(f"Account Created Successfully! Welcome, {lname}!")
                        st.session_state["authenticated"] = True
                        st.session_state["user_email"] = email.lower()
                        st.session_state["page"] = "bmi"
                        st.rerun()
                    else:
                        st.error(msg)

            st.markdown('</div>', unsafe_allow_html=True)
    with tab_login:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:

            st.markdown('<div class="login-card">', unsafe_allow_html=True)
            st.markdown('<div class="login-title">Login</div>', unsafe_allow_html=True)

            login_email = st.text_input("Email", key="login_email_inp")
            login_pwd = st.text_input("Password", type="password", key="login_pwd_inp")
            if st.button("Login", use_container_width=True):
                ok, msg = login_user(login_email.lower().strip(), login_pwd)
                if ok:
                    # ALWAYS navigate to food_pref after login (per your requirement)
                    meta = json_read("users_meta")
                    user_meta = next((m for m in meta if m.get("email") == login_email.lower()), None)
                    last_name = user_meta.get("last_name", "") if user_meta else ""

                    st.success(f"Login successful! Welcome, {last_name}!")
                    st.session_state["authenticated"] = True
                    st.session_state["user_email"] = login_email.lower()

                    # Force nav to food_pref regardless of BMI status (as requested)
                    st.session_state["page"] = "food_pref"
                    st.rerun()
                else:
                    st.error(msg)

import requests

GROQ_API_KEY = "YOUR_GROQ_API_KEY"

def chat_llama(messages):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages
    }

    res = requests.post(url, headers=headers, json=payload)
    data = res.json()

    if "error" in data:
        return f"❗ API Error: {data['error'].get('message','Unknown error')}"

    return data["choices"][0]["message"]["content"]

            
# ---------------------- BMI page ----------------------
# ---------------------- BMI page with plan selection ----------------------
def bmi_page():
    st.header("BMI Calculator — Let's get your baseline")

    user = st.session_state.get("user_email", "")
    meta = json_read("users_meta")

    # Safe lookup
    user_meta = next((m for m in meta if isinstance(m, dict) and m.get("email") == user), {})
    last_name = user_meta.get("last_name", "")
    st.write(f"Welcome, **{last_name}** — Enter your details to calculate BMI and pick a plan.")

    height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=165.0)
    weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=60.0)
    age = st.number_input("Age", min_value=5, max_value=100, value=25)

    # ------------------ Calculate BMI button ------------------
    if st.button("Calculate BMI"):

        bmi = round(weight / ((height / 100) ** 2), 2)
        st.session_state["bmi_value"] = bmi

        min_weight = round(18.5 * ((height / 100) ** 2), 1)
        max_weight = round(24.9 * ((height / 100) ** 2), 1)

        st.session_state["min_weight"] = min_weight
        st.session_state["max_weight"] = max_weight

        # Determine BMI status
        if bmi < 18.5:
            bmi_status = "Underweight"
        elif 18.5 <= bmi < 25:
            bmi_status = "Normal"
        elif 25 <= bmi < 30:
            bmi_status = "Overweight"
        else:
            bmi_status = "Obese"

        # Weight to lose/gain
        if weight > max_weight:
            weight_to_lose = round(weight - max_weight, 1)
            weight_to_gain = 0
        elif weight < min_weight:
            weight_to_gain = round(min_weight - weight, 1)
            weight_to_lose = 0
        else:
            weight_to_lose = 0
            weight_to_gain = 0

        # Display BMI results
        st.success(f"Your BMI is **{bmi}** ({bmi_status})")

        if weight_to_lose > 0:
            st.warning(f"⚖ You need to lose **{weight_to_lose} kg** to reach a healthy weight.")
        elif weight_to_gain > 0:
            st.info(f"⚖ You need to gain **{weight_to_gain} kg** to reach a healthy weight.")
        else:
            st.success("🎉 You are already within a healthy weight range!")

        # -------- SAVE USER META --------
        meta = json_read("users_meta")
        for m in meta:
            if m["email"] == user:
                m["bmi_value"] = bmi
                m["bmi_status"] = bmi_status
                m["min_weight"] = min_weight
                m["max_weight"] = max_weight
                m["current_weight"] = weight
                m["weight_to_lose"] = weight_to_lose
                m["weight_to_gain"] = weight_to_gain
                break

        json_write("users_meta", meta)

        # ⭐ -------- SAVE WEIGHT HISTORY FOR PREDICTION PAGE --------
        save_weight_log_nosql(
            email=user,
            weight=weight,
            bmi=bmi,
            min_weight=min_weight,
            max_weight=max_weight
        )

        st.success("BMI recorded & weight history updated! ✔")

    # Stop if BMI not calculated
    if "bmi_value" not in st.session_state:
        return

    # ------------------ Plan Selection ------------------
    st.write("### 📅 Choose Your Plan")

    plan = st.selectbox("Choose your plan", ["1 Month", "3 Months", "6 Months", "1 Year"])
    schedule_type = st.radio("Time Schedule", ["Recommended Times", "Custom Time"], horizontal=True)

    if schedule_type == "Recommended Times":
        chosen_times = st.selectbox(
            "Select a recommended time pattern:",
            ["8 AM – 12 PM – 7 PM", "9 AM – 1 PM – 8 PM", "7 AM – 11 AM – 6 PM"],
        )
    else:
        morning = st.time_input("Morning time")
        afternoon = st.time_input("Afternoon time")
        evening = st.time_input("Evening time")
        chosen_times = f"{morning} – {afternoon} – {evening}"

    # ------------------ Final Save ------------------
    if st.button("Save Plan & Continue"):
        meta = json_read("users_meta")

        for m in meta:
            if m["email"] == user:
                m["bmi_completed"] = True
                m["plan"] = {
                    "duration": plan,
                    "times": chosen_times,
                    "start_date": str(pd.Timestamp.now())
                }

        json_write("users_meta", meta)

        st.success("Plan saved! Redirecting…")
        st.session_state["page"] = "food_pref"
        st.rerun()

# ---------------------- Food preference page ----------------------
def food_pref_page():
    st.header("Choose Your Food Preference")
    st.write("Please select one option below:")
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🥗 Vegetarian", key="veg_pref_btn", use_container_width=True):
            st.session_state["preference"] = "veg"
            st.session_state["page"] = "recipe_categories"
            st.rerun()
    with col2:
        if st.button("🍗 Non-Vegetarian", key="nonveg_pref_btn", use_container_width=True):
            st.session_state["preference"] = "nonveg"
            st.session_state["page"] = "recipe_categories"
            st.rerun()
    st.write("")
    if st.button("⬅ Back to Dashboard", key="back_to_dashboard_btn"):
        st.session_state["page"] = "profile" if st.session_state.get("authenticated") else "home"
        st.rerun()

# ---------------------- Recipe Categories / Ingredients / Generation ----------------------
def recipe_categories_page():
    st.subheader("🔍 Can't find your recipe?")
    manual_name = st.text_input("Enter recipe name manually (optional)")

    if st.button("Search Recipe", key="manual_recipe_search"):
        if manual_name.strip():
        # Save manual recipe to session
            st.session_state["manual_recipe_name"] = manual_name.strip()
            st.session_state["page"] = "manual_ingredients"
            st.rerun()
        else:
            st.warning("Please enter a recipe name.")

    st.header("Choose a Recipe Category")
    pref = st.session_state.get("preference", "veg")
    df = load_recipes()
    st.subheader("🥤 Smoothies (Common for Veg & Non-Veg)")
    smoothies = df[(df["category"] == "smoothie") & (df["type"] == "common")]
    for i, row in smoothies.iterrows():
        recipe = row["recipe_name"]
        if st.button(recipe, key=f"smoothie_{i}"):
            st.session_state["selected_recipe"] = recipe
            st.session_state["page"] = "ingredients"
            st.rerun()
    st.markdown("---")
    st.subheader("🍚 Rice Bowls")
    rice_bowls = df[(df["category"] == "rice_bowl") & (df["type"] == pref)]
    for i, row in rice_bowls.iterrows():
        recipe = row["recipe_name"]
        if st.button(recipe, key=f"rice_{i}"):
            st.session_state["selected_recipe"] = recipe
            st.session_state["page"] = "ingredients"
            st.rerun()
    st.markdown("---")
    st.subheader("🥗 Salads")
    salads = df[(df["category"] == "salad") & (df["type"] == pref)]
    for i, row in salads.iterrows():
        recipe = row["recipe_name"]
        if st.button(recipe, key=f"salad_{i}"):
            st.session_state["selected_recipe"] = recipe
            st.session_state["page"] = "ingredients"
            st.rerun()
    st.markdown("---")
    st.subheader("🍛 Curries")
    curries = df[(df["category"] == "curry") & (df["type"] == pref)]
    for i, row in curries.iterrows():
        recipe = row["recipe_name"]
        if st.button(recipe, key=f"curry_{i}"):
            st.session_state["selected_recipe"] = recipe
            st.session_state["page"] = "ingredients"
            st.rerun()
    st.markdown("---")
    if st.button("⬅ Back to Preference Page", key="back_to_pref_btn"):
        st.session_state["page"] = "food_pref"
        st.rerun()
def ingredients_page():
    recipe = st.session_state.get("selected_recipe")
    if not recipe:
        st.error("No recipe selected!")
        return

    st.header(f"Ingredients for {recipe}")

    # Load ingredients
    ingredients = load_ingredients(recipe)
    if not ingredients:
        st.warning("⚠ No ingredients found in the database for this recipe!")
        return

    st.subheader("Select Ingredients:")

    # ----------------------------------------
    # ⭐ ADD SELECT ALL CHECKBOX HERE
    # ----------------------------------------
    if "select_all" not in st.session_state:
        st.session_state["select_all"] = False

    select_all = st.checkbox("Select All", key="select_all")

    selected = []

    # Render all ingredient checkboxes
    for idx, ing in enumerate(ingredients):
        cb_key = f"chk_ing_{idx}_{ing}"

        # If SELECT ALL is clicked → set all boxes True
        if select_all:
            st.session_state[cb_key] = True

        checked = st.checkbox(ing, key=cb_key)

        if checked:
            selected.append(ing)

    st.session_state["selected_ingredients"] = selected

    # ----------------------------------------
    # Preferences Section
    # ----------------------------------------
    st.markdown("---")
    st.subheader("Preferences & Allergies")

    pref_text = st.text_area("Your Preferences (optional):", 
                             value=st.session_state.get("preferences", ""), 
                             key="prefs_textarea")

    allergy_text = st.text_area("Allergies (optional):",
                                value=st.session_state.get("allergies", ""), 
                                key="allergy_textarea")

    st.session_state["preferences"] = pref_text
    st.session_state["allergies"] = allergy_text

    # Navigation Buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("⬅ Back to Categories", key="back_to_categories_btn"):
            st.session_state["page"] = "recipe_categories"
            st.rerun()

    with col2:
        if st.button("Make Recipe", key="make_recipe_btn"):
            email = st.session_state.get("user_email")
            if email:
                ok, msg = save_preferences_nosql(email, pref_text, allergy_text)
                if not ok:
                    st.warning("Could not save preferences: " + msg)

            st.session_state["page"] = "generate_recipe"
            st.rerun()

def manual_ingredients_page():
    recipe_name = st.session_state.get("manual_recipe_name", "")

    if not recipe_name:
        st.error("No recipe name provided.")
        return

    st.title(f"🧾 Ingredients for {recipe_name} (AI Fetched)")

    st.info("Fetching ingredients using Gemini…")

    # Ask Gemini to extract only list of ingredients
    prompt = f"""
    List the ingredients required to make the recipe '{recipe_name}'.
    Return ONLY a list of ingredient names, one per line.
    Do NOT include steps, quantities, or explanation.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        ingredients_text = response.text.strip()
        ingredients_list = [line.strip("-• ").strip() for line in ingredients_text.split("\n") if line.strip()]

        if not ingredients_list:
            st.error("No ingredients found.")
            return

        st.session_state["selected_recipe"] = recipe_name
        st.session_state["manual_ingredients_list"] = ingredients_list

        st.subheader("Select Ingredients:")
        selected = []
        for ing in ingredients_list:
            if st.checkbox(ing, key=f"manual_chk_{ing}"):
                selected.append(ing)

        st.session_state["selected_ingredients"] = selected

        if st.button("Make Recipe", key="manual_make_recipe"):
            st.session_state["page"] = "generate_recipe"
            st.rerun()

        if st.button("⬅ Back", key="manual_back"):
            st.session_state["page"] = "recipe_categories"
            st.rerun()

    except Exception as e:
        st.error(f"Gemini error: {e}")



def generate_recipe_page(guest_mode=False):
    st.header("Your Personalized Recipe")

    ingredients = st.session_state.get("selected_ingredients", [])
    preferences = st.session_state.get("preferences", "")
    allergies = st.session_state.get("allergies", "")
    bmi = st.session_state.get("bmi_value", 22)
    recipe_name = st.session_state.get("selected_recipe", "Custom Recipe")
    email = st.session_state.get("user_email")

    # Load user BMI info
    users = json_read("users_meta")
    user_meta = next((u for u in users if u.get("email") == email), {})

    bmi_status = user_meta.get("bmi_status")
    current_weight = user_meta.get("current_weight")
    min_weight = user_meta.get("min_weight")
    max_weight = user_meta.get("max_weight")
    weight_to_lose = user_meta.get("weight_to_lose")
    weight_to_gain = user_meta.get("weight_to_gain")

    if not ingredients:
        st.error("No ingredients selected!")
        return

    # ---------------- Gemini Prompt (Updated & Strict) ---------------- #

    prompt = f"""
    You are a diet-specialized nutritionist AI.

    Create a healthy weight-managed recipe with these inputs:

    Recipe Name: {recipe_name}
    Selected Ingredients: {ingredients}
    User Preferences: {preferences}
    Allergies: {allergies}

    User Health Data:
    - BMI: {bmi}
    - BMI Status: {bmi_status}
    - Current Weight: {current_weight} kg
    - Healthy Weight Range: {min_weight}–{max_weight} kg
    - Weight to Lose: {weight_to_lose} kg
    - Weight to Gain: {weight_to_gain} kg

    ---------------- DIET RULES ----------------

    For BMI < 18.5 (Underweight):
      - Provide high-calorie meals: 500–700 kcal
      - Increase carbohydrates & protein
      - Include healthy fats

    For BMI 18.5–24.9 (Normal):
      - Provide balanced meals: 350–500 kcal

    For BMI 25–29.9 (Overweight):
      - Provide weight-loss meals: 250–350 kcal

    For BMI ≥ 30 (Obese):
      - STRICT weight-loss meals: 200–300 kcal MAX
      - Very low carbs, low fat, fiber-rich

    --------------------------------------------------------------
    OUTPUT FORMAT (VERY IMPORTANT):

    Ingredients (with grams):
    - item: grams

    Steps:
    1. step
    2. step

    Nutrition:
    - Calories: <number> kcal
    - Protein: <number> g
    - Carbs: <number> g
    - Fats: <number> g

    Follow this format exactly.
    """

    # ------------------------------------------------------------- #

    st.info("Generating recipe...")
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        recipe_text = response.text

        st.success("Recipe Ready!")
        st.markdown(recipe_text)

        # Extract & save nutrition
        nutrition = extract_nutrition_from_text(recipe_text)
        st.session_state["generated_recipe_text"] = recipe_text
        st.session_state["nutri_ingredients"] = ingredients
        st.session_state["nutrition_values"] = nutrition

        # Save meal log if logged in
        if email and not guest_mode:
            ok, msg = save_meal_log_nosql(email, recipe_name, ingredients, recipe_text, nutrition)
            if not ok:
                st.warning("Could not save meal log: " + msg)
            else:
                st.success("Saved recipe & nutrition to your history.")

        if st.button("📊 Show Nutrition Chart Page"):
            st.session_state["page"] = "nutrition_chart"
            st.rerun()

    except Exception as e:
        st.error(f"Failed to generate recipe: {e}")

    if st.button("⬅ Back to Ingredients"):
        st.session_state["page"] = "ingredients"
        st.rerun()

# ---------------------- Nutrition Chart ----------------------
def nutrition_chart_page():
    st.header("🍽 Nutrition Breakdown")
    recipe_name = st.session_state.get("selected_recipe", "Recipe")
    recipe_text = st.session_state.get("generated_recipe_text", "")
    ingredients_list = st.session_state.get("nutri_ingredients", [])
    nutrition = st.session_state.get("nutrition_values", {})
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader(f"Ingredients for {recipe_name}")
        rows = []
        for item in ingredients_list:
            qty = extract_quantity(item, recipe_text)
            rows.append([item, qty])
        df = pd.DataFrame(rows, columns=["Ingredient", "Quantity"])
        st.table(df)
    with col2:
        st.subheader("Nutrition Pie Chart")
        show_nutrition_piechart(nutrition)
        st.markdown("**Nutrition values:**")
        for k, v in nutrition.items():
            if v and float(v) > 0:
                unit = "kcal" if k == "calories" else "g"
                st.write(f"- {k.capitalize()}: {v} {unit}")
    if st.button("⬅ Back to Recipe", key="back_from_nutri_btn"):
        st.session_state["page"] = "generate_recipe"
        st.rerun()

# ---------------------- Profile & small pages ----------------------
def _ensure_nosql_files():
    for name in ("users_meta", "user_plans", "meal_logs", "weight_logs", "preferences"):
        p = json_path(name)
        if not os.path.exists(p):
            json_write(name, [])

def load_user_profile(email):
    _ensure_nosql_files()
    profiles = json_read("users_meta")
    for p in profiles:
        if p.get("email") == email:
            return p
    return {"email": email, "first_name": "", "last_name": "", "phone": ""}

def save_user_profile(profile):
    _ensure_nosql_files()
    profiles = json_read("users_meta")
    profiles = [p for p in profiles if p.get("email") != profile.get("email")]
    profiles.append(profile)
    json_write("users_meta", profiles)

def load_user_plans(email):
    _ensure_nosql_files()
    plans = json_read("user_plans")
    return [p for p in plans if p.get("email") == email]

def save_user_plans(plans_all):
    json_write("user_plans", plans_all)

def load_user_meals(email, limit=50):
    _ensure_nosql_files()
    meals = json_read("meal_logs")
    user_meals = [m for m in meals if m.get("email") == email]
    return user_meals[-limit:]

def load_user_weight_logs(email, limit=200):
    try:
        conn = get_db_connection()
        df = pd.read_sql_query("SELECT * FROM user_history WHERE user_email = ? ORDER BY date", conn, params=(email,))
        if df.shape[0] > 0:
            try:
                df['date'] = pd.to_datetime(df['date'])
            except:
                pass
            return df.tail(limit)
    except Exception:
        pass
    _ensure_nosql_files()
    logs = json_read("weight_logs")
    user = [l for l in logs if l.get("email") == email]
    if not user:
        return pd.DataFrame(columns=["date", "weight"])
    df = pd.DataFrame(user)
    if "date" in df.columns and "weight" in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'])
        except:
            pass
        return df.tail(limit)
    return pd.DataFrame(user).tail(limit)

def profile_page():
    st.set_page_config(layout="wide")

    # Ensure local files
    try:
        _ensure_nosql_files()
    except:
        pass

    email = st.session_state.get("user_email")
    if not email:
        st.error("No logged-in user.")
        return

    meta = json_read("users_meta")
    user_meta = next((m for m in meta if m.get("email") == email), {})

    # Check if PROFILE MODE is active
    profile_mode = st.session_state.get("open_profile_page", False)

    # ----------------------- 1️⃣ NORMAL DASHBOARD PAGE -----------------------
    if not profile_mode:

        top_cols = st.columns([4, 1])
        with top_cols[0]:
            st.markdown(f"### Hi, {user_meta.get('last_name', '')} 👋")
            st.caption("Manage your personal dashboard")

        with top_cols[1]:
            if st.button("👤 Profile"):
                st.session_state["open_profile_page"] = True
                st.rerun()

        st.markdown("---")

        # Dashboard sections...
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("📇 Account")
            st.write("Name:", f"{user_meta.get('first_name','')} {user_meta.get('last_name','')}")
            st.write("Email:", email)
            st.write("Phone:", user_meta.get("phone", "-"))

        with c2:
            st.subheader("⚖ BMI")
            st.write("BMI:", user_meta.get("bmi_value", "-"))
            st.write("Status:", user_meta.get("bmi_status", "-"))

        with c3:
            st.subheader("📅 Plan Info")
            plan = user_meta.get("plan") or {}
            st.write("Duration:", plan.get("duration", "-"))
            st.write("Times:", plan.get("times", "-"))

        st.markdown("---")

        

        return  # END DASHBOARD PAGE

    # ----------------------- 2️⃣ PROFILE SETTINGS PAGE -----------------------
    st.markdown("## ⚙ Profile Settings")
    st.caption("Edit your info, view history, or logout")

    # ----- EDIT PROFILE -----
    st.markdown("### ✏ Edit Profile")
    with st.form("edit_profile_form"):
        new_fn = st.text_input("First Name", user_meta.get("first_name", ""))
        new_ln = st.text_input("Last Name", user_meta.get("last_name", ""))
        new_email = st.text_input("Email", email)
        new_phone = st.text_input("Phone", user_meta.get("phone", ""))

        save_btn = st.form_submit_button("Save Changes")

    if save_btn:
        current_email = email
        cleaned_email = new_email.lower().strip()

        # Update auth table if email changed
        if cleaned_email != current_email:
            ok, err = update_user_email(current_email, cleaned_email)
            if not ok:
                st.error(err)
            else:
                st.session_state["user_email"] = cleaned_email
                email = cleaned_email

        # Update NoSQL metadata
        for m in meta:
            if m.get("email") == current_email:
                m["email"] = email
                m["first_name"] = new_fn
                m["last_name"] = new_ln
                m["phone"] = new_phone

        json_write("users_meta", meta)
        st.success("Profile updated!")

    st.markdown("---")

    # ----- NAVIGATION -----
    st.markdown("📚 History")
    if st.button("📘 Meal History"):
        st.session_state["page"] = "meal_history"
        st.session_state["open_profile_page"] = False
        st.rerun()

    if st.button("📉 Weight History"):
        st.session_state["page"] = "profile_weights"
        st.session_state["open_profile_page"] = False
        st.rerun()

    st.markdown("---")

    # ----- ACCOUNT ACTIONS -----
    st.markdown("### 🔐 Account")

    if st.button("🚪 Logout"):
        st.session_state.clear()
        st.session_state["page"] = "login"
        st.rerun()

    if st.button("🗑 Delete Account"):
        # Remove meta entry
        meta = [m for m in meta if m.get("email") != email]
        json_write("users_meta", meta)

        # Remove meals
        logs = json_read("meal_logs")
        logs = [m for m in logs if m.get("email") != email]
        json_write("meal_logs", logs)

        st.session_state.clear()
        st.session_state["page"] = "login"
        st.success("Account deleted")
        st.rerun()

    st.markdown("---")

    # ----- CLOSE BUTTON -----
    if st.button("❌ Close"):
        st.session_state["open_profile_page"] = False
        st.rerun()

# Extra small pages
def meal_history_page():
    user = st.session_state.get("user_email")
    st.header("Your Meal History")
    if st.button("⬅️ Back to Profile"):
        st.session_state["open_profile_page"] = True
        st.session_state["page"] = "profile"
        st.rerun()
    meals = load_meal_history_nosql(user)
    if meals:
        df = pd.DataFrame(meals)
        st.dataframe(df[['date', 'recipe_name', 'generated_recipe']].tail(50))
    else:
        st.info("No meals saved yet.")

def settings_page():
    st.header("Settings")
    user = st.session_state.get("user_email")
    meta = json_read("users_meta")
    user_meta = next((m for m in meta if m.get("email") == user), {})
    st.write("Notification plan:", user_meta.get("plan", {}))
    if st.button("Delete account data (meta & meal history)", key=f"del_data_{user}"):
        meta = [m for m in meta if m.get("email") != user]
        json_write("users_meta", meta)
        meals = json_read("meal_logs")
        meals = [m for m in meals if m.get("email") != user]
        json_write("meal_logs", meals)
        st.success("Local account data removed (meta & meal history). To remove SQL account, delete from DB manually.")

def profile_weights_page():
    email = st.session_state.get("user_email")
    st.header("📉 Weight History")

    # ------------------ WEIGHT LOGS ------------------
    logs = load_weight_history_nosql(email)

    if logs:
        df = pd.DataFrame(logs)

        if "date" in df.columns:
            try:
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date")
            except:
                pass

        # Line chart (optional)
        if "weight" in df.columns and "date" in df.columns:
            st.subheader("📈 Weight Change Over Time")
            st.line_chart(df.set_index("date")["weight"])

        st.subheader("📋 Recent Weight Logs")
        st.table(df.tail(50).fillna("-"))

    else:
        st.info("No weight entries yet.")

    st.markdown("---")

    # ------------------ WEIGHT PREDICTION ------------------
    st.subheader("🤖 Weight Prediction")
    prediction = estimate_weight_loss_from_meals(email)

    if prediction:
        loss = prediction["predicted_loss_30_days"]
        days = prediction["days_needed"]

        st.write(f"📉 **Based on your past meals, you may lose {loss} kg in the next 30 days.**")

        if days and days > 0:
            st.write(f"🎯 **Estimated days to reach your goal weight: {days} days**")
        else:
            st.warning("⚠ **Your calorie intake is too high for weight loss.**")
    else:
        st.info("Not enough meal data to generate a prediction yet.")

    st.markdown("---")

    # ------------------ BACK BUTTON ------------------
    if st.button("⬅️ Back to Profile"):
        st.session_state["open_profile_page"] = True
        st.session_state["page"] = "profile"
        st.rerun()

 
# ---------------------------------------
# LAYOUT
# ---------------------------------------
left, center, right = st.columns([2, 6, ])
 
 
# ---------------------------------------
# RIGHT CHAT PANEL
# ---------------------------------------
with right:
    if st.session_state.bot_open:
 
        st.markdown("<div class='bot-box'>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:center;color:#2c7d3d;'>🥗 NutriBot</h3>", unsafe_allow_html=True)
 
        # INITIAL MESSAGE
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I’m NutriBot 💚. How can I help you today?"}
            ]
 
        # DISPLAY CHAT
        for msg in st.session_state.messages:
            bg = "#e8ffe8" if msg["role"] == "user" else "#f9f9f9"
            align = "right" if msg["role"] == "user" else "left"
 
            st.markdown(
                f"<div style='background:{bg};padding:10px;border-radius:10px;margin:6px;text-align:{align};'>{msg['content']}</div>",
                unsafe_allow_html=True
            )
 
        # -------------------------
        # MESSAGE SEND FUNCTION
        # -------------------------
        def send_message():
            user_msg = st.session_state.chat_input

            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_msg})

            # Get model response
            reply = chat_llama(st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": reply})

            # Clear the input safely
            st.session_state.chat_input = ""
 
        # -------------------------
        # CHAT INPUT (SAFE CLEAR)
        # -------------------------
        st.text_input(
            "Type your message...",
            key="chat_input",
            on_change=send_message
        )
 
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------- APP ROUTER & MAIN ----------------------
def main():
    # init session state keys
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "page" not in st.session_state:
        st.session_state["page"] = "home"
    if "guest" not in st.session_state:
        st.session_state["guest"] = False
    if "theme" not in st.session_state:
        st.session_state["theme"] = "light"

    # Apply a simple theme toggle without altering logic
    if st.session_state["theme"] == "dark":
        st.markdown("""
            <style>
                .main { background: #0d1117; color: #e6edf3; }
                .stApp { background: #0d1117; }
            </style>
        """, unsafe_allow_html=True)

    page = st.session_state.get("page", "home")

    # Home (Guest vs Login)
    if page == "home":
        home_page()
        return

    # Login page
    if page == "login" and not st.session_state.get("authenticated", False):
        login_page()
        return

    # Guest flow
    if st.session_state.get("guest", False):
        # Guest pages (no profile, no saving history)
        if page == "food_pref":
            food_pref_page()
        elif page == "recipe_categories":
            recipe_categories_page()
        elif page == "ingredients":
            ingredients_page()
        elif page == "manual_ingredients":           # ⭐ ADD THIS LINE
            manual_ingredients_page()
        elif page == "generate_recipe":
            generate_recipe_page(guest_mode=True)
        elif page == "nutrition_chart":
            nutrition_chart_page()
        else:
            st.session_state["page"] = "food_pref"
            st.rerun()
        return

    # Authenticated user flow
    if st.session_state.get("authenticated", False):
        st.sidebar.title("Nutri-Chef")
        if st.sidebar.button("Profile", key="nav_profile_btn"):
            st.session_state["page"] = "profile"
        if st.sidebar.button("BMI", key="nav_bmi_btn"):
            st.session_state["page"] = "bmi"
        if st.sidebar.button("Food Preference", key="nav_foodpref_btn"):
            st.session_state["page"] = "food_pref"

        page = st.session_state.get("page", "profile")
        if page == "profile":
            profile_page()
        elif page == "bmi":
            bmi_page()
        elif page == "food_pref":
            food_pref_page()
        elif page == "recipe_categories":
            recipe_categories_page()
        elif page == "ingredients":
            ingredients_page()
        elif page == "manual_ingredients":
            manual_ingredients_page()
        elif page == "generate_recipe":
            generate_recipe_page(guest_mode=False)
        elif page == "nutrition_chart":
            nutrition_chart_page()
        elif page == "meal_history":
            meal_history_page()
        elif page == "settings":
            settings_page()
        elif page == "profile_weights":
            profile_weights_page()
        

        else:
            st.session_state["page"] = "profile"
            st.rerun()
             # ⭐ ALWAYS SHOW THE BOT
        return
    home_page()
if __name__ == "__main__":
    main()
