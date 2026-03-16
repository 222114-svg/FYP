import os
import sys
import json
import datetime
import gradio as gr

# ── make sure the project root is on the path so we can import inference.py ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ── paths ────────────────────────────────────────────────────────────────────
USERS_FILE    = os.path.join(BASE_DIR, "users.json")
FEEDBACK_FILE = os.path.join(BASE_DIR, "feedback.json")
LOGO_PATH     = os.path.join(BASE_DIR, "logo.png")

# ═══════════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    return []

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE) as f:
            return json.load(f)
    return []

def save_feedback(entry):
    data = load_feedback()
    data.append(entry)
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f, indent=2)

def run_inference(video_path: str) -> str:
    """
    Run the full inference pipeline from step4.py.
    Returns the predicted text string.
    """
    if not video_path or not os.path.exists(video_path):
        return "⚠ No valid video file provided."

    try:
        from step4 import (
            SignBridgeInference,
            MODEL_PATH,
            LABEL_ENCODER_PATH
        )

        if not os.path.exists(MODEL_PATH):
            return "⚠ Model path not found: " + MODEL_PATH
        if not os.path.exists(LABEL_ENCODER_PATH):
            return "⚠ Label path not found: " + LABEL_ENCODER_PATH

        # Initialize the inference engine
        engine = SignBridgeInference(MODEL_PATH, LABEL_ENCODER_PATH)
        
        # Run prediction
        result = engine.predict(video_path)
        
        # Return the final sentence result
        return result
        
    except Exception as exc:
        return f"Error: {exc}"

# ═══════════════════════════════════════════════════════════════════════════════
# GRADIO THEME & CSS 
# ═══════════════════════════════════════════════════════════════════════════════

custom_theme = gr.themes.Base(
    primary_hue="blue",
    neutral_hue="slate",
).set(
    body_background_fill="#0d1117",
    body_background_fill_dark="#0d1117",
    body_text_color="#e6edf3",
    body_text_color_dark="#e6edf3",
    background_fill_primary="#161b22",
    background_fill_primary_dark="#161b22",
    background_fill_secondary="#0d1117",
    background_fill_secondary_dark="#0d1117",
    border_color_primary="#30363d",
    border_color_primary_dark="#30363d",
    button_primary_background_fill="#2f81f7",
    button_primary_background_fill_hover="#388bfd",
    button_primary_text_color="#e6edf3",
    button_secondary_background_fill="#161b22",
    button_secondary_background_fill_hover="#30363d",
    button_secondary_border_color="#2f81f7",
    button_secondary_text_color="#2f81f7",
    block_background_fill="#161b22",
    block_background_fill_dark="#161b22",
    block_border_width="1px",
    block_border_color="#30363d",
    block_border_color_dark="#30363d",
    block_radius="12px",
    shadow_drop="0 4px 15px rgba(0,0,0,0.2)",
    input_background_fill="#0d1117",
    input_background_fill_dark="#0d1117",
)

css = """
.gradio-container { border: none !important; }
.logo-img { margin: 0 auto; display: block; max-width: 90px; border-radius: 12px; }
.text-accent { color: #2f81f7 !important; font-weight: bold; }
.text-success { color: #3fb950 !important; }
.text-error { color: #f85149 !important; }
.sidebar-col { border-right: 1px solid #30363d; padding-right: 20px; }
hr { border-color: #30363d !important; }
"""

logo_kwargs = {
    "value": LOGO_PATH if os.path.exists(LOGO_PATH) else None,
    "show_label": False,
    "container": False,
    "interactive": False,
    "elem_classes": "logo-img",
    "visible": os.path.exists(LOGO_PATH)
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

with gr.Blocks(title="Sign Language Translator") as app:
    
    current_user_state = gr.State(None)

    # ────────────────────────────────────────────────────────────────────────
    # AUTHENTICATION VIEWS
    # ────────────────────────────────────────────────────────────────────────
    with gr.Column(visible=True) as login_view:
        gr.Markdown("<br><br><br>")
        with gr.Row():
            with gr.Column(scale=1): pass
            with gr.Column(scale=2, elem_classes="block_background_fill"):
                gr.Image(**logo_kwargs) 
                gr.Markdown("## Welcome back\nSign in to your account", elem_classes="text-center")
                log_user = gr.Textbox(label="Username", placeholder="Enter your username")
                log_pass = gr.Textbox(label="Password", placeholder="Enter your password", type="password")
                log_err  = gr.Markdown("", elem_classes="text-error")
                btn_login = gr.Button("Sign In", variant="primary")
                btn_go_reg = gr.Button("Don't have an account? Register", variant="secondary")
            with gr.Column(scale=1): pass

    with gr.Column(visible=False) as register_view:
        gr.Markdown("<br><br><br>")
        with gr.Row():
            with gr.Column(scale=1): pass
            with gr.Column(scale=2, elem_classes="block_background_fill"):
                gr.Image(**logo_kwargs)
                gr.Markdown("## Create Account\nJoin Sign Language Translator", elem_classes="text-center")
                reg_user  = gr.Textbox(label="Username", placeholder="Choose a username")
                reg_email = gr.Textbox(label="Email", placeholder="your@email.com")
                reg_pass  = gr.Textbox(label="Password", placeholder="Create a password", type="password")
                reg_conf  = gr.Textbox(label="Confirm Password", placeholder="Repeat password", type="password")
                reg_msg   = gr.Markdown("")
                btn_reg = gr.Button("Create Account", variant="primary")
                btn_go_log = gr.Button("Already have an account? Sign In", variant="secondary")
            with gr.Column(scale=1): pass

    # ────────────────────────────────────────────────────────────────────────
    # MAIN APPLICATION VIEW
    # ────────────────────────────────────────────────────────────────────────
    with gr.Row(visible=False) as main_app_view:
        
        # ── SIDEBAR ──
        with gr.Column(scale=2, elem_classes="sidebar-col"):
            gr.Image(**logo_kwargs)
            gr.Markdown("### Sign Language <span class='text-accent'>Translator</span>")
            gr.Markdown("---")
            sidebar_user_badge = gr.Markdown("**Signed in as: ...**")
            gr.Markdown("---")
            gr.Markdown("**NAVIGATION**")
            btn_nav_dash = gr.Button("📊 Dashboard", variant="secondary")
            btn_nav_feed = gr.Button("⭐ Feedback", variant="secondary")
            gr.Markdown("---")
            btn_nav_logout = gr.Button("🚪 Logout", variant="secondary")

        # ── CONTENT AREA ──
        with gr.Column(scale=8):
            
            # --- DASHBOARD ---
            with gr.Column(visible=True) as dashboard_view:
                gr.Markdown("## 📊 Dashboard\nUpload a video or record from camera, then translate.")
                gr.Markdown("---")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📁 Upload Video")
                        upload_vid = gr.Video(sources=["upload"], label="Upload MP4, AVI, MOV")
                    with gr.Column(scale=1):
                        gr.Markdown("### 📷 Open Camera")
                        camera_vid = gr.Video(sources=["webcam"], label="Record Webcam")

                gr.Markdown("---")
                gr.Markdown("### 🔤 Translate")
                translate_status = gr.Markdown("Ready.")
                btn_translate = gr.Button("▶ Translate Selected Video", variant="primary")
                output_text = gr.Textbox(label="Translation Output", lines=4, interactive=False)

            # --- FEEDBACK ---
            with gr.Column(visible=False) as feedback_view:
                gr.Markdown("## ⭐ Feedback\nRate your experience and share your thoughts.")
                gr.Markdown("---")
                feed_rating = gr.Radio(
                    choices=["1 - Poor", "2 - Fair", "3 - Good", "4 - Very Good", "5 - Excellent"], 
                    label="Rate the Translation Quality"
                )
                feed_comment = gr.Textbox(label="Additional Comments", lines=5, placeholder="Share your experience...")
                feed_msg = gr.Markdown("")
                btn_submit_feed = gr.Button("Submit Feedback", variant="primary")

    # ═══════════════════════════════════════════════════════════════════════════════
    # EVENT HANDLERS & LOGIC
    # ═══════════════════════════════════════════════════════════════════════════════

    # --- Login Logic ---
    def process_login(username, password):
        if not username or not password:
            return gr.Column(visible=True), gr.Row(visible=False), gr.Column(visible=True), gr.Column(visible=False), None, "", "**⚠ Please fill in all fields.**"
        
        users = load_users()
        for u in users:
            if u["username"] == username and u["password"] == password:
                return (
                    gr.Column(visible=False), # Hide Login
                    gr.Row(visible=True),     # Show Main App
                    gr.Column(visible=True),  # Show Dashboard
                    gr.Column(visible=False), # Hide Feedback
                    username,                 # Update State
                    f"**Signed in as: {username}**", # Update Badge
                    ""                        # Clear Error
                )
        return gr.Column(visible=True), gr.Row(visible=False), gr.Column(visible=True), gr.Column(visible=False), None, "", "**⚠ Invalid username or password.**"

    btn_login.click(
        fn=process_login,
        inputs=[log_user, log_pass],
        outputs=[login_view, main_app_view, dashboard_view, feedback_view, current_user_state, sidebar_user_badge, log_err]
    )

    # --- Registration Logic ---
    def process_register(username, email, password, confirm):
        if not username or not email or not password or not confirm:
            return "<span class='text-error'>⚠ All fields are required.</span>"
        if password != confirm:
            return "<span class='text-error'>⚠ Passwords do not match.</span>"
        
        users = load_users()
        if any(u["username"] == username for u in users):
            return "<span class='text-error'>⚠ Username already taken.</span>"
            
        users.append({
            "username": username,
            "email": email,
            "password": password,
            "created": datetime.datetime.now().isoformat(),
        })
        save_users(users)
        return "<span class='text-success'>✔ Account created! You can now sign in.</span>"

    btn_reg.click(fn=process_register, inputs=[reg_user, reg_email, reg_pass, reg_conf], outputs=[reg_msg])

    # --- View Routing ---
    btn_go_reg.click(fn=lambda: (gr.Column(visible=False), gr.Column(visible=True)), outputs=[login_view, register_view])
    btn_go_log.click(fn=lambda: (gr.Column(visible=True), gr.Column(visible=False)), outputs=[login_view, register_view])
    btn_nav_dash.click(fn=lambda: (gr.Column(visible=True), gr.Column(visible=False)), outputs=[dashboard_view, feedback_view])
    btn_nav_feed.click(fn=lambda: (gr.Column(visible=False), gr.Column(visible=True)), outputs=[dashboard_view, feedback_view])
    
    def process_logout():
        return (
            gr.Column(visible=True),  # Show Login
            gr.Row(visible=False),    # Hide Main App
            None,                     # Clear State
            "", "", ""                # Clear Textboxes
        )
    btn_nav_logout.click(fn=process_logout, outputs=[login_view, main_app_view, current_user_state, log_user, log_pass, log_err])

    # --- Translation Logic (CLEANED OUTPUT FIX) ---
    def process_translation(upload_file, camera_file):
        yield "Running inference...", "⏳ Please wait while MediaPipe extracts keypoints..."
        
        # 1. Selection logic
        video_obj = camera_file if camera_file else upload_file
        
        if not video_obj:
            yield "Status: Error", "⚠ Please upload or record a video first."
            return
            
        # 2. Extract path
        if isinstance(video_obj, dict) and "path" in video_obj:
            video_path = video_obj["path"]
        elif isinstance(video_obj, str):
            video_path = video_obj
        else:
            video_path = getattr(video_obj, 'name', str(video_obj))
            
        # 3. Run Inference
        result_raw = run_inference(video_path)
        
        # 4. FIX: Filter out the MediaPipe dict output to show ONLY the sentence
        if isinstance(result_raw, dict):
            final_text = result_raw.get("sentence", "Translation complete (no sentence generated).")
        else:
            # If run_inference returned an error string instead of a dict
            final_text = str(result_raw)
            
        yield "✔ Translation complete.", final_text

    btn_translate.click(
        fn=process_translation,
        inputs=[upload_vid, camera_vid],
        outputs=[translate_status, output_text]
    )

    # --- Feedback Logic ---
    def process_feedback(user, rating, comment):
        if not rating:
            return "<span class='text-error'>⚠ Please select a star rating.</span>", rating, comment
        if not comment or not comment.strip():
            return "<span class='text-error'>⚠ Please write a comment before submitting.</span>", rating, comment
            
        entry = {
            "username":  user or "Anonymous",
            "rating":    int(rating.split(" ")[0]),
            "label":     rating.split(" - ")[1],
            "comment":   comment.strip(),
            "timestamp": datetime.datetime.now().isoformat(),
        }
        save_feedback(entry)
        return "<span class='text-success'>✔ Thank you — feedback submitted successfully!</span>", None, ""

    btn_submit_feed.click(
        fn=process_feedback, 
        inputs=[current_user_state, feed_rating, feed_comment], 
        outputs=[feed_msg, feed_rating, feed_comment]
    )


if __name__ == "__main__":
    os.makedirs(os.path.join(BASE_DIR, "uploads"), exist_ok=True)
    
    app.launch(
        server_name="127.0.0.1", 
        server_port=7860, 
        footer_links=["gradio"], 
        theme=custom_theme, 
        css=css
    )