"""
Sign Language Translator — Professional Flet UI
================================================
Sections
  • Login / Register  (with logo.png)
  • Sidebar  (Dashboard · Feedback · Logout)
  • Dashboard
      – Upload Video  → file picker → run inference
      – Open Camera   → live preview → record / stop → run inference
  • Feedback         → 5-star rating + text  → stored in feedback.json
  • Logout           → back to Login
"""

import os
import sys
import json
import threading
import datetime
import time
import tempfile
import socket
import inspect
import ctypes.util
import secrets

import flet as ft

# Flet compatibility aliases (old/new naming)
if not hasattr(ft, "Icons") and hasattr(ft, "icons"):
    ft.Icons = ft.icons
if not hasattr(ft, "Button") and hasattr(ft, "ElevatedButton"):
    ft.Button = ft.ElevatedButton

# ── make sure the project root is on the path so we can import inference.py ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ── paths ────────────────────────────────────────────────────────────────────
USERS_FILE    = os.path.join(BASE_DIR, "users.json")
FEEDBACK_FILE = os.path.join(BASE_DIR, "feedback.json")
LOGO_PATH     = os.path.join(BASE_DIR, "logo.png")
UPLOADS_DIR   = os.path.join(BASE_DIR, "uploads")

# ── colour palette ───────────────────────────────────────────────────────────
BG_DARK   = "#0d1117"
BG_CARD   = "#161b22"
BG_SIDE   = "#0d1117"
ACCENT    = "#2f81f7"
ACCENT2   = "#388bfd"
TEXT_PRI  = "#e6edf3"
TEXT_SEC  = "#8b949e"
BORDER    = "#30363d"
SUCCESS   = "#3fb950"
ERROR_CLR = "#f85149"
STAR_ON   = "#e3b341"
STAR_OFF  = "#30363d"

# ── Flet compatibility helpers (different enum names across versions) ───────
IMAGE_FIT_CONTAIN = (
    getattr(getattr(ft, "ImageFit", None), "CONTAIN", None)
    or getattr(getattr(ft, "BoxFit", None), "CONTAIN", None)
)
CLIP_HARD_EDGE = getattr(getattr(ft, "ClipBehavior", None), "HARD_EDGE", None)
MOUSE_CURSOR_CLICK = getattr(getattr(ft, "MouseCursor", None), "CLICK", None)
ALIGN_CENTER = (
    getattr(getattr(ft, "alignment", None), "center", None)
    or getattr(getattr(ft, "Alignment", None), "CENTER", None)
)


def pad_symmetric(horizontal=0, vertical=0):
    if hasattr(ft, "padding") and hasattr(ft.padding, "symmetric"):
        return ft.padding.symmetric(horizontal=horizontal, vertical=vertical)
    return ft.Padding(left=horizontal, right=horizontal, top=vertical, bottom=vertical)


def pad_only(left=0, top=0, right=0, bottom=0):
    if hasattr(ft, "padding") and hasattr(ft.padding, "only"):
        return ft.padding.only(left=left, top=top, right=right, bottom=bottom)
    return ft.Padding(left=left, top=top, right=right, bottom=bottom)


def border_radius_all(value=0):
    if hasattr(ft, "border_radius") and hasattr(ft.border_radius, "all"):
        return ft.border_radius.all(value)
    if hasattr(ft, "BorderRadius") and hasattr(ft.BorderRadius, "all"):
        return ft.BorderRadius.all(value)
    return value


def border_all(width, color):
    if hasattr(ft, "border") and hasattr(ft.border, "all"):
        return ft.border.all(width, color)
    return ft.Border.all(width, color)


def border_only(left=None, top=None, right=None, bottom=None):
    if hasattr(ft, "border") and hasattr(ft.border, "only"):
        return ft.border.only(left=left, top=top, right=right, bottom=bottom)
    return ft.Border.only(left=left, top=top, right=right, bottom=bottom)


# ═══════════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return []
                data = json.loads(content)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return []
                data = json.loads(content)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save_feedback(entry):
    data = load_feedback()
    data.append(entry)
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f, indent=2)


def run_inference(video_path: str) -> str:
    
    try:
        from step import title_to_sentence

        if not os.path.exists(video_path):
            return "⚠ Video file not found."

        text = title_to_sentence(video_path)
        return text.strip() if text.strip() else "(No Sign Detected)"
    except Exception as exc:
        return f"Error: {exc}"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main(page: ft.Page):
    page.title        = "Sign Language Translator"
    page.bgcolor      = BG_DARK
    page.theme_mode   = ft.ThemeMode.DARK
    page.padding      = 0
    page.window.width  = 1100
    page.window.height = 720

    # ── shared state ─────────────────────────────────────────────────────────
    current_user   = {"username": None}
    selected_video = {"path": None}        # path chosen by upload or camera
    camera_active  = {"value": False}
    recording      = {"value": False}
    cam_thread     = {"ref": None}
    video_writer   = {"ref": None}
    recorded_path  = {"path": None}
    ui_refs        = {"status_txt": None, "video_name": None}
    browser_upload = {"by_name": {}}
    writer_lock    = threading.Lock()

    # ── root container (swap views here) ─────────────────────────────────────
    root = ft.Column(expand=True, spacing=0)
    page.add(root)

    # ── file picker (registered once for the lifetime of the app) ────────────
    file_picker = ft.FilePicker()
    # Flet compatibility:
    # - newer builds: page.overlay.append(...)
    # - some builds: page.services.append(...)
    if hasattr(page, "overlay"):
        page.overlay.append(file_picker)
    elif hasattr(page, "services"):
        page.services.append(file_picker)
    else:
        raise RuntimeError("This Flet version does not support FilePicker registration on Page.")

    os.makedirs(UPLOADS_DIR, exist_ok=True)

    def start_browser_upload(file_name: str):
        """Upload file from browser picker into local UPLOADS_DIR."""
        try:
            safe_name = f"{int(time.time())}_{os.path.basename(file_name)}"
            upload_url = page.get_upload_url(safe_name, 600)
            local_path = os.path.join(UPLOADS_DIR, safe_name)
            browser_upload["by_name"][file_name] = local_path
            file_picker.upload([
                ft.FilePickerUploadFile(name=file_name, upload_url=upload_url)
            ])
        except Exception as exc:
            if ui_refs["status_txt"] is not None:
                ui_refs["status_txt"].value = f"⚠ Upload setup failed: {exc}"
            page.update()

    def apply_picked_files(files):
        if files:
            picked = files[0]
            path = getattr(picked, "path", None)
            name = getattr(picked, "name", None)

            # In WEB_BROWSER mode, many Flet builds return name but not local path.
            if path:
                selected_video["path"] = path
                display_name = os.path.basename(path)
                if ui_refs["video_name"] is not None:
                    ui_refs["video_name"].value = display_name
                if ui_refs["status_txt"] is not None:
                    ui_refs["status_txt"].value = f"✔ Video loaded: {display_name}"
            else:
                selected_video["path"] = None
                display_name = name or "Selected file"
                if ui_refs["video_name"] is not None:
                    ui_refs["video_name"].value = display_name
                if ui_refs["status_txt"] is not None:
                    ui_refs["status_txt"].value = "Uploading selected video…"

                if name:
                    start_browser_upload(name)
        else:
            selected_video["path"] = None
            if ui_refs["video_name"] is not None:
                ui_refs["video_name"].value = "No video selected"
            if ui_refs["status_txt"] is not None:
                ui_refs["status_txt"].value = "No file selected."
        page.update()

    def on_file_picker_result(e):
        # Older Flet versions provide results only via this callback.
        apply_picked_files(getattr(e, "files", None))

    def on_file_picker_upload(e):
        file_name = getattr(e, "file_name", None)
        progress = getattr(e, "progress", None)
        error = getattr(e, "error", None)

        if error:
            if ui_refs["status_txt"] is not None:
                ui_refs["status_txt"].value = f"⚠ Upload failed: {error}"
            page.update()
            return

        if progress is not None and progress < 1:
            if ui_refs["status_txt"] is not None:
                ui_refs["status_txt"].value = f"Uploading… {int(progress * 100)}%"
            page.update()
            return

        if file_name:
            local_path = browser_upload["by_name"].get(file_name)
            if local_path and os.path.exists(local_path):
                selected_video["path"] = local_path
                if ui_refs["video_name"] is not None:
                    ui_refs["video_name"].value = os.path.basename(local_path)
                if ui_refs["status_txt"] is not None:
                    ui_refs["status_txt"].value = "✔ Video uploaded and ready for translation"
                page.update()

    file_picker.on_result = on_file_picker_result
    file_picker.on_upload = on_file_picker_upload

    # ══════════════════════════════════════════════════════════════════════════
    # REUSABLE WIDGETS
    # ══════════════════════════════════════════════════════════════════════════

    def field(label, password=False, hint=""):
        return ft.TextField(
            label=label,
            hint_text=hint,
            password=password,
            can_reveal_password=password,
            bgcolor=BG_CARD,
            border_color=BORDER,
            focused_border_color=ACCENT,
            label_style=ft.TextStyle(color=TEXT_SEC),
            text_style=ft.TextStyle(color=TEXT_PRI),
            hint_style=ft.TextStyle(color=TEXT_SEC, size=12),
            border_radius=8,
            height=52,
        )

    def btn_primary(text, on_click, icon=None, width=None):
        return ft.Button(
            text=text,
            icon=icon,
            on_click=on_click,
            width=width,
            style=ft.ButtonStyle(
                bgcolor=ACCENT,
                color=TEXT_PRI,
                shape=ft.RoundedRectangleBorder(radius=8),
                padding=pad_symmetric(horizontal=24, vertical=14),
            ),
        )

    def btn_outline(text, on_click, icon=None, width=None):
        return ft.OutlinedButton(
            text=text,
            icon=icon,
            on_click=on_click,
            width=width,
            style=ft.ButtonStyle(
                side=ft.BorderSide(1, ACCENT),
                color=ACCENT,
                shape=ft.RoundedRectangleBorder(radius=8),
                padding=pad_symmetric(horizontal=24, vertical=14),
            ),
        )

    def logo_widget(size=64):
        if os.path.exists(LOGO_PATH):
            return ft.Image(
                src="logo.png",  # Must be purely relative to work over Flet's HTTP asset server
                width=size,
                height=size,
                fit=IMAGE_FIT_CONTAIN,
                border_radius=border_radius_all(12),
            )
        return ft.Icon(ft.Icons.SIGN_LANGUAGE, size=size, color=ACCENT)

    def card(content, padding=24, expand=None):
        return ft.Container(
            content=content,
            bgcolor=BG_CARD,
            border_radius=12,
            border=border_all(1, BORDER),
            padding=padding,
            expand=expand,
            shadow=ft.BoxShadow(
                spread_radius=1,
                blur_radius=15,
                color="#20000000",  # Hex string with alpha (AARRGGBB), 20 is ~12.5% opacity
                offset=ft.Offset(0, 4),
            )
        )

    def section_title(text):
        return ft.Text(text, size=20, weight=ft.FontWeight.W_700, color=TEXT_PRI)

    def sub_text(text):
        return ft.Text(text, size=13, color=TEXT_SEC)

    # ══════════════════════════════════════════════════════════════════════════
    # AUTH PAGES
    # ══════════════════════════════════════════════════════════════════════════

    def show_login():
        tf_user = field("Username", hint="Enter your username")
        tf_pass = field("Password", password=True, hint="Enter your password")
        err_txt = ft.Text("", color=ERROR_CLR, size=13)

        def do_login(e):
            u, p = tf_user.value.strip(), tf_pass.value.strip()
            if not u or not p:
                err_txt.value = "Please fill in all fields."
                page.update()
                return
            users = load_users()
            user = next((usr for usr in users if usr.get("username") == u), None)
            if user is None:
                err_txt.value = "User not found."
                page.update()
                return
            if user.get("password") != p:
                err_txt.value = "Invalid password."
                page.update()
                return

            current_user["username"] = u
            show_app()
            page.update()

        def go_register(e):
            show_register()

        auth_form = ft.Column(
            [
                ft.Row([logo_widget(56), ft.Column([
                    ft.Text("Sign Language", size=22, weight=ft.FontWeight.W_700, color=TEXT_PRI),
                    ft.Text("Translator", size=22, weight=ft.FontWeight.W_700, color=ACCENT),
                ], spacing=0)], alignment=ft.MainAxisAlignment.CENTER, spacing=16),
                ft.Divider(height=24, color="transparent"),
                section_title("Welcome back"),
                sub_text("Sign in to your account"),
                ft.Divider(height=16, color="transparent"),
                tf_user,
                ft.Divider(height=8, color="transparent"),
                tf_pass,
                ft.Divider(height=8, color="transparent"),
                err_txt,
                ft.Divider(height=4, color="transparent"),
                btn_primary("Sign In", do_login, icon=ft.Icons.LOGIN, width=340),
                ft.Divider(height=12, color="transparent"),
                ft.Row([
                    sub_text("Don't have an account?"),
                    ft.TextButton(content=ft.Text("Register", color=ACCENT),
                                  on_click=go_register,
                                  style=ft.ButtonStyle(color=ACCENT)),
                ], alignment=ft.MainAxisAlignment.CENTER),
            ],
            width=340,
            spacing=4,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )

        root.controls = [
            ft.Row(
                [
                    # left branding panel
                    ft.Container(
                        content=ft.Column(
                            [
                                logo_widget(90),
                                ft.Divider(height=20, color="transparent"),
                                ft.Text("Sign Language\nTranslator", size=28,
                                        weight=ft.FontWeight.W_700, color=TEXT_PRI,
                                        text_align=ft.TextAlign.CENTER),
                                ft.Divider(height=12, color="transparent"),
                                ft.Text(
                                    "Real-time AI-powered translation\nof sign language to text.",
                                    size=14, color=TEXT_SEC,
                                    text_align=ft.TextAlign.CENTER,
                                ),
                            ],
                            alignment=ft.MainAxisAlignment.CENTER,
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        bgcolor=BG_CARD,
                        border=border_only(right=ft.BorderSide(1, BORDER)),
                        expand=2,
                        padding=40,
                    ),
                    # right form panel
                    ft.Container(
                        content=ft.Column(
                            [auth_form],
                            alignment=ft.MainAxisAlignment.CENTER,
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        expand=3,
                        padding=40,
                    ),
                ],
                expand=True,
                spacing=0,
            )
        ]
        page.update()

    def show_register():
        tf_user  = field("Username", hint="Choose a username")
        tf_email = field("Email", hint="your@email.com")
        tf_pass  = field("Password", password=True, hint="Create a password")
        tf_conf  = field("Confirm Password", password=True, hint="Repeat password")
        err_txt  = ft.Text("", color=ERROR_CLR, size=13)
        ok_txt   = ft.Text("", color=SUCCESS, size=13)

        def do_register(e):
            u  = tf_user.value.strip()
            em = tf_email.value.strip()
            p  = tf_pass.value.strip()
            c  = tf_conf.value.strip()
            err_txt.value = ""
            ok_txt.value  = ""
            if not u or not em or not p or not c:
                err_txt.value = "All fields are required."
                page.update(); return
            if p != c:
                err_txt.value = "Passwords do not match."
                page.update(); return
            users = load_users()
            for usr in users:
                if usr["username"] == u:
                    err_txt.value = "Username already taken."
                    page.update(); return
            users.append({
                "username": u,
                "email": em,
                "password": p,
                "created": datetime.datetime.now().isoformat(),
            })
            save_users(users)
            ok_txt.value = "✔ Account created! Redirecting…"
            page.update()
            time.sleep(1.2)
            show_login()

        def go_login(e):
            show_login()

        reg_form = ft.Column(
            [
                ft.Row([logo_widget(48), ft.Column([
                    ft.Text("Create Account", size=20,
                            weight=ft.FontWeight.W_700, color=TEXT_PRI),
                    sub_text("Join Sign Language Translator"),
                ], spacing=2)], alignment=ft.MainAxisAlignment.CENTER, spacing=14),
                ft.Divider(height=20, color="transparent"),
                tf_user,
                ft.Divider(height=8, color="transparent"),
                tf_email,
                ft.Divider(height=8, color="transparent"),
                tf_pass,
                ft.Divider(height=8, color="transparent"),
                tf_conf,
                ft.Divider(height=6, color="transparent"),
                err_txt, ok_txt,
                ft.Divider(height=4, color="transparent"),
                btn_primary("Create Account", do_register,
                            icon=ft.Icons.PERSON_ADD, width=340),
                ft.Divider(height=10, color="transparent"),
                ft.Row([
                    sub_text("Already have an account?"),
                    ft.TextButton(content=ft.Text("Sign In", color=ACCENT),
                                  on_click=go_login,
                                  style=ft.ButtonStyle(color=ACCENT)),
                ], alignment=ft.MainAxisAlignment.CENTER),
            ],
            width=340,
            spacing=4,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )

        root.controls = [
            ft.Row(
                [
                    ft.Container(
                        content=ft.Column(
                            [
                                logo_widget(90),
                                ft.Divider(height=20, color="transparent"),
                                ft.Text("Sign Language\nTranslator", size=28,
                                        weight=ft.FontWeight.W_700, color=TEXT_PRI,
                                        text_align=ft.TextAlign.CENTER),
                                ft.Divider(height=12, color="transparent"),
                                ft.Text(
                                    "Create your account to start\ntranslating sign language.",
                                    size=14, color=TEXT_SEC,
                                    text_align=ft.TextAlign.CENTER,
                                ),
                            ],
                            alignment=ft.MainAxisAlignment.CENTER,
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        bgcolor=BG_CARD,
                        border=border_only(right=ft.BorderSide(1, BORDER)),
                        expand=2,
                        padding=40,
                    ),
                    ft.Container(
                        content=ft.Column(
                            [reg_form],
                            alignment=ft.MainAxisAlignment.CENTER,
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        expand=3,
                        padding=40,
                    ),
                ],
                expand=True,
                spacing=0,
            )
        ]
        page.update()

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN APP (sidebar + content area)
    # ══════════════════════════════════════════════════════════════════════════

    # content_area holds the current page inside the app
    content_area = ft.Column(expand=True, scroll=ft.ScrollMode.AUTO, spacing=0)

    # active nav item state
    nav_state = {"active": "dashboard"}

    # sidebar nav items
    sidebar_items = []

    def make_nav_item(label, icon, route):
        is_active = nav_state["active"] == route

        def on_click(e):
            nav_state["active"] = route
            refresh_sidebar()
            if route == "dashboard":
                show_dashboard()
            elif route == "feedback":
                show_feedback()
            elif route == "logout":
                do_logout()
            page.update()

        return ft.Container(
            content=ft.Row(
                [
                    ft.Icon(icon, size=20,
                            color=ACCENT if is_active else TEXT_SEC),
                    ft.Text(label, size=14,
                            weight=ft.FontWeight.W_600 if is_active else ft.FontWeight.W_400,
                            color=ACCENT if is_active else TEXT_SEC),
                ],
                spacing=12,
            ),
            on_click=on_click,
            border_radius=8,
                 padding=pad_symmetric(horizontal=16, vertical=12),
            bgcolor=f"{ACCENT}22" if is_active else "transparent",
                 border=border_only(left=ft.BorderSide(3, ACCENT))
                     if is_active else border_only(left=ft.BorderSide(3, "transparent")),
            animate=ft.Animation(200, ft.AnimationCurve.EASE_IN_OUT),
        )

    def build_sidebar():
        return ft.Container(
            content=ft.Column(
                [
                    # logo + app name
                    ft.Container(
                        content=ft.Column(
                            [
                                logo_widget(48),
                                ft.Divider(height=8, color="transparent"),
                                ft.Text("Sign Language", size=13,
                                        weight=ft.FontWeight.W_700, color=TEXT_PRI),
                                ft.Text("Translator", size=13,
                                        weight=ft.FontWeight.W_700, color=ACCENT),
                            ],
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                            spacing=2,
                        ),
                        padding=pad_symmetric(vertical=24),
                    ),
                    ft.Divider(height=1, color=BORDER),
                    ft.Divider(height=8, color="transparent"),
                    # user badge
                    ft.Container(
                        content=ft.Row(
                            [
                                ft.Container(
                                    content=ft.Text(
                                        (current_user["username"] or "U")[0].upper(),
                                        size=14, weight=ft.FontWeight.W_700, color=TEXT_PRI,
                                    ),
                                    bgcolor=ACCENT,
                                    width=34, height=34,
                                    border_radius=17,
                                    alignment=ALIGN_CENTER,
                                ),
                                ft.Column(
                                    [
                                        ft.Text(current_user["username"] or "",
                                                size=13, weight=ft.FontWeight.W_600,
                                                color=TEXT_PRI),
                                        ft.Text("Signed in", size=11, color=TEXT_SEC),
                                    ],
                                    spacing=1,
                                ),
                            ],
                            spacing=10,
                        ),
                        padding=pad_symmetric(horizontal=16, vertical=8),
                    ),
                    ft.Divider(height=12, color="transparent"),
                    ft.Text("  NAVIGATION", size=10, color=TEXT_SEC,
                            weight=ft.FontWeight.W_600),
                    ft.Divider(height=6, color="transparent"),
                    make_nav_item("Dashboard",  ft.Icons.DASHBOARD_OUTLINED,  "dashboard"),
                    make_nav_item("Feedback",   ft.Icons.STAR_RATE_OUTLINED,  "feedback"),
                    ft.Divider(height=1, color=BORDER),
                    ft.Divider(height=6, color="transparent"),
                    make_nav_item("Logout",     ft.Icons.LOGOUT_ROUNDED,      "logout"),
                ],
                spacing=4,
                expand=True,
            ),
            bgcolor=BG_SIDE,
            width=220,
            border=border_only(right=ft.BorderSide(1, BORDER)),
            padding=pad_only(bottom=16),
        )

    sidebar_ref = ft.Ref[ft.Container]()

    def refresh_sidebar():
        app_layout.controls[0] = build_sidebar()
        page.update()

    def do_logout():
        camera_active["value"] = False
        current_user["username"] = None
        selected_video["path"]   = None
        show_login()

    # ── main layout (sidebar + content) ──────────────────────────────────────
    app_layout = ft.Row(
        [build_sidebar(), content_area],
        expand=True,
        spacing=0,
    )

    def show_app():
        nav_state["active"] = "dashboard"
        root.controls = [app_layout]
        app_layout.controls[0] = build_sidebar()
        page.update()
        show_dashboard()

    # ══════════════════════════════════════════════════════════════════════════
    # DASHBOARD
    # ══════════════════════════════════════════════════════════════════════════

    def show_dashboard():
        nav_state["active"] = "dashboard"
        app_layout.controls[0] = build_sidebar()

        status_txt = ft.Text("", color=TEXT_SEC, size=13)
        video_name = ft.Text("No video selected", color=TEXT_SEC, size=13)
        ui_refs["status_txt"] = status_txt
        ui_refs["video_name"] = video_name
        output_field = ft.TextField(
            label="Translation Output",
            multiline=True,
            min_lines=4,
            read_only=True,
            bgcolor=BG_DARK,
            border_color=BORDER,
            focused_border_color=ACCENT,
            label_style=ft.TextStyle(color=TEXT_SEC),
            text_style=ft.TextStyle(color=SUCCESS, size=15,
                                    weight=ft.FontWeight.W_500),
            border_radius=8,
        )
        spinner = ft.ProgressRing(width=22, height=22, stroke_width=2.5,
                                  color=ACCENT, visible=False)

        # ── camera preview image ──────────────────────────────────────────────
        cam_image    = ft.Image(src="", width=460, height=300, fit=IMAGE_FIT_CONTAIN,
                                border_radius=10, visible=False, gapless_playback=True)
        cam_status   = ft.Text("", color=TEXT_SEC, size=12)
        rec_dot      = ft.Container(width=10, height=10, border_radius=5,
                                    bgcolor=ERROR_CLR, visible=False)
        rec_label    = ft.Text("", color=ERROR_CLR, size=12, visible=False,
                               weight=ft.FontWeight.W_600)
        frame_size   = {"w": 640, "h": 480}   # updated from first camera frame

        # ── file picker ───────────────────────────────────────────────────────
        async def pick_file(e):
            ret = file_picker.pick_files(
                dialog_title="Select a video file",
                allowed_extensions=["mp4", "avi", "mov", "mkv", "webm"],
                allow_multiple=False,
            )
            # Newer Flet may return awaitable result; older returns None and
            # triggers file_picker.on_result callback instead.
            if inspect.isawaitable(ret):
                files = await ret
                apply_picked_files(files)

        # ── camera ───────────────────────────────────────────────────────────
        def start_camera(e):
            import cv2, base64
            # Silence verbose OpenCV camera probe messages (constant varies by build)
            try:
                cv2.setLogLevel(3)   # 3 = LOG_LEVEL_WARNING — suppresses noisy INFO logs
            except Exception:
                pass
            camera_active["value"] = True
            recording["value"]     = False
            cam_image.visible      = True
            cam_status.value       = "Opening camera…"
            rec_btn.visible        = True
            stop_btn.visible       = False
            page.update()

            def cam_loop():
                # Try V4L2 backend first (Linux), then fall back to auto-detect
                cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    # Fallback to defaults to prevent VideoWriter from crashing later
                    frame_size["w"], frame_size["h"] = 640, 480
                    camera_active["value"] = False
                    cam_image.visible      = False
                    cam_status.value       = "⚠ No camera found. Please connect a webcam."
                    rec_btn.visible        = False
                    page.update()
                    return
                
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cam_status.value = "Camera active — press Record to capture"
                page.update()
                first_frame = True
                while camera_active["value"]:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Detect actual frame dimensions from the first frame
                    if first_frame:
                        frame_size["h"], frame_size["w"] = frame.shape[:2]
                        first_frame = False
                        
                    # Copy frame to avoid memory race condition with ffmpeg internal thread
                    display_frame = frame.copy()
                    
                    # Write every frame to the video file while recording (this can be async buffered in cv2)
                    if recording["value"]:
                        with writer_lock:
                            vw = video_writer["ref"]
                        if vw is not None:
                            vw.write(frame)
                        
                    # Encode for live preview display
                    _, buf = cv2.imencode(".jpg", display_frame,
                                         [cv2.IMWRITE_JPEG_QUALITY, 60])
                    b64 = base64.b64encode(buf).decode()
                    cam_image.src = f"data:image/jpeg;base64,{b64}"
                    try:
                        # Direct image update is highly preferred, fallback to page.update if it raises
                        cam_image.update()
                    except Exception:
                        pass
                    # No time.sleep() needed; cap.read() automatically blocks to hardware FPS (e.g. 30fps)

                cap.release()
                with writer_lock:
                    if video_writer["ref"]:
                        video_writer["ref"].release()
                        video_writer["ref"] = None
                cam_image.visible = False
                cam_status.value  = ""
                page.update()

            cam_thread["ref"] = threading.Thread(target=cam_loop, daemon=True)
            cam_thread["ref"].start()

        def start_recording(e):
            import cv2
            os.makedirs(UPLOADS_DIR, exist_ok=True)
            rec_name = f"camera_recording_{int(time.time())}.mp4"
            rec_path = os.path.join(UPLOADS_DIR, rec_name)
            recorded_path["path"] = rec_path
            
            # Avoid MJPG PTS issues; use MP4V first, then fallback to XVID.
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(
                rec_path, fourcc, 25.0,
                (frame_size["w"], frame_size["h"]),
            )
            if not vw.isOpened():
                rec_name = f"camera_recording_{int(time.time())}.avi"
                rec_path = os.path.join(UPLOADS_DIR, rec_name)
                recorded_path["path"] = rec_path
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                vw = cv2.VideoWriter(
                    rec_path, fourcc, 25.0,
                    (frame_size["w"], frame_size["h"]),
                )

            with writer_lock:
                # release any stale writer before replacing
                if video_writer["ref"] is not None:
                    video_writer["ref"].release()
                video_writer["ref"] = vw

            recording["value"] = True
            rec_dot.visible   = True
            rec_label.value   = "● REC"
            rec_label.visible = True
            rec_btn.visible   = False
            stop_btn.visible  = True
            cam_status.value  = "Recording…"
            page.update()

        def stop_recording(e):
            recording["value"] = False
            # Let camera thread own final writer release to avoid race/segfault.
            path = recorded_path["path"]
            camera_active["value"] = False
            if cam_thread["ref"] and cam_thread["ref"].is_alive():
                cam_thread["ref"].join(timeout=1.5)

            if path and os.path.exists(path) and os.path.getsize(path) > 0:
                selected_video["path"] = path
                video_name.value       = os.path.basename(path)
                status_txt.value       = "✔ Recorded video ready for translation"
            else:
                status_txt.value       = "⚠ Recording failed. Please try again."
            rec_dot.visible   = False
            rec_label.visible = False
            stop_btn.visible  = False
            rec_btn.visible   = True
            cam_status.value  = "Recording saved. Camera stopped."
            page.update()

        rec_btn  = btn_primary("Record",       start_recording, ft.Icons.FIBER_MANUAL_RECORD)
        stop_btn = btn_outline("Stop",         stop_recording,  ft.Icons.STOP_CIRCLE_OUTLINED)
        stop_btn.visible = False
        rec_btn.visible  = False

        # ── translate ─────────────────────────────────────────────────────────
        def do_translate(e):
            path = selected_video["path"]
            if not path or not os.path.exists(path):
                output_field.value = "⚠ Please upload or record a video first."
                page.update()
                return
            print("Translating...")
            spinner.visible      = True
            output_field.value   = ""
            status_txt.value     = "Translating..."
            page.update()

            def infer():
                result = run_inference(path)
                output_field.value = result
                spinner.visible    = False
                status_txt.value   = "Translation complete."
                print("Done")
                page.update()

            threading.Thread(target=infer, daemon=True).start()

        # ── layout ────────────────────────────────────────────────────────────
        content_area.controls = [
            ft.Container(
                content=ft.Column(
                    [
                        # header
                        ft.Row([
                            ft.Icon(ft.Icons.DASHBOARD_OUTLINED, color=ACCENT, size=24),
                            ft.Column([
                                section_title("Dashboard"),
                                sub_text("Upload a video or record from camera, then translate."),
                            ], spacing=2),
                        ], spacing=12),
                        ft.Divider(height=20, color=BORDER),

                        # two cards side-by-side
                        ft.Row(
                            [
                                # ── Upload card ───────────────────────────────
                                card(ft.Column(
                                    [
                                        ft.Row([
                                            ft.Icon(ft.Icons.UPLOAD_FILE_ROUNDED,
                                                    color=ACCENT, size=22),
                                            ft.Text("Upload Video", size=16,
                                                    weight=ft.FontWeight.W_600,
                                                    color=TEXT_PRI),
                                        ], spacing=10),
                                        ft.Divider(height=12, color="transparent"),
                                        ft.Container(
                                            content=ft.Column(
                                                [
                                                    ft.Icon(ft.Icons.CLOUD_UPLOAD_OUTLINED,
                                                            size=46, color=TEXT_SEC),
                                                    ft.Divider(height=8, color="transparent"),
                                                    sub_text("MP4 · AVI · MOV · MKV"),
                                                ],
                                                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                            ),
                                            bgcolor=BG_DARK,
                                            border=border_all(1, BORDER),
                                            border_radius=8,
                                            padding=pad_symmetric(vertical=32),
                                            alignment=ALIGN_CENTER,
                                        ),
                                        ft.Divider(height=12, color="transparent"),
                                        btn_primary("Browse File", pick_file,
                                                    ft.Icons.FOLDER_OPEN_OUTLINED),
                                        ft.Divider(height=8, color="transparent"),
                                        ft.Row([
                                            ft.Icon(ft.Icons.MOVIE_OUTLINED,
                                                    size=16, color=TEXT_SEC),
                                            ft.Container(content=video_name, expand=True, clip_behavior=CLIP_HARD_EDGE),
                                        ], spacing=6),
                                    ],
                                    spacing=0,
                                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                ), padding=24, expand=1),

                                ft.Container(width=24),

                                # ── Camera card ───────────────────────────────
                                card(ft.Column(
                                    [
                                        ft.Row([
                                            ft.Icon(ft.Icons.VIDEOCAM_OUTLINED,
                                                    color=ACCENT, size=22),
                                            ft.Text("Open Camera", size=16,
                                                    weight=ft.FontWeight.W_600,
                                                    color=TEXT_PRI),
                                        ], spacing=10),
                                        ft.Divider(height=12, color="transparent"),
                                        ft.Container(
                                            content=cam_image,
                                            alignment=ALIGN_CENTER,
                                            bgcolor=BG_DARK,
                                            border_radius=8,
                                            padding=8,
                                        ),
                                        ft.Divider(height=8, color="transparent"),
                                        ft.Row([rec_dot, rec_label], spacing=6, alignment=ft.MainAxisAlignment.CENTER),
                                        ft.Divider(height=4, color="transparent"),
                                        ft.Row([
                                            btn_primary("Open Camera", start_camera,
                                                        ft.Icons.CAMERA_ALT_OUTLINED),
                                            rec_btn,
                                            stop_btn,
                                        ], spacing=10, wrap=True, alignment=ft.MainAxisAlignment.CENTER),
                                        ft.Divider(height=6, color="transparent"),
                                        ft.Container(content=cam_status, alignment=ALIGN_CENTER),
                                    ],
                                    spacing=0,
                                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                ), padding=24, expand=1),
                            ],
                            alignment=ft.MainAxisAlignment.START,
                            vertical_alignment=ft.CrossAxisAlignment.START,
                            expand=False,
                            wrap=False,
                            spacing=0,
                        ),

                        ft.Divider(height=24, color=BORDER),

                        # ── Translate section ─────────────────────────────────
                        card(ft.Column(
                            [
                                ft.Row([
                                    ft.Icon(ft.Icons.TRANSLATE_ROUNDED,
                                            color=ACCENT, size=22),
                                    ft.Text("Translate", size=16,
                                            weight=ft.FontWeight.W_600, color=TEXT_PRI),
                                    spinner,
                                ], spacing=10),
                                ft.Divider(height=12, color="transparent"),
                                status_txt,
                                ft.Divider(height=8, color="transparent"),
                                btn_primary("Translate", do_translate,
                                            ft.Icons.PLAY_ARROW_ROUNDED, width=180),
                                ft.Divider(height=16, color="transparent"),
                                output_field,
                            ],
                            spacing=0,
                        ), padding=24),
                    ],
                    spacing=0,
                    expand=True,
                ),
                expand=True,
                padding=32,
            )
        ]
        page.update()

    # ══════════════════════════════════════════════════════════════════════════
    # FEEDBACK
    # ══════════════════════════════════════════════════════════════════════════

    def show_feedback():
        nav_state["active"] = "feedback"
        app_layout.controls[0] = build_sidebar()

        star_state = {"value": 0, "hover": 0}
        star_icons = []
        err_txt = ft.Text("", color=ERROR_CLR, size=13)
        ok_txt  = ft.Text("", color=SUCCESS, size=13)
        rating_label = ft.Text("No rating selected", color=TEXT_SEC, size=13)

        STAR_LABELS = ["", "Poor", "Fair", "Good", "Very Good", "Excellent"]

        def render_stars():
            active = star_state["value"]
            for i, ic in enumerate(star_icons):
                ic.name  = ft.Icons.STAR_ROUNDED if i < active else ft.Icons.STAR_BORDER_ROUNDED
                ic.color = STAR_ON if i < active else STAR_OFF
            rating_label.value = (STAR_LABELS[star_state["value"]]
                                  if star_state["value"] else "No rating selected")
            rating_label.color = STAR_ON if star_state["value"] else TEXT_SEC
            page.update()

        def make_star(idx):
            ic = ft.Icon(ft.Icons.STAR_BORDER_ROUNDED, size=38, color=STAR_OFF)
            star_icons.append(ic)

            def on_click(e):
                star_state["value"] = idx + 1
                render_stars()

            return ft.GestureDetector(
                content=ft.Container(content=ic, padding=4),
                on_tap=on_click,
                mouse_cursor=MOUSE_CURSOR_CLICK,
            )

        stars_row = ft.Row(
            [make_star(i) for i in range(5)],
            spacing=4,
        )

        tf_comment = ft.TextField(
            label="Your feedback",
            hint_text="Share your experience with the translator…",
            multiline=True,
            min_lines=4,
            max_lines=8,
            bgcolor=BG_DARK,
            border_color=BORDER,
            focused_border_color=ACCENT,
            label_style=ft.TextStyle(color=TEXT_SEC),
            text_style=ft.TextStyle(color=TEXT_PRI),
            hint_style=ft.TextStyle(color=TEXT_SEC, size=12),
            border_radius=8,
        )

        def submit_feedback(e):
            err_txt.value = ""
            ok_txt.value  = ""
            if star_state["value"] == 0:
                err_txt.value = "Please select a star rating."
                page.update(); return
            comment = tf_comment.value.strip()
            if not comment:
                err_txt.value = "Please write a comment before submitting."
                page.update(); return
            entry = {
                "username":  current_user["username"],
                "rating":    star_state["value"],
                "label":     STAR_LABELS[star_state["value"]],
                "comment":   comment,
                "timestamp": datetime.datetime.now().isoformat(),
            }
            save_feedback(entry)
            ok_txt.value       = "✔ Thank you — feedback submitted successfully!"
            tf_comment.value   = ""
            star_state["value"] = 0
            star_state["hover"] = 0
            render_stars()
            page.update()

        content_area.controls = [
            ft.Container(
                content=ft.Column(
                    [
                        ft.Row([
                            ft.Icon(ft.Icons.STAR_RATE_OUTLINED, color=ACCENT, size=24),
                            ft.Column([
                                section_title("Feedback"),
                                sub_text("Rate your experience and share your thoughts."),
                            ], spacing=2),
                        ], spacing=12),
                        ft.Divider(height=20, color=BORDER),

                        card(ft.Column(
                            [
                                ft.Text("Rate the Translation Quality",
                                        size=15, weight=ft.FontWeight.W_600, color=TEXT_PRI),
                                ft.Divider(height=10, color="transparent"),
                                stars_row,
                                rating_label,
                                ft.Divider(height=20, color="transparent"),

                                ft.Text("Additional Comments",
                                        size=15, weight=ft.FontWeight.W_600, color=TEXT_PRI),
                                ft.Divider(height=10, color="transparent"),
                                tf_comment,
                                ft.Divider(height=14, color="transparent"),
                                err_txt, ok_txt,
                                ft.Divider(height=4, color="transparent"),
                                btn_primary("Submit Feedback", submit_feedback,
                                            ft.Icons.SEND_ROUNDED, width=200),
                            ],
                            spacing=4,
                        ), padding=28),
                    ],
                    spacing=0,
                    expand=True,
                ),
                expand=True,
                padding=32,
            )
        ]
        page.update()

    # ── kick off ──────────────────────────────────────────────────────────────
    show_login()


if __name__ == "__main__":
    # Required for browser uploads in this Flet runtime.
    os.environ.setdefault("FLET_SECRET_KEY", secrets.token_hex(32))

    # Pick a free port to avoid "address already in use" startup errors.
    requested_port = int(os.getenv("FLET_PORT", "8000"))

    def is_port_free(port: int) -> bool:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s6:
            s6.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s6.bind(("::", port, 0, 0))
                return True
            except OSError:
                pass
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s4:
            s4.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s4.bind(("0.0.0.0", port))
                return True
            except OSError:
                return False

    run_port = requested_port
    if not is_port_free(run_port):
        for p in range(requested_port + 1, requested_port + 21):
            if is_port_free(p):
                run_port = p
                break

    # Desktop view is the most reliable for FilePicker local paths.
    # Override with: FLET_VIEW=web to force browser mode.
    view_env = os.getenv("FLET_VIEW", "desktop").strip().lower()
    app_view = ft.AppView.FLET_APP
    if view_env in {"web", "browser", "web_browser"}:
        app_view = ft.AppView.WEB_BROWSER

    # Linux desktop Flet runtime needs libmpv (libmpv.so.1).
    # If missing, auto-fallback to browser mode to keep app usable.
    if app_view == ft.AppView.FLET_APP and ctypes.util.find_library("mpv") is None:
        app_view = ft.AppView.WEB_BROWSER

    ft.app(main, view=app_view, port=run_port, assets_dir=BASE_DIR, upload_dir=UPLOADS_DIR)
