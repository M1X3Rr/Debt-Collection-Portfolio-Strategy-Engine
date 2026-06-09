# Main user interface for the debt collection strategy engine
#
# This GUI uses CustomTkinter for a modern dark UI and provides:
# - Authentication (admin vs user / skip mode)
# - Prediction (strategy engine) tab with portfolio upload & effort scenarios
# - Training tab with GBM training progress (admin only; no per-epoch NN loss)
# - Testing tab for model evaluation (admin only)
# - Training data insights and suggested actions display
# - Client filtering for simulation results

## ---- Technical issues and improvements START ----

    # []
    # []
    # []
    # []
    # []
    # []
    # []
    # []

## ---- Technical issues and improvements END ----


import json
import math
import os
import sys
import tempfile
import threading
import markdown2 as md
from datetime import datetime
from typing import Dict, List, Literal, Optional, Tuple

import customtkinter as ctk
import joblib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from tkinter import filedialog, messagebox
import tkinter as tk

from tkhtmlview import HTMLScrolledText as HtmlScrolledText

from portfolio_charts import (
    compute_monthly_breakdown,
    format_portfolio_length_for_ui,
    list_portfolio_length_clients,
    parse_portfolio_length_input,
    portfolio_lengths_config,
    resolve_payout_months_for_segment,
)
from predict import (
    debtor_row_to_features,
    get_training_data_insights,
    load_artifacts_for_inference,
    predict_paid_values_for_dataframe,
    recommend_portfolio_strategy,
)
from prediction_cap import USE_PAID_PREDICTION_CAP, is_prediction_cap_active
from split import (
    basic_cleaning,
    case_duration_weeks_series,
    compute_weekly_rates,
    fit_preprocessors,
    load_config,
    load_raw_data,
    LogStandardScaler,  # noqa: F401 - needed for joblib unpickling
    prepare_splits,
    save_splits_and_artifacts,
    split_dataset,
    stream_split_excel,
)
from gbm_inference import load_paid_bundle
from train import train_all_gbm

USERS_FILE = "users.json"


def read_table_file(path: str) -> pd.DataFrame:
    """
    Load a tabular file that may be CSV or Excel.

    Parameters
    ----------
    path : str
        Path to the file. Supports `.csv`, `.xlsx`, `.xls`.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    lower = path.lower()
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    
    if lower.endswith(".csv"):
        # Try UTF-8 first, fallback to cp1252 if it fails
        try:
            return pd.read_csv(path, encoding="utf-8")
        except (UnicodeDecodeError, UnicodeError):
            return pd.read_csv(path, encoding="cp1252")
            
    raise ValueError("Unsupported file format. Please use .csv, .xlsx, or .xls.")


def load_users() -> List[Dict]:
    """
    Load user credentials from users.json.

    #NOTE: Expected schema:
    # [
    #   {"username": "admin", "password": "********", "role": "admin"},
    #   {"username": "user", "password": "********", "role": "user"}
    # ]
    #

    Returns
    -------
    List[Dict]
        List of user records.
    """
    if not os.path.exists(USERS_FILE):
        # Create a default users file with a single admin account.
        default_users = [
            {"username": "admin", "password": "change-me", "role": "admin"},
            {"username": "user", "password": "change-me", "role": "user"},
        ]
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(default_users, f, indent=2)
        return default_users

    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            data = f.read().strip()
            if not data:
                return []
            return json.loads(data)
    except Exception:
        return []


def authenticate(username: str, password: str) -> Optional[str]:
    """
    Authenticate a user and return the associated role if valid.

    Parameters
    ----------
    username : str
        Username from the login form.
    password : str
        Password from the login form.

    Returns
    -------
    Optional[str]
        "admin" / "user" if credentials are valid, otherwise None.
    """
    users = load_users()
    for user in users:
        if (
            user.get("username") == username
            and user.get("password") == password
        ):
            return user.get("role", "user")
    return None


def _weekly_cap_as_case_total(weekly_cap: float, avg_weeks_open: float) -> int:
    """Convert a per-week cap into a max total actions per case."""
    return int(math.ceil(weekly_cap * max(float(avg_weeks_open), 1.0 / 7.0)))


def compute_effort_levels(
    config: dict,
    baseline_actions: Dict[str, float],
    percentage_change: float,
    *,
    baseline_is_weekly: bool = False,
    avg_weeks_open: float = 1.0,
) -> Dict[str, Dict[str, int]]:
    """
    Define Low / Medium / High effort action combinations based on the baseline.

    The baseline is derived from portfolio/training averages or user input.
    The effort levels are computed by applying a percentage increase/decrease
    around that baseline.

    - Low = baseline * (1 - percentage_change)
    - Medium = baseline
    - High = baseline * (1 + percentage_change)

    When baseline_is_weekly is True, baseline_actions are per-week rates and
    returned integers are total actions per case (weekly rate * avg_weeks_open).

    Values are rounded to integers and clamped to >= 0.
    The grid is used only as a soft reference for minimum differentiation.
    """
    grid = config.get("actions_grid", {})
    weekly_caps: Dict[str, float] = config.get("weekly_action_caps", {})
    action_features = config["columns"].get("action_features", [])
    pct = abs(float(percentage_change)) if percentage_change != 0 else 0.2
    avg_weeks = max(float(avg_weeks_open), 1.0 / 7.0)

    low: Dict[str, int] = {}
    medium: Dict[str, int] = {}
    high: Dict[str, int] = {}

    for action in action_features:
        base = float(baseline_actions.get(action, 0.0))
        cap = float(weekly_caps.get(action, float("inf")))
        cap_total: Optional[int] = None
        if cap > 0 and math.isfinite(cap):
            cap_total = (
                _weekly_cap_as_case_total(cap, avg_weeks)
                if baseline_is_weekly
                else int(cap)
            )

        if baseline_is_weekly:
            weekly_rate = base
            if cap > 0:
                weekly_rate = min(weekly_rate, cap)
            if weekly_rate > 0 and pct > 0:
                delta_weekly = max(weekly_rate * pct, 1.0 / avg_weeks)
            else:
                delta_weekly = 0.0
            med_weekly = weekly_rate
            low_weekly = max(0.0, weekly_rate - delta_weekly)
            high_weekly = weekly_rate + delta_weekly
            if cap > 0:
                low_weekly = min(low_weekly, cap)
                med_weekly = min(med_weekly, cap)
                high_weekly = min(high_weekly, cap)
            low_val = int(math.ceil(low_weekly * avg_weeks - 1e-9))
            med_val = int(math.ceil(med_weekly * avg_weeks - 1e-9))
            high_val = int(math.ceil(high_weekly * avg_weeks - 1e-9))
        else:
            base_int = int(round(base))
            if cap_total is not None:
                base_int = min(base_int, cap_total)
            if base_int > 0 and pct > 0:
                delta = max(1, int(round(base_int * pct)))
            else:
                delta = 0

            med_val = max(0, base_int)
            low_val = max(0, base_int - delta)
            high_val = base_int + delta

            if cap_total is not None:
                low_val = min(low_val, cap_total)
                med_val = min(med_val, cap_total)
                high_val = min(high_val, cap_total)

        # Ensure minimum differentiation if all values collapse (e.g. very
        # small baselines or tight caps). Use the actions_grid as a soft
        # guide for steps.
        if low_val == med_val == high_val:
            grid_values = sorted(grid.get(action, [0, 1, 2, 3, 4, 5]))
            if len(grid_values) >= 2:
                step = max(1, grid_values[1] - grid_values[0])
            else:
                step = 1

            if med_val > 0:
                low_val = max(0, med_val - step)
            high_val = med_val + step

            if cap_total is not None:
                low_val = min(low_val, cap_total)
                high_val = min(high_val, cap_total)

        if cap_total is not None:
            low_val = min(low_val, cap_total)
            med_val = min(med_val, cap_total)
            high_val = min(high_val, cap_total)

        # Enforce order low <= medium <= high (avoid rounding/capping drift).
        low_val = max(0, min(low_val, med_val))
        high_val = max(med_val, high_val)

        # Store integer action counts.
        low[action] = int(low_val)
        medium[action] = int(med_val)
        high[action] = int(high_val)

    # Ensure that for each effort level at least one of the \"active\"
    # contact channels (Calls, SMS, Emails) is non-zero. Letters are
    # allowed to be zero even when others are negative.
    contact_actions = [a for a in action_features if a in {"Calls", "SMS", "Emails"}]

    def _ensure_min_contact(level: Dict[str, int]) -> None:
        if not contact_actions:
            return
        if sum(level.get(a, 0) for a in contact_actions) > 0:
            return
        # All contact channels are zero – try to bump one up to 1 while
        # respecting weekly caps (if any).
        for a in contact_actions:
            cap = float(weekly_caps.get(a, float("inf")))
            if cap != 0.0:
                level[a] = max(1, level.get(a, 0))
                break

    _ensure_min_contact(low)
    _ensure_min_contact(medium)
    _ensure_min_contact(high)

    return {"Low": low, "Medium": medium, "High": high}


class LoginFrame(ctk.CTkFrame):
    """Login frame displayed at startup."""

    def __init__(self, master, on_login):
        super().__init__(master)
        self.on_login = on_login

        # Center the login frame in the window
        self.grid(row=0, column=0, padx=20, pady=20, sticky="")
        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure((0, 1, 2, 3, 4), weight=1)

        title = ctk.CTkLabel(
            self,
            text="Debt Collection Strategy Engine",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        title.grid(row=0, column=0, pady=(40, 10), padx=20)

        self.username_entry = ctk.CTkEntry(
            self,
            placeholder_text="Username",
            width=250,
        )
        self.username_entry.grid(row=1, column=0, pady=10, padx=20)

        self.password_entry = ctk.CTkEntry(
            self,
            placeholder_text="Password",
            show="*",
            width=250,
        )
        self.password_entry.grid(row=2, column=0, pady=10, padx=20)

        login_btn = ctk.CTkButton(
            self,
            text="Login (Admin/User)",
            command=self._handle_login,
        )
        login_btn.grid(row=3, column=0, pady=(10, 5), padx=20)

        skip_btn = ctk.CTkButton(
            self,
            text="Skip / User Mode",
            fg_color="#444444",
            hover_color="#666666",
            command=self._handle_skip,
        )
        skip_btn.grid(row=4, column=0, pady=(5, 40), padx=20)

    def _handle_login(self) -> None:
        username = self.username_entry.get().strip()
        password = self.password_entry.get().strip()
        role = authenticate(username, password)
        if role is None:
            messagebox.showerror("Login failed", "Invalid username or password.")
            return
        self.on_login(role)

    def _handle_skip(self) -> None:
        # Skip login and go directly to user mode.
        self.on_login("user")


class MainApp(ctk.CTk):
    """Main CustomTkinter application shell."""

    def __init__(self) -> None:
        super().__init__()

        # Global appearance (dark mode, McGrath & Arthur style: deep red + gold).
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.title("Debt Collection Strategy Engine")
        self.geometry("1200x800")
        
        # Start in full screen (maximized) mode
        self.after(10, lambda: self.state("zoomed"))

        # Set application icon (robust to different working directories).
        # On Windows, taskbar/window icon uses .ico; iconphoto alone often only affects child windows.
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            if sys.platform == "win32":
                ico_path = os.path.join(base_dir, "logo.ico")
                if os.path.exists(ico_path):
                    self.iconbitmap(ico_path)
                else:
                    png_path = os.path.join(base_dir, "logo.png")
                    if os.path.exists(png_path):
                        try:
                            from PIL import Image
                            img = Image.open(png_path)
                            if img.mode != "RGBA":
                                img = img.convert("RGBA")
                            fd, ico_path_temp = tempfile.mkstemp(suffix=".ico", prefix="app_icon_")
                            os.close(fd)
                            img.save(ico_path_temp, format="ICO", sizes=[(256, 256), (48, 48), (32, 32), (16, 16)])
                            self.iconbitmap(ico_path_temp)
                            self._icon_temp_path = ico_path_temp
                        except Exception as pil_e:
                            print(f"Icon conversion failed: {pil_e}")
                            self._app_icon = tk.PhotoImage(file=png_path)
                            self.iconphoto(True, self._app_icon)
                    else:
                        self._app_icon = None
            else:
                icon_path = os.path.join(base_dir, "logo.png")
                if os.path.exists(icon_path):
                    self._app_icon = tk.PhotoImage(file=icon_path)
                    self.iconphoto(True, self._app_icon)
                else:
                    self._app_icon = None
        except Exception as e:
            print(f"Icon not found: {e}")

        self.role: Optional[str] = None
        self._loading_window: Optional[ctk.CTkToplevel] = None
        
        # Configure grid weights for centering login frame
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Show login first.
        self.login_frame = LoginFrame(self, self._on_login_success)
        self.main_frame: Optional[ctk.CTkFrame] = None

    def _on_login_success(self, role: str) -> None:
        self.role = role
        if self.login_frame:
            self.login_frame.destroy()

        # Show a lightweight loading dialog while the main UI (and
        # training data) are being prepared. Using `after` ensures the
        # dialog is rendered before the heavier work starts.
        self._show_loading_screen("Loading model and training data...\nThis may take a few seconds.")
        self.after(100, self._build_main_ui)

    def _show_loading_screen(self, message: str) -> None:
        """Display a simple modal loading dialog."""
        if self._loading_window is not None:
            try:
                self._loading_window.destroy()
            except Exception:
                pass
            self._loading_window = None

        loading = ctk.CTkToplevel(self)
        loading.title("Please wait")
        loading.geometry("420x180")
        loading.transient(self)
        loading.grab_set()

        # Center on screen
        loading.update_idletasks()
        x = (loading.winfo_screenwidth() - 420) // 2
        y = (loading.winfo_screenheight() - 180) // 2
        loading.geometry(f"+{x}+{y}")

        frame = ctk.CTkFrame(loading, corner_radius=10)
        frame.pack(expand=True, fill="both", padx=20, pady=20)

        label = ctk.CTkLabel(
            frame,
            text=message,
            font=ctk.CTkFont(size=14, weight="bold"),
            justify="center",
        )
        label.pack(pady=(10, 15), padx=10)

        progress = ctk.CTkProgressBar(frame, mode="indeterminate")
        progress.pack(fill="x", padx=40, pady=(0, 10))
        progress.start()

        tip = ctk.CTkLabel(
            frame,
            text="Tip: First load can be slower if training data is large.",
            font=ctk.CTkFont(size=11),
            text_color="#AAAAAA",
            justify="center",
            wraplength=360,
        )
        tip.pack(pady=(0, 5), padx=10)

        self._loading_window = loading

    def _hide_loading_screen(self) -> None:
        """Close the loading dialog if it is visible."""
        if self._loading_window is not None:
            try:
                self._loading_window.grab_release()
            except Exception:
                pass
            try:
                self._loading_window.destroy()
            except Exception:
                pass
            self._loading_window = None

    def _return_to_login(self) -> None:
        """Return from the main UI back to the login screen."""
        # Destroy main content frame if it exists.
        if self.main_frame is not None:
            try:
                self.main_frame.destroy()
            except Exception:
                pass
            self.main_frame = None

        # Reset role information.
        self.role = None

        # Re-create the login frame in the center of the window.
        self.login_frame = LoginFrame(self, self._on_login_success)

    def _build_main_ui(self) -> None:
        # Main content frame (tabs, header, etc.).
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # Header frame with title and help button.
        header_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        header_frame.pack(fill="x", pady=10, padx=10)

        # Title on the left.
        header = ctk.CTkLabel(
            header_frame,
            text="Debt Collection Portfolio Strategy Engine",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        header.pack(side="left", padx=10)

        # Right-side controls: return-to-login and help buttons.
        button_container = ctk.CTkFrame(header_frame, fg_color="transparent")
        button_container.pack(side="right")

        return_btn = ctk.CTkButton(
            button_container,
            text="Logout",
            width=80,
            height=35,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#444444",
            hover_color="#666666",
            command=self._return_to_login,
        )
        return_btn.pack(side="left", padx=(0, 8))

        # Help button on the right of the logout button.
        help_btn = ctk.CTkButton(
            button_container,
            text="?",
            width=35,
            height=35,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#8B0000",
            hover_color="#B22222",
            command=self._show_help,
        )
        help_btn.pack(side="left")

        role_text = f"Logged in as: {self.role or 'user'}"
        role_label = ctk.CTkLabel(
            self.main_frame,
            text=role_text,
            font=ctk.CTkFont(size=12, weight="normal"),
        )
        role_label.pack(pady=(0, 5))

        # Tabs.
        tabview = ctk.CTkTabview(self.main_frame, width=1150, height=680)
        tabview.pack(expand=True, fill="both", padx=10, pady=10)
        self._main_tabview = tabview

        prediction_tab = tabview.add("Prediction")
        insights_tab = tabview.add("Training Insights")
        training_tab = tabview.add("Training")
        testing_tab = tabview.add("Testing")

        self.prediction_ui = PredictionTab(prediction_tab)
        self.insights_ui = InsightsTab(insights_tab, auto_load=False)
        self.training_ui = TrainingTab(training_tab, enabled=(self.role == "admin"))
        self.testing_ui = TestingTab(testing_tab, enabled=(self.role == "admin"))
        if hasattr(tabview, "_segmented_button"):
            try:
                base_command = tabview._segmented_button.cget("command")
                def _combined_tab_command(selected_tab: str) -> None:
                    if callable(base_command):
                        base_command(selected_tab)
                    self._on_main_tab_changed(selected_tab)
                tabview._segmented_button.configure(command=_combined_tab_command)
            except Exception:
                pass

        # Main UI is now ready; hide the loading screen if it is showing.
        self._hide_loading_screen()

    def _on_main_tab_changed(self, selected_tab: str) -> None:
        if selected_tab == "Training Insights" and hasattr(self, "insights_ui"):
            try:
                self.insights_ui.ensure_loaded()
            except Exception:
                pass

    def _show_help(self) -> None:
        """Display user_guide.md content in a scrollable dialog."""
        readme_path = "user_guide.md"
        if os.path.exists(readme_path):
            try:
                with open(readme_path, "r", encoding="utf-8") as f:
                    md_content = f.read()
                # Preserve single newlines from the Markdown file so the
                # text layout matches what is written in user_guide.md.
                content = md.markdown(md_content, extras=["break-on-newline"])
            except Exception as e:
                content = f"<h1>Error reading user guide:</h1><p>{e}</p>"
        else:
            content = "<h1>User Guide file not found.</h1><p>Please contact the administrator.</p>"

        # Create help dialog window.
        help_window = ctk.CTkToplevel(self)
        help_window.title("Help - User Guide")
        help_window.geometry("900x700")
        help_window.transient(self)
        help_window.grab_set()

        # Center the window on screen.
        help_window.update_idletasks()
        x = (help_window.winfo_screenwidth() - 900) // 2
        y = (help_window.winfo_screenheight() - 700) // 2
        help_window.geometry(f"+{x}+{y}")

        # Header.
        header = ctk.CTkLabel(
            help_window,
            text="User Guide",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        header.pack(pady=(15, 10))

        # Large scrollable area for rendered HTML user guide.
        # Use tkhtmlview's HTMLScrolledText, which manages its own scrollbars and
        # expands to fill the available space. Style it to better match the
        # dark CustomTkinter theme.
        html_view = HtmlScrolledText(
            help_window,
            html=content,
            background="#1E1E1E",
            borderwidth=0,
            relief="flat",
            padx=20,
            pady=20,
        )
        html_view.pack(expand=True, fill="both", padx=20, pady=(0, 10))

        # Ensure text and HTML tags render in light color on dark background.
        html_view.configure(fg="#FFFFFF", insertbackground="#FFFFFF")
        for tag_name in html_view.tag_names():
            try:
                html_view.tag_config(tag_name, foreground="#FFFFFF")
            except Exception:
                pass

        # Subtly style the internal scrollbar (created by HtmlScrolledText)
        # so it blends into the dark theme instead of standing out.
        for child in html_view.winfo_children():
            if isinstance(child, tk.Scrollbar):
                try:
                    child.configure(
                        background="#1E1E1E",
                        troughcolor="#1E1E1E",
                        borderwidth=0,
                        highlightthickness=0,
                        elementborderwidth=0,
                        width=8,
                    )
                except Exception:
                    pass

        # Make the help content read-only so users cannot edit the
        # underlying user_guide.md text from this window.
        try:
            html_view.configure(state="disabled")
        except Exception:
            # If tkhtmlview doesn't support disabling, ignore gracefully.
            pass

        # Close button.
        close_btn = ctk.CTkButton(
            help_window,
            text="Close",
            width=120,
            fg_color="#8B0000",
            hover_color="#B22222",
            command=help_window.destroy,
        )
        close_btn.pack(pady=(0, 15))


class PredictionTab(ctk.CTkFrame):
    """Prediction/Strategy tab for user mode."""

    def __init__(self, master):
        super().__init__(master)
        self.pack(expand=True, fill="both")

        self.config_dict = load_config()
        self.uploaded_df: Optional[pd.DataFrame] = None
        self.action_features = list(self.config_dict["columns"].get("action_features", []))
        self.action_costs = self._normalize_action_costs(
            self.config_dict.get("action_costs", {})
        )
        self.effort_levels: Dict[str, Dict[str, int]] = {}
        self._effective_effort_levels: Dict[str, Dict[str, int]] = {}
        self._effort_levels_by_client: Dict[str, Dict[str, Dict[str, int]]] = {}
        self._baseline_weekly_by_client: Dict[str, Dict[str, float]] = {}
        self._actions_never_by_client: Dict[str, set] = {}
        self._training_baseline_actions: Optional[Dict[str, float]] = None

        # Matplotlib figure embedded inside the Tkinter frame.
        self.fig: Optional[Figure] = None
        self.ax = None
        self.canvas: Optional[FigureCanvasTkAgg] = None
        self.client_filter_var = ctk.StringVar(value="Full Portfolio")
        self.import_filter_var = ctk.StringVar(value="All Imports")
        self.baseline_mode_var = ctk.StringVar(value="Default")
        self.percent_change_var = ctk.StringVar(value="100")
        self._portfolio_baseline_actions: Optional[Dict[str, float]] = None
        self.target_paid_value_var = ctk.StringVar(value="")
        self._effort_selection_var = ctk.StringVar(value="Medium")
        self._chart_view: Literal["portfolio", "monthly"] = "portfolio"
        self._clients_long_active_history: Dict[str, bool] = {}
        self._client_duration_stats: Dict[str, Dict[str, float]] = {}
        self._training_avg_weeks_open: Optional[float] = None

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # Left side controls.
        control_frame = ctk.CTkFrame(self, corner_radius=10)
        control_frame.grid(row=0, column=0, sticky="nsw", padx=5, pady=5)

        title = ctk.CTkLabel(
            control_frame,
            text="Prediction / Strategy",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        title.pack(pady=(5, 5), padx=10)

        # Button frame for upload and clear
        btn_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        btn_frame.pack(pady=(5, 5), padx=10, fill="x")

        upload_btn = ctk.CTkButton(
            btn_frame,
            text="Upload Portfolio Data",
            command=self._handle_upload,
        )
        upload_btn.pack(side="left", padx=(0, 5))

        clear_btn = ctk.CTkButton(
            btn_frame,
            text="Clear Data",
            command=self._clear_uploaded_data,
            fg_color="#8B0000",
            hover_color="#B22222",
            width=80,
        )
        clear_btn.pack(side="left")

        self.status_label = ctk.CTkLabel(
            control_frame,
            text="No file uploaded.",
            wraplength=260,
            justify="left",
        )
        self.status_label.pack(pady=(5, 10), padx=10)

        # Progress bar for strategy simulation
        self.progress_bar = ctk.CTkProgressBar(control_frame, mode="determinate")
        self.progress_bar.pack(pady=(0, 5), padx=10, fill="x")
        self.progress_bar.set(0)
        self.progress_label = ctk.CTkLabel(
            control_frame,
            text="",
            font=ctk.CTkFont(size=10),
            text_color="#888888",
        )
        self.progress_label.pack(pady=(0, 5), padx=10)

        run_btn = ctk.CTkButton(
            control_frame,
            text="Run Strategy Simulation",
            command=self._run_strategy,
        )
        run_btn.pack(pady=(10, 10), padx=10)

        # Collapsible Effort Settings section
        settings_frame = ctk.CTkFrame(control_frame, corner_radius=8)
        settings_frame.pack(pady=(5, 5), padx=10, fill="x")
        
        # Header with toggle button
        settings_header = ctk.CTkFrame(settings_frame, fg_color="transparent")
        settings_header.pack(fill="x", padx=5, pady=(5, 0))
        self.settings_expanded = ctk.BooleanVar(value=False)
        toggle_btn_settings = ctk.CTkButton(
            settings_header,
            text="▶ Effort Settings",
            command=lambda: self._toggle_section("settings"),
            fg_color="transparent",
            hover_color="#2b2b2b",
            anchor="w",
            width=200,
        )
        toggle_btn_settings.pack(side="left")
        self.settings_toggle_btn = toggle_btn_settings
        
        # Content frame (collapsed by default)
        self.settings_content_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        # Don't pack initially - will be shown when expanded

        baseline_row = ctk.CTkFrame(self.settings_content_frame)
        baseline_row.pack(fill="x", pady=(2, 5), padx=5)
        ctk.CTkLabel(baseline_row, text="Baseline:").pack(side="left")
        self.baseline_menu = ctk.CTkOptionMenu(
            baseline_row,
            values=["Default", "Training Average", "Custom"],
            variable=self.baseline_mode_var,
            command=self._toggle_baseline_mode,
        )
        self.baseline_menu.pack(side="right", fill="x", expand=True, padx=5)

        self.custom_baseline_frame = ctk.CTkFrame(self.settings_content_frame)
        self.custom_baseline_frame.pack(fill="x", pady=(0, 5), padx=5)
        self.custom_baseline_entries: Dict[str, ctk.CTkEntry] = {}
        for action in self.action_features:
            row = ctk.CTkFrame(self.custom_baseline_frame)
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=f"{action}:").pack(side="left")
            entry = ctk.CTkEntry(row, width=80)
            entry.pack(side="right", padx=5)
            self.custom_baseline_entries[action] = entry

        percent_row = ctk.CTkFrame(self.settings_content_frame)
        percent_row.pack(fill="x", pady=2, padx=5)
        ctk.CTkLabel(percent_row, text="Change % (+/-):").pack(side="left")
        self.percent_entry = ctk.CTkEntry(
            percent_row,
            width=80,
            textvariable=self.percent_change_var,
        )
        self.percent_entry.pack(side="right", padx=5)

        target_row = ctk.CTkFrame(self.settings_content_frame)
        target_row.pack(fill="x", pady=2, padx=5)
        ctk.CTkLabel(target_row, text="Target Paid Value:").pack(side="left")
        self.target_value_entry = ctk.CTkEntry(
            target_row,
            width=120,
            textvariable=self.target_paid_value_var,
        )
        self.target_value_entry.pack(side="right", padx=5)

        self.computed_percent_label = ctk.CTkLabel(
            self.settings_content_frame,
            text="Computed % from target: -",
            anchor="w",
        )
        self.computed_percent_label.pack(pady=(2, 8), padx=5, fill="x")

        # Collapsible Action Costs section
        costs_frame = ctk.CTkFrame(control_frame, corner_radius=8)
        costs_frame.pack(pady=(0, 10), padx=10, fill="x")
        
        # Header with toggle button
        costs_header = ctk.CTkFrame(costs_frame, fg_color="transparent")
        costs_header.pack(fill="x", padx=5, pady=(5, 0))
        self.costs_expanded = ctk.BooleanVar(value=False)
        toggle_btn_costs = ctk.CTkButton(
            costs_header,
            text="▶ Action Costs (EUR / action)",
            command=lambda: self._toggle_section("costs"),
            fg_color="transparent",
            hover_color="#2b2b2b",
            anchor="w",
            width=250,
        )
        toggle_btn_costs.pack(side="left")
        self.costs_toggle_btn = toggle_btn_costs
        
        # Content frame (collapsed by default)
        self.costs_content_frame = ctk.CTkFrame(costs_frame, fg_color="transparent")
        # Don't pack initially - will be shown when expanded

        self.cost_entries: Dict[str, ctk.CTkEntry] = {}
        for action in self.action_features:
            row = ctk.CTkFrame(self.costs_content_frame)
            row.pack(fill="x", pady=2, padx=5)
            ctk.CTkLabel(row, text=f"{action}:").pack(side="left")
            entry = ctk.CTkEntry(row, width=80)
            entry.insert(0, f"{self.action_costs.get(action, 0.0):.2f}")
            entry.pack(side="right", padx=5)
            self.cost_entries[action] = entry

        save_costs_btn = ctk.CTkButton(
            self.costs_content_frame,
            text="Save Costs to configuration",
            command=self._save_action_costs,
        )
        save_costs_btn.pack(pady=(6, 8), padx=5)

        # Collapsible Portfolio Lengths section (monthly breakdown horizon per client)
        lengths_frame = ctk.CTkFrame(control_frame, corner_radius=8)
        lengths_frame.pack(pady=(0, 10), padx=10, fill="x")

        lengths_header = ctk.CTkFrame(lengths_frame, fg_color="transparent")
        lengths_header.pack(fill="x", padx=5, pady=(5, 0))
        self.lengths_expanded = ctk.BooleanVar(value=False)
        toggle_btn_lengths = ctk.CTkButton(
            lengths_header,
            text="▶ Portfolio Lengths",
            command=lambda: self._toggle_section("lengths"),
            fg_color="transparent",
            hover_color="#2b2b2b",
            anchor="w",
            width=250,
        )
        toggle_btn_lengths.pack(side="left")
        self.lengths_toggle_btn = toggle_btn_lengths

        self.lengths_content_frame = ctk.CTkFrame(lengths_frame, fg_color="transparent")

        ctk.CTkLabel(
            self.lengths_content_frame,
            text="Max active months per client (blank = no limit, e.g. 8d for days)",
            wraplength=260,
            justify="left",
            font=ctk.CTkFont(size=10),
            text_color="#888888",
        ).pack(fill="x", padx=5, pady=(2, 4))

        self.portfolio_length_scroll = ctk.CTkScrollableFrame(
            self.lengths_content_frame,
            height=220,
        )
        self.portfolio_length_scroll.pack(fill="x", padx=5, pady=(0, 4))
        self.portfolio_length_entries: Dict[str, ctk.CTkEntry] = {}
        self._populate_portfolio_length_fields()

        save_lengths_btn = ctk.CTkButton(
            self.lengths_content_frame,
            text="Save Portfolio Lengths",
            command=self._save_portfolio_lengths,
        )
        save_lengths_btn.pack(pady=(4, 8), padx=5)

        # Right side graph.
        graph_container = ctk.CTkFrame(self, corner_radius=10)
        graph_container.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        graph_container.rowconfigure(0, weight=0)  # Chart toolbar
        graph_container.rowconfigure(1, weight=1)  # Graph area expands
        graph_container.rowconfigure(2, weight=0)  # Matplotlib toolbar
        graph_container.rowconfigure(3, weight=0)  # Filter button
        graph_container.rowconfigure(4, weight=0)  # Filter status label
        graph_container.columnconfigure(0, weight=1)

        self.chart_toolbar_frame = ctk.CTkFrame(graph_container, fg_color="transparent")
        self.chart_toolbar_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 0))
        self.chart_toolbar_frame.columnconfigure(0, weight=1)
        chart_toolbar_frame = self.chart_toolbar_frame

        self.effort_toolbar_inner = ctk.CTkFrame(chart_toolbar_frame, fg_color="transparent")
        self.effort_toolbar_inner.pack(side="left", fill="x", expand=True)

        ctk.CTkLabel(
            self.effort_toolbar_inner,
            text="Effort:",
            font=ctk.CTkFont(size=12),
        ).pack(side="left", padx=(0, 8))

        self.effort_segmented = ctk.CTkSegmentedButton(
            self.effort_toolbar_inner,
            values=["Low", "Medium", "High"],
            command=self._on_effort_segment_changed,
            font=ctk.CTkFont(size=12),
        )
        self.effort_segmented.set("Medium")
        self.effort_segmented.pack(side="left")
        self.effort_toolbar_inner.pack_forget()

        drill_btn_frame = ctk.CTkFrame(chart_toolbar_frame, fg_color="transparent")
        drill_btn_frame.pack(side="right")

        self.drill_through_btn = ctk.CTkButton(
            drill_btn_frame,
            text="Monthly Breakdown",
            command=self._enter_monthly_view,
            fg_color="#1f6aa5",
            hover_color="#144870",
            width=160,
            height=28,
            font=ctk.CTkFont(size=12),
        )
        self.drill_through_btn.pack(side="left", padx=(0, 6))
        self.drill_through_btn.pack_forget()

        self.back_portfolio_btn = ctk.CTkButton(
            drill_btn_frame,
            text="Back to Portfolio View",
            command=self._enter_portfolio_view,
            fg_color="#666666",
            hover_color="#888888",
            width=160,
            height=28,
            font=ctk.CTkFont(size=12),
        )
        self.back_portfolio_btn.pack(side="left")
        self.back_portfolio_btn.pack_forget()

        # Matplotlib Figure and Canvas for interactive viewing (zoom, pan).
        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.fig.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.15)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_container)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        toolbar = NavigationToolbar2Tk(
            self.canvas,
            graph_container,
            pack_toolbar=False,
        )
        toolbar.update()
        toolbar.grid(row=2, column=0, sticky="ew", padx=5, pady=(0, 5))

        # Filter button at the bottom center of the graph
        filter_btn = ctk.CTkButton(
            graph_container,
            text="Filter Results",
            command=self._show_filter_popup,
            fg_color="#1f6aa5",
            hover_color="#144870",
            width=150,
            height=35,
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        filter_btn.grid(row=3, column=0, pady=(5, 10))

        # Current filter display label
        self.filter_status_label = ctk.CTkLabel(
            graph_container,
            text="Filter: Full Portfolio | All Imports",
            font=ctk.CTkFont(size=11),
            text_color="#888888",
        )
        self.filter_status_label.grid(row=4, column=0, pady=(0, 5))

        self._hover_bars = None
        self._hover_meta = []
        self._hover_annot = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="#222222", ec="#AAAAAA", alpha=0.9),
            color="white",
        )
        self._hover_annot.set_visible(False)
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self._toggle_baseline_mode(self.baseline_mode_var.get())
        
        # Store suggested effort from historical data
        self._suggested_effort: Optional[Dict[str, int]] = None
        self._suggested_effort_by_client: Dict[str, Dict[str, int]] = {}
        self._load_suggested_effort()
        
        # Initialize filter menus (will be populated when data is loaded)
        self._client_filter_values = ["Full Portfolio"]
        self._import_filter_values = ["All Imports"]
        self.chart_toolbar_frame.grid_remove()

    def _handle_upload(self) -> None:
        path = filedialog.askopenfilename(
            title="Select portfolio data file",
            filetypes=[
                ("Supported files", "*.xlsx *.xls *.csv"),
                ("Excel files", "*.xlsx *.xls"),
                ("CSV files", "*.csv"),
            ],
        )
        if not path:
            return

        try:
            df = read_table_file(path)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to read data file:\n{exc}")
            return

        missing, extra = self._validate_columns(df)
        if missing:
            details = "Missing columns:\n  - " + "\n  - ".join(missing)
            messagebox.showerror("Column mismatch vs config.json", details)
            return
        if extra:
            messagebox.showwarning(
                "Extra columns detected",
                "These columns will be ignored by the model:\n  - "
                + "\n  - ".join(extra),
            )

        self.uploaded_df = df
        self._results_df = None
        self._reset_chart_view_state()
        self._render_chart_placeholder(
            "Portfolio uploaded — run Strategy Simulation to view the chart."
        )
        self.status_label.configure(
            text=f"Loaded {os.path.basename(path)} with {len(df)} rows."
        )

        # Update client filter options (string-safe for mixed types).
        client_col = self.config_dict["columns"]["categorical_features"][0]
        clients = (
            df[client_col]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        self._client_filter_values = ["Full Portfolio"] + sorted(clients)
        self.client_filter_var.set("Full Portfolio")

        if "Import Name" in df.columns:
            imports = (
                df["Import Name"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            self._import_filter_values = ["All Imports"] + sorted(imports)
        else:
            self._import_filter_values = ["All Imports"]
        self.import_filter_var.set("All Imports")
        self._update_filter_status_label()
        self._apply_default_effort_settings(df)
        self._populate_portfolio_length_fields()

    def _clear_uploaded_data(self) -> None:
        """Clear the uploaded portfolio data and reset the UI."""
        self.uploaded_df = None
        self._results_df = None
        self._portfolio_baseline_actions = None
        self.target_paid_value_var.set("")
        if self.baseline_mode_var.get() == "Default":
            self._sync_baseline_display_fields()
        self.effort_levels = {}
        self._effective_effort_levels = {}
        self._effort_levels_by_client = {}
        self._actions_never_by_client = {}

        # Reset status
        self.status_label.configure(text="No file uploaded.")

        # Reset filters
        self._client_filter_values = ["Full Portfolio"]
        self.client_filter_var.set("Full Portfolio")
        self._import_filter_values = ["All Imports"]
        self.import_filter_var.set("All Imports")
        self._update_filter_status_label()

        self._reset_chart_view_state()

        # Clear the graph
        if self.ax is not None:
            self.ax.clear()
            self.ax.set_title("No data loaded")
            self.ax.set_xlabel("Effort Level")
            self.ax.set_ylabel("Total Predicted Paid Value (EUR)")
            if self.canvas is not None:
                self.canvas.draw_idle()

    def _load_suggested_effort(self) -> None:
        """Load suggested effort levels from historical training data."""
        try:
            insights = get_training_data_insights()
            
            # Choose the effort bucket with the best payoff:
            # maximize (Avg Paid Value) / (Avg Total Actions).
            # This finds the most efficient historical effort level instead
            # of always assuming that "High" effort is best.
            buckets = insights.get("effort_buckets", {})
            best_bucket_name = None
            best_score = -float("inf")
            best_bucket = None

            for name, stats in buckets.items():
                avg_paid = float(stats.get("avg_paid_value", 0.0))
                avg_total_actions = sum(
                    float(v) for v in stats.get("avg_actions", {}).values()
                )
                if avg_total_actions > 0:
                    score = avg_paid / avg_total_actions
                else:
                    # If there is no effort, fall back to raw paid value
                    score = avg_paid

                if score > best_score:
                    best_score = score
                    best_bucket_name = name
                    best_bucket = stats

            avg_weeks = max(float(insights.get("avg_weeks_open") or 1.0), 1.0)
            if best_bucket is not None:
                per_week = best_bucket.get("avg_actions_per_week") or {}
                if per_week:
                    self._suggested_effort = {
                        action: int(round(float(per_week.get(action, 0.0)) * avg_weeks))
                        for action in self.action_features
                    }
                else:
                    self._suggested_effort = {
                        action: int(round(val))
                        for action, val in best_bucket.get("avg_actions", {}).items()
                    }
            else:
                # Fallback: use training average actions per week when available.
                per_week_stats = insights.get("action_stats_per_week") or {}
                if per_week_stats:
                    self._suggested_effort = {
                        action: int(round(float(per_week_stats[action]["mean"]) * avg_weeks))
                        for action in self.action_features
                        if action in per_week_stats
                    }
                else:
                    self._suggested_effort = {
                        action: int(round(stats["mean"]))
                        for action, stats in insights.get("action_stats", {}).items()
                    }
            self._suggested_effort_by_client = {
                self._normalized_client_key(client): {
                    action: int(value)
                    for action, value in actions.items()
                }
                for client, actions in (insights.get("suggested_effort_by_client") or {}).items()
            }
            self._clients_long_active_history = dict(
                insights.get("clients_long_active_history") or {}
            )
            self._client_duration_stats = dict(
                insights.get("client_duration_stats") or {}
            )
            avg_weeks = insights.get("avg_weeks_open")
            self._training_avg_weeks_open = float(avg_weeks) if avg_weeks else None
        except Exception as e:
            print(f"Could not load suggested effort from training data: {e}")
            self._suggested_effort = None
            self._suggested_effort_by_client = {}
            self._clients_long_active_history = {}
            self._client_duration_stats = {}
            self._training_avg_weeks_open = None

    def _show_filter_popup(self) -> None:
        """Show a popup dialog with filter options."""
        # Create popup window
        popup = ctk.CTkToplevel(self)
        popup.title("Filter Results")
        popup.geometry("350x300")
        popup.transient(self.winfo_toplevel())
        popup.grab_set()
        
        # Center the popup on screen
        popup.update_idletasks()
        x = (popup.winfo_screenwidth() - 350) // 2
        y = (popup.winfo_screenheight() - 300) // 2
        popup.geometry(f"+{x}+{y}")
        
        # Header
        header = ctk.CTkLabel(
            popup,
            text="Filter Simulation Results",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        header.pack(pady=(20, 15))
        
        # Client Filter
        client_frame = ctk.CTkFrame(popup, fg_color="transparent")
        client_frame.pack(fill="x", padx=20, pady=5)
        
        ctk.CTkLabel(
            client_frame,
            text="Client:",
            font=ctk.CTkFont(size=12),
            width=80,
            anchor="w",
        ).pack(side="left")
        
        client_menu = ctk.CTkOptionMenu(
            client_frame,
            values=self._client_filter_values,
            variable=self.client_filter_var,
            width=200,
        )
        client_menu.pack(side="right", fill="x", expand=True)
        
        # Import Filter
        import_frame = ctk.CTkFrame(popup, fg_color="transparent")
        import_frame.pack(fill="x", padx=20, pady=5)
        
        ctk.CTkLabel(
            import_frame,
            text="Import:",
            font=ctk.CTkFont(size=12),
            width=80,
            anchor="w",
        ).pack(side="left")
        
        import_menu = ctk.CTkOptionMenu(
            import_frame,
            values=self._import_filter_values,
            variable=self.import_filter_var,
            width=200,
        )
        import_menu.pack(side="right", fill="x", expand=True)
        
        # Buttons frame
        btn_frame = ctk.CTkFrame(popup, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=(30, 20))
        
        def apply_and_close():
            self._update_graph_for_filter(None)
            self._update_filter_status_label()
            popup.destroy()
        
        def reset_filters():
            self.client_filter_var.set("Full Portfolio")
            self.import_filter_var.set("All Imports")
        
        reset_btn = ctk.CTkButton(
            btn_frame,
            text="Reset",
            command=reset_filters,
            fg_color="#666666",
            hover_color="#888888",
            width=100,
        )
        reset_btn.pack(side="left", padx=(0, 10))
        
        apply_btn = ctk.CTkButton(
            btn_frame,
            text="Apply Filter",
            command=apply_and_close,
            fg_color="#1f6aa5",
            hover_color="#144870",
            width=120,
        )
        apply_btn.pack(side="right")
        
        # Info label
        info_label = ctk.CTkLabel(
            popup,
            text="Select filters to view specific client or import results",
            font=ctk.CTkFont(size=10),
            text_color="#888888",
        )
        info_label.pack(pady=(0, 15))

    def _update_filter_status_label(self) -> None:
        """Update the filter status label below the graph."""
        client = self.client_filter_var.get()
        import_name = self.import_filter_var.get()
        
        # Shorten the display if needed
        if len(client) > 30:
            client = client[:27] + "..."
        if len(import_name) > 30:
            import_name = import_name[:27] + "..."
        
        self.filter_status_label.configure(
            text=f"Filter: {client} | {import_name}"
        )

    def _validate_columns(self, df: pd.DataFrame):
        """
        Validate that the uploaded file matches the expected schema.

        Expected columns:
        - Import metadata columns
        - All categorical_features and numerical_features from config.json
        (Action and Paid Value columns are optional.)
        """
        cols_cfg = self.config_dict["columns"]
        expected = set(
            cols_cfg["categorical_features"] + cols_cfg["numerical_features"]
        )
        # Import metadata columns expected in portfolio_data_real.xlsx.
        expected.update({"Import Name", "Import date"})
        # Debtor identifier column used to disambiguate Product placeholders.
        expected.add(cols_cfg.get("debtor_name_column", "Name"))

        actual = set(df.columns)
        missing = sorted(expected - actual)
        # Extra columns are allowed but reported.
        extra = sorted(actual - expected)
        return missing, extra

    def _update_progress(self, value: float, text: str = "") -> None:
        """Update progress bar and label."""
        self.progress_bar.set(value)
        if text:
            self.progress_label.configure(text=text)
        self.update_idletasks()

    def _run_strategy(self) -> None:
        if self.uploaded_df is None or self.uploaded_df.empty:
            messagebox.showwarning("No data", "Please upload portfolio_data_real.xlsx first.")
            return

        self._update_progress(0.0, "Starting strategy simulation...")

        try:
            # Always use the fixed‑effort (Low/Medium/High) strategy simulation,
            # which runs quickly enough for normal users.
            results_df = self._simulate_effort_levels(self.uploaded_df)
        except Exception as exc:
            self._update_progress(0.0, "")
            messagebox.showerror("Error", f"Failed to run strategy simulation:\n{exc}")
            return

        # Save predictions to Excel with timestamp inside data/predictions/.
        date_str = datetime.now().strftime("%Y-%m-%d")
        predictions_dir = os.path.join("data", "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
        out_path = os.path.join(predictions_dir, f"predictions_output_{date_str}.xlsx")
        try:
            results_df.to_excel(out_path, index=False)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to save predictions:\n{exc}")
            return

        self.status_label.configure(
            text=f"Strategy simulation complete.\nSaved results to {out_path}"
        )

        # Keep aggregated results for further filtering.
        self._results_df = results_df

        # Update client filter with detected clients from results (with case counts)
        cols_cfg = self.config_dict["columns"]
        client_col = cols_cfg["categorical_features"][0]
        if client_col in results_df.columns:
            # Get unique clients with their case counts
            client_counts = results_df[client_col].astype(str).value_counts()
            total_cases = len(results_df)
            values = [f"Full Portfolio ({total_cases:,} cases)"]
            for client in sorted(client_counts.index):
                count = client_counts[client]
                values.append(f"{client} ({count:,} cases)")
            self._client_filter_values = values
            self.client_filter_var.set(values[0])

        # Update import filter if available
        if "Import Name" in results_df.columns:
            imports = (
                results_df["Import Name"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            self._import_filter_values = ["All Imports"] + sorted(imports)
        else:
            self._import_filter_values = ["All Imports"]
        self.import_filter_var.set("All Imports")
        self._update_filter_status_label()

        # Build initial graph for "Full Portfolio".
        self._update_progress(1.0, "Complete!")
        self.after(500, lambda: self._update_progress(0.0, ""))
        self._reset_chart_view_state()
        self._refresh_chart()

    def _simulate_optimal_actions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Use recommend_portfolio_strategy to find optimal actions per case.
        This is slower but gives the best predictions per case.
        """
        self._update_progress(0.1, "Loading model artifacts...")
        config = load_config()
        action_costs = self.action_costs

        def progress_callback(current: int, total: int) -> None:
            if total > 0:
                progress = 0.1 + 0.85 * (current / total)
                self._update_progress(progress, f"Processing cases: {current}/{total}")

        self._update_progress(0.15, "Finding optimal actions per case...")
        result_df = recommend_portfolio_strategy(df, config=config, action_costs=action_costs, progress_callback=progress_callback)

        # Rename columns to match expected format for graph rendering
        if "Optimal_Predicted_Value" in result_df.columns:
            result_df = result_df.rename(columns={"Optimal_Predicted_Value": "Pred_Optimal"})
            # Create dummy Low/Medium/High columns for compatibility with graph rendering
            result_df["Pred_Low"] = result_df["Pred_Optimal"] * 0.8
            result_df["Pred_Medium"] = result_df["Pred_Optimal"] * 0.9
            result_df["Pred_High"] = result_df["Pred_Optimal"]
            result_df["Pred_Suggested"] = result_df["Pred_Optimal"]

        # Store optimal actions as effort levels for hover display
        optimal_actions = {}
        for action in self.action_features:
            col = f"Optimal_{action}"
            if col in result_df.columns:
                optimal_actions[action] = int(result_df[col].median())
        self.effort_levels = {"Optimal": optimal_actions}

        return result_df

    def _simulate_effort_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate Low / Medium / High / Suggested effort per client subset
        while reusing one global model artifact.
        """
        self._update_progress(0.1, "Loading model artifacts...")
        config, encoder, scaler, target_scaler = load_artifacts_for_inference()
        self._update_progress(0.2, "Loading GBM bundle...")
        bundle = load_paid_bundle(config)
        self._update_progress(0.3, "Computing effort levels...")

        cols_cfg = config["columns"]
        baseline_mode = self.baseline_mode_var.get()
        global_baseline_actions = self._get_baseline_actions()

        # Convert weekly baselines to approximate total actions per case
        # using the historical average case duration (in weeks). This keeps
        # model inputs in the same space it was trained on (totals), while
        # the UI edits remain in per‑week units.
        insights = get_training_data_insights()
        avg_weeks_open = insights.get("avg_weeks_open") or 1.0
        if avg_weeks_open <= 0:
            avg_weeks_open = 1.0

        # If some actions (e.g. Letters) were never used historically at all,
        # cap them at zero in all effort levels so we don't suggest actions
        # the portfolio has effectively never applied.
        never_used_actions = set(insights.get("never_used_actions") or [])
        config_for_effort = dict(config)
        weekly_caps_cfg = dict(config_for_effort.get("weekly_action_caps") or {})
        for action in never_used_actions:
            weekly_caps_cfg[action] = 0.0
        config_for_effort["weekly_action_caps"] = weekly_caps_cfg

        # Also capture per-client \"never used\" actions so we can zero them
        # only for clients that historically never applied them. This is used
        # later inside predict_for_actions_df.
        never_used_by_client = insights.get("never_used_actions_by_client") or {}
        actions_never_by_client: Dict[str, set] = {}
        for client, actions in never_used_by_client.items():
            for action in actions:
                normalized = self._normalized_client_key(client)
                actions_never_by_client.setdefault(action, set()).add(normalized)

        # Also enforce hard-zero channels from the currently simulated dataset.
        # This makes effort scenarios respect client behavior in the uploaded
        # portfolio slice even when training history contains mixed patterns.
        client_col_name = cols_cfg.get("categorical_features", ["Client"])[0]
        if client_col_name in df.columns:
            grouped = df.groupby(df[client_col_name].astype(str).str.strip())
            for client, group in grouped:
                normalized_client = self._normalized_client_key(client)
                for action in self.action_features:
                    if action in group.columns:
                        series = pd.to_numeric(group[action], errors="coerce").fillna(0)
                        if float(series.max()) == 0.0:
                            actions_never_by_client.setdefault(action, set()).add(normalized_client)
        self._actions_never_by_client = actions_never_by_client

        categorical_features = [
            c
            for c in cols_cfg["categorical_features"]
            if c not in cols_cfg.get("exclude_features", [])
        ]
        numeric_like_features = [
            c
            for c in (cols_cfg["numerical_features"] + cols_cfg["action_features"])
            if c not in cols_cfg.get("exclude_features", [])
        ]

        def constrain_actions_for_client(actions: Dict[str, int], normalized_client: str) -> Dict[str, int]:
            constrained = {a: int(actions.get(a, 0)) for a in self.action_features}
            for action, clients_never in actions_never_by_client.items():
                if action in constrained and normalized_client in clients_never:
                    constrained[action] = 0
            return constrained

        def resolve_suggested_actions(
            normalized_client: str,
            medium_actions: Dict[str, int],
        ) -> Dict[str, int]:
            raw = self._suggested_effort_by_client.get(normalized_client) or self._suggested_effort
            if raw:
                return {a: int(raw.get(a, 0)) for a in self.action_features}
            return {a: int(medium_actions.get(a, 0)) for a in self.action_features}

        def predict_for_actions_df(base_df: pd.DataFrame, actions: Dict[str, int]) -> np.ndarray:
            # If all actions are zero (no effort), enforce zero predicted
            # recovery for every case. This encodes the domain rule that
            # without any contact attempts there is no collection.
            if not any(actions.values()):
                print("DEBUG predict_for_actions_df - All actions are 0, returning zeros for predictions.")
                return np.zeros(len(base_df), dtype=float)

            working = base_df.copy()

            # Apply the same action level to all rows by default.
            for action, val in actions.items():
                working[action] = val

            # Then, for any (client, action) combinations where the action has
            # never been used historically, force that action to 0 so we don't
            # propose it for that client even if it is allowed elsewhere.
            client_col = cols_cfg.get("categorical_features", ["Client"])[0]
            if client_col in working.columns and actions_never_by_client:
                clients_series = working[client_col].astype(str).map(self._normalized_client_key)
                for action, clients_never in actions_never_by_client.items():
                    if action in working.columns and clients_never:
                        mask = clients_series.isin(clients_never)
                        if mask.any():
                            working.loc[mask, action] = 0

            print(f"DEBUG predict_for_actions_df - Actions applied: {actions}")
            print(f"DEBUG - Working df shape: {working.shape}, columns: {list(working.columns)[:10]}...")

            if categorical_features:
                for col in categorical_features:
                    if col not in working.columns:
                        working[col] = "Unknown"
                    working[col] = working[col].fillna("Unknown").astype(str)

            if numeric_like_features:
                if hasattr(scaler, "feature_names_in_"):
                    numeric_cols = list(scaler.feature_names_in_)
                else:
                    numeric_cols = list(numeric_like_features)
                print(f"DEBUG - Numeric cols from scaler: {numeric_cols}")
                weekly_names = cols_cfg.get("weekly_action_features", [])
                # Keep weekly-rate features aligned with the currently tested
                # total action plan so each effort level produces distinct
                # model inputs.
                for weekly_col in weekly_names:
                    if weekly_col.endswith("_per_week"):
                        action_name = weekly_col.replace("_per_week", "")
                        if action_name in actions:
                            working[weekly_col] = actions.get(action_name, 0) / max(avg_weeks_open, 1.0)
                if "total_actions" in working.columns:
                    working["total_actions"] = 0.0
                    for action_name in self.action_features:
                        if action_name in working.columns:
                            working["total_actions"] += pd.to_numeric(
                                working[action_name], errors="coerce"
                            ).fillna(0.0)
                if "total_actions_per_week" in working.columns:
                    working["total_actions_per_week"] = 0.0
                    for action_name in self.action_features:
                        weekly_col = f"{action_name}_per_week"
                        if weekly_col in working.columns:
                            working["total_actions_per_week"] += pd.to_numeric(
                                working[weekly_col], errors="coerce"
                            ).fillna(0.0)
                for col in numeric_cols:
                    if col not in working.columns:
                        if col in weekly_names and col.endswith("_per_week"):
                            action_name = col.replace("_per_week", "")
                            working[col] = actions.get(action_name, 0) / max(avg_weeks_open, 1.0)
                        else:
                            working[col] = 0
                        print(f"DEBUG - Added missing column: {col}")
                mean_map = {}
                if hasattr(scaler, "mean_") and hasattr(scaler, "feature_names_in_"):
                    mean_map = dict(zip(scaler.feature_names_in_, scaler.mean_))
                for col in numeric_cols:
                    working[col] = pd.to_numeric(working[col], errors="coerce")
                    if working[col].isna().any():
                        working[col] = working[col].fillna(mean_map.get(col, 0))
                if "Debtor Age" in numeric_cols and "Debtor Age" in working.columns:
                    da = working["Debtor Age"]
                    if da.abs().max() > 1e10:
                        try:
                            dt_series = pd.to_datetime(da, unit="ns", errors="coerce")
                            today = pd.Timestamp.now()
                            ages_years = (today - dt_series).dt.days / 365.25
                            working["Debtor Age"] = ages_years.clip(0, 120).fillna(0)
                        except Exception:
                            working["Debtor Age"] = da.clip(0, 120).replace([np.nan, -np.inf, np.inf], 0)

            preds = predict_paid_values_for_dataframe(
                working, config, encoder, scaler, target_scaler, bundle=bundle
            )
            case_value_col = "Case Value"

            # Soft portfolio-level diagnostic only (no scaling of predictions):
            # compare total predicted to historical Paid/Case ratio statistics
            # and log when the simulation is far outside the training envelope.
            if case_value_col in base_df.columns:
                total_cv = float(
                    pd.to_numeric(base_df[case_value_col], errors="coerce").fillna(0).sum()
                )
                if total_cv > 0 and preds.sum() > 0:
                    ratio_cfg = insights.get("paid_to_case_ratio") or {}
                    p90_ratio = float(ratio_cfg.get("p90", 0.6))
                    sanity_cfg = config.get("prediction_sanity", {}) if isinstance(config, dict) else {}
                    warn_factor = float(sanity_cfg.get("portfolio_ratio_warn", 2.0))
                    hard_floor = float(sanity_cfg.get("portfolio_ratio_min_warn", 1.0))
                    max_reasonable_ratio = max(hard_floor, p90_ratio * warn_factor)
                    realised_ratio = preds.sum() / max(total_cv, 1e-6)
                    if realised_ratio > max_reasonable_ratio:
                        print(
                            "WARNING: Portfolio BRUT prediction exceeds configured "
                            f"sanity envelope (ratio={realised_ratio:.2f}, "
                            f"p90={p90_ratio:.2f}, max_reasonable={max_reasonable_ratio:.2f})."
                        )

            cap_note = (
                "cap active"
                if is_prediction_cap_active(config)
                else f"cap off (USE_PAID_PREDICTION_CAP={USE_PAID_PREDICTION_CAP})"
            )
            print(f"DEBUG - Final preds sum ({cap_note}): {preds.sum():.2f}, count: {len(preds)}")
            return preds

        def resolve_percent_change_for_simulation() -> float:
            manual_pct = self._get_percent_change()
            target_total = self._get_target_paid_value()
            if target_total is None or target_total <= 0:
                self._update_computed_percent_label(None)
                return manual_pct

            portfolio_weeks = self._avg_weeks_open_from_df(base_df)
            probe_baseline = (
                self._compute_baseline_actions_from_df(base_df)
                if baseline_mode == "Default"
                else global_baseline_actions
            )
            probe_levels = compute_effort_levels(
                config_for_effort,
                probe_baseline,
                manual_pct,
                baseline_is_weekly=True,
                avg_weeks_open=portfolio_weeks,
            )
            medium_probe = {
                action: int(probe_levels["Medium"].get(action, 0))
                for action in self.action_features
            }
            probe_preds = predict_for_actions_df(base_df, medium_probe)
            medium_predicted = float(np.sum(probe_preds))
            if medium_predicted <= 0:
                self._update_computed_percent_label(None)
                return manual_pct

            derived_pct = self._pct_from_target(medium_predicted, target_total)
            self._update_computed_percent_label(derived_pct)
            return derived_pct

        base_df = self._ensure_weekly_rate_columns(df.copy())
        base_df = base_df.drop(columns=[cols_cfg["target_column"]], errors="ignore")

        print(f"DEBUG - Input dataframe shape: {base_df.shape}")
        if base_df.empty:
            print("ERROR - DataFrame is empty!")
            return df

        client_col = cols_cfg.get("categorical_features", ["Client"])[0]
        if client_col in base_df.columns:
            grouped = list(base_df.groupby(base_df[client_col].astype(str).str.strip(), sort=False))
        else:
            grouped = [("Full Portfolio", base_df)]

        pct_change = resolve_percent_change_for_simulation()

        result_df = df.copy()
        result_df["Pred_Low"] = 0.0
        result_df["Pred_Medium"] = 0.0
        result_df["Pred_High"] = 0.0
        result_df["Pred_Suggested"] = 0.0
        self._effort_levels_by_client = {}
        self._baseline_weekly_by_client = {}
        fallback_effective_levels: Optional[Dict[str, Dict[str, int]]] = None
        n_groups = len(grouped)

        for i, (client_name, client_df) in enumerate(grouped, start=1):
            normalized_client = self._normalized_client_key(client_name)
            idx = client_df.index
            progress = 0.35 + 0.5 * (i - 1) / max(n_groups, 1)
            self._update_progress(progress, f"Simulating client {i}/{n_groups}: {client_name}")

            if baseline_mode == "Custom":
                working_baseline = dict(global_baseline_actions)
            elif baseline_mode == "Default":
                working_baseline = self._compute_baseline_actions_from_df(client_df)
            else:
                working_baseline = dict(global_baseline_actions)
            client_weeks = self._avg_weeks_open_from_df(client_df)
            self._baseline_weekly_by_client[normalized_client] = dict(working_baseline)
            effort_levels = compute_effort_levels(
                config_for_effort,
                working_baseline,
                pct_change,
                baseline_is_weekly=True,
                avg_weeks_open=client_weeks,
            )
            constrained_levels = {
                level: constrain_actions_for_client(level_actions, normalized_client)
                for level, level_actions in effort_levels.items()
            }
            # If client constraints collapse levels (e.g. Letters forced to 0),
            # keep at least one-step differentiation between Low/Medium/High.
            preferred_actions = [a for a in ["Calls", "SMS", "Emails", "Letters"] if a in self.action_features]
            if constrained_levels["High"] == constrained_levels["Medium"]:
                for action in preferred_actions:
                    if normalized_client in actions_never_by_client.get(action, set()):
                        continue
                    constrained_levels["High"][action] = int(constrained_levels["Medium"].get(action, 0)) + 1
                    break
            if constrained_levels["Low"] == constrained_levels["Medium"]:
                for action in preferred_actions:
                    if normalized_client in actions_never_by_client.get(action, set()):
                        continue
                    current = int(constrained_levels["Medium"].get(action, 0))
                    if current > 0:
                        constrained_levels["Low"][action] = current - 1
                        break
            suggested_actions = resolve_suggested_actions(normalized_client, constrained_levels["Medium"])
            constrained_levels["Suggested"] = constrain_actions_for_client(suggested_actions, normalized_client)

            preds_low = predict_for_actions_df(client_df, constrained_levels["Low"])
            preds_med = predict_for_actions_df(client_df, constrained_levels["Medium"])
            preds_high = predict_for_actions_df(client_df, constrained_levels["High"])
            preds_suggested = predict_for_actions_df(client_df, constrained_levels["Suggested"])

            result_df.loc[idx, "Pred_Low"] = preds_low
            result_df.loc[idx, "Pred_Medium"] = preds_med
            result_df.loc[idx, "Pred_High"] = preds_high
            result_df.loc[idx, "Pred_Suggested"] = preds_suggested
            self._effort_levels_by_client[normalized_client] = constrained_levels
            if fallback_effective_levels is None:
                fallback_effective_levels = constrained_levels

        if fallback_effective_levels is None:
            fallback_effective_levels = {
                "Low": {a: 0 for a in self.action_features},
                "Medium": {a: 0 for a in self.action_features},
                "High": {a: 0 for a in self.action_features},
                "Suggested": {a: 0 for a in self.action_features},
            }
        self.effort_levels = dict(fallback_effective_levels)
        self._effective_effort_levels = dict(self.effort_levels)

        # Validation: Check Case Value column exists and log comparison
        case_value_col = "Case Value"
        if case_value_col in df.columns:
            case_value_total = float(df[case_value_col].sum())
            print(f"DEBUG Validation - Total Case Value: {case_value_total:,.2f} EUR")
        else:
            print(f"WARNING Validation - Case Value column '{case_value_col}' not found in input dataframe")

        # Check for non-monotonic predictions (Low > Med or Med > High)
        total_low = float(result_df["Pred_Low"].sum())
        total_med = float(result_df["Pred_Medium"].sum())
        total_high = float(result_df["Pred_High"].sum())
        
        # Log prediction totals for validation
        print(f"DEBUG Validation - Prediction totals: Low={total_low:,.2f}, Medium={total_med:,.2f}, High={total_high:,.2f} EUR")
        if case_value_col in df.columns and case_value_total > 0:
            print(
                "DEBUG Validation - Predictions vs Case Value: "
                f"Low={total_low/case_value_total*100:.1f}%, "
                f"Medium={total_med/case_value_total*100:.1f}%, "
                f"High={total_high/case_value_total*100:.1f}%"
            )

        if total_med < total_low or total_high < total_med:
            print(
                "INFO: Non-monotonic totals observed (raw model output kept as-is): "
                f"Low={total_low:.2f}, Medium={total_med:.2f}, High={total_high:.2f}"
            )

        return result_df

    def _render_chart_placeholder(self, message: str) -> None:
        if self.ax is None or self.canvas is None:
            return
        self.ax.clear()
        self.ax.set_title(message, fontsize=11)
        self.ax.set_xlabel("")
        self.ax.set_ylabel("Value (EUR)")
        self.fig.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.12)
        self._hover_bars = []
        self._hover_meta = []
        self.canvas.draw_idle()

    def _update_chart_toolbar_visibility(self, show_drill: bool = False) -> None:
        if not hasattr(self, "effort_toolbar_inner"):
            return
        show_effort = self._chart_view == "monthly"
        show_row = show_effort or show_drill
        if show_effort:
            self.effort_toolbar_inner.pack(side="left", fill="x", expand=True)
        else:
            self.effort_toolbar_inner.pack_forget()
        if hasattr(self, "chart_toolbar_frame"):
            if show_row:
                self.chart_toolbar_frame.grid(
                    row=0, column=0, sticky="ew", padx=5, pady=(5, 0)
                )
            else:
                self.chart_toolbar_frame.grid_remove()

    def _reset_chart_view_state(self) -> None:
        """Return to portfolio view after upload or new simulation."""
        self._chart_view = "portfolio"
        self._effort_selection_var.set("Medium")
        if hasattr(self, "effort_segmented"):
            self.effort_segmented.set("Medium")
        self._update_drill_button_state()

    def _on_effort_segment_changed(self, value: str) -> None:
        self._effort_selection_var.set(value)
        self._refresh_chart()

    def _enter_monthly_view(self) -> None:
        self._chart_view = "monthly"
        self._update_drill_button_state()
        self._refresh_chart()

    def _enter_portfolio_view(self) -> None:
        self._chart_view = "portfolio"
        self._update_drill_button_state()
        self._refresh_chart()

    def _update_drill_button_state(self) -> None:
        if not hasattr(self, "drill_through_btn"):
            return
        if self._chart_view == "monthly":
            self.drill_through_btn.pack_forget()
            self.back_portfolio_btn.pack(side="left")
            self._update_chart_toolbar_visibility(show_drill=True)
            return
        self.back_portfolio_btn.pack_forget()
        active_key = self._resolve_active_client_key()
        show_drill = getattr(self, "_results_df", None) is not None
        if show_drill:
            self.drill_through_btn.pack(side="left", padx=(0, 6))
        else:
            self.drill_through_btn.pack_forget()
        self._update_chart_toolbar_visibility(show_drill=show_drill)

    def _resolve_active_client_key(self) -> Optional[str]:
        df = getattr(self, "_results_df", None)
        if df is None or df.empty:
            return None
        cols_cfg = self.config_dict["columns"]
        client_col = cols_cfg["categorical_features"][0]
        if client_col not in df.columns:
            return None

        client_filter = self.client_filter_var.get()
        if not client_filter.startswith("Full Portfolio"):
            client_name = (
                client_filter.rsplit(" (", 1)[0] if " (" in client_filter else client_filter
            )
            return self._normalized_client_key(client_name)

        sliced = self._slice_results_df(
            df,
            client_filter,
            self.import_filter_var.get(),
        )
        if sliced.empty:
            return None
        unique_clients = sliced[client_col].astype(str).str.strip().unique()
        if len(unique_clients) != 1:
            return None
        return self._normalized_client_key(unique_clients[0])

    def _slice_results_df(
        self,
        df: pd.DataFrame,
        client_filter: str,
        import_filter: str,
    ) -> pd.DataFrame:
        cols_cfg = self.config_dict["columns"]
        client_col = cols_cfg["categorical_features"][0]
        out = df
        if not client_filter.startswith("Full Portfolio"):
            client_name = (
                client_filter.rsplit(" (", 1)[0] if " (" in client_filter else client_filter
            )
            out = out[out[client_col].astype(str) == client_name]
        if not import_filter.startswith("All Imports") and "Import Name" in out.columns:
            import_name = (
                import_filter.rsplit(" (", 1)[0] if " (" in import_filter else import_filter
            )
            out = out[out["Import Name"].astype(str) == import_name]
        return out

    def _get_payout_months_for_segment(self, df: pd.DataFrame) -> int:
        return resolve_payout_months_for_segment(df, self.config_dict)

    def _refresh_chart(self) -> None:
        df = getattr(self, "_results_df", None)
        client_filter = self.client_filter_var.get()
        import_filter = self.import_filter_var.get()
        if df is None:
            if self.uploaded_df is not None and not self.uploaded_df.empty:
                self._render_chart_placeholder(
                    "Portfolio uploaded — run Strategy Simulation to view the chart."
                )
            else:
                self._render_chart_placeholder("No data loaded")
            self._update_drill_button_state()
            return

        sliced = self._slice_results_df(df, client_filter, import_filter)
        if sliced.empty:
            self._render_chart_placeholder(
                "No cases match the current filter. Adjust client or import filter."
            )
            self._update_drill_button_state()
            return

        if self._chart_view == "portfolio":
            self._render_portfolio_graph(df, client_filter, import_filter)
        else:
            self._render_monthly_breakdown(df, client_filter, import_filter)
        self._update_drill_button_state()

    def _render_portfolio_graph(
        self,
        df: pd.DataFrame,
        client_filter: str,
        import_filter: str,
    ) -> None:
        if self.fig is None or self.ax is None or self.canvas is None:
            return

        df = self._slice_results_df(df, client_filter, import_filter)
        n_cases = len(df)
        level_names = ["Low", "Medium", "High", "Suggested"]
        pred_columns = ["Pred_Low", "Pred_Medium", "Pred_High", "Pred_Suggested"]

        baseline_paid: Optional[float] = None
        if "Paid Value" in df.columns:
            baseline_paid = float(pd.to_numeric(df["Paid Value"], errors="coerce").fillna(0).sum())
        segment_weeks = self._avg_weeks_open_from_df(df)
        baseline_weekly = self._compute_baseline_actions_from_df(df)
        if not client_filter.startswith("Full Portfolio"):
            selected_client_name = (
                client_filter.rsplit(" (", 1)[0] if " (" in client_filter else client_filter
            )
            cached = self._baseline_weekly_by_client.get(
                self._normalized_client_key(selected_client_name)
            )
            if cached:
                baseline_weekly = cached
        baseline_actions_per_case: Dict[str, float] = {
            a: baseline_weekly.get(a, 0.0) * segment_weeks for a in self.action_features
        }

        selected_client_name = (
            client_filter.rsplit(" (", 1)[0] if " (" in client_filter else client_filter
        )
        selected_client_key = self._normalized_client_key(selected_client_name)
        source_levels = (
            self._effort_levels_by_client.get(selected_client_key, self.effort_levels)
            if not client_filter.startswith("Full Portfolio")
            else self.effort_levels
        )

        effort_labels: List[str] = []
        brut_totals: List[float] = []
        label_weeks = self._avg_weeks_open_from_df(df)
        for level_name, pred_col in zip(level_names, pred_columns):
            raw_actions = source_levels.get(level_name, {})
            actions = self._apply_client_action_constraints(raw_actions, client_filter)
            self._effective_effort_levels[level_name] = actions
            if level_name == "Suggested":
                action_str = self._format_weekly_rates_label(baseline_weekly)
                effort_labels.append(f"Historical\n({action_str})")
            else:
                action_str = self._format_actions_per_week_label(
                    actions, label_weeks, round_up_fractional=True
                )
                effort_labels.append(f"{level_name}\n({action_str})")
            if pred_col in df.columns:
                brut_totals.append(float(df[pred_col].sum()))
            else:
                brut_totals.append(0.0)

        self.ax.clear()
        x = np.arange(len(level_names))
        bar_colors = ["#2196F3", "#2196F3", "#2196F3", "#9E9E9E"]
        edge_colors = ["#2196F3", "#2196F3", "#2196F3", "#666666"]
        # Single BarContainer required for ax.bar_label (not a list of patches).
        bars_brut = self.ax.bar(
            x,
            brut_totals,
            width=0.55,
            color=bar_colors,
            edgecolor=edge_colors,
            label="BRUT (predicted paid)",
        )
        for patch, level_name in zip(bars_brut, level_names):
            if level_name == "Suggested":
                patch.set_hatch("///")
                patch.set_label("Suggested (Historical)")

        self.ax.set_xticks(x)
        self.ax.set_xticklabels(effort_labels)
        title_suffix = client_filter if client_filter != "Full Portfolio" else "Full Portfolio"
        self.ax.set_title(
            f"Predicted paid value (BRUT) by effort level ({title_suffix})",
            fontsize=11,
        )
        self.ax.set_xlabel(
            "Effort Level — actions per week (C=Calls, L=Letters, S=SMS, E=Emails)",
            fontsize=9,
        )
        self.ax.set_ylabel("Value (EUR)", fontsize=9)
        self.ax.legend(loc="upper right", fontsize=9)

        total_case_value = 0.0
        if "Case Value" in df.columns:
            total_case_value = float(
                pd.to_numeric(df["Case Value"], errors="coerce").fillna(0).sum()
            )

        bar_value_labels: List[str] = []
        for level_name, brut_val in zip(level_names, brut_totals):
            if level_name == "Suggested":
                bar_value_labels.append(f"{brut_val:,.0f}")
            elif total_case_value > 0:
                recovery_pct = brut_val / total_case_value * 100.0
                bar_value_labels.append(f"{brut_val:,.0f}\n({recovery_pct:.1f}%)")
            else:
                bar_value_labels.append(f"{brut_val:,.0f}")

        all_values = brut_totals
        if all_values:
            y_min = min(all_values)
            y_max = max(all_values)
            y_range = y_max - y_min if y_max != y_min else abs(y_max) if y_max != 0 else 1
            y_padding = max(y_range * 0.1, abs(y_min) * 0.05 if y_min < 0 else 0)
            extra_top = y_range * 0.08 if any("\n" in label for label in bar_value_labels) else 0
            self.ax.set_ylim(
                bottom=max(y_min - y_padding, 0),
                top=y_max + y_range * 0.1 + extra_top,
            )

        self.ax.bar_label(bars_brut, labels=bar_value_labels, padding=3, fontsize=7)
        self.ax.grid(axis="y", linestyle="--", alpha=0.3)
        self.ax.tick_params(axis="x", labelsize=7)
        self.fig.tight_layout(pad=1.0)

        self._update_hover_meta_grouped(
            list(bars_brut),
            brut_totals,
            level_names,
            n_cases,
            baseline_paid=baseline_paid,
            baseline_actions_per_case=baseline_actions_per_case,
        )
        self.canvas.draw_idle()

    def _render_monthly_breakdown(
        self,
        df: pd.DataFrame,
        client_filter: str,
        import_filter: str,
    ) -> None:
        if self.fig is None or self.ax is None or self.canvas is None:
            return

        df = self._slice_results_df(df, client_filter, import_filter)
        effort_level = self._effort_selection_var.get()
        payout_months = self._get_payout_months_for_segment(df)

        try:
            breakdown_df = compute_monthly_breakdown(
                df,
                effort_level,
                self.config_dict,
                payout_months,
            )
        except ValueError as exc:
            self.ax.clear()
            self.ax.set_title(f"Monthly breakdown unavailable: {exc}")
            self.canvas.draw_idle()
            return

        n_months = len(breakdown_df)
        x = np.arange(n_months)
        width = 0.35

        total_case_value = 0.0
        if "Case Value" in df.columns:
            total_case_value = float(
                pd.to_numeric(df["Case Value"], errors="coerce").fillna(0).sum()
            )

        self.ax.clear()
        bars_payout = self.ax.bar(
            x - width / 2,
            breakdown_df["monthly_payout"],
            width,
            label="Predicted monthly payout",
            color="#2196F3",
        )
        bars_remaining = self.ax.bar(
            x + width / 2,
            breakdown_df["remaining_total"],
            width,
            label="Remaining portfolio total",
            color="#FF9800",
        )
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(
            [f"Month {int(m)}" for m in breakdown_df["month"]]
        )
        title_suffix = client_filter if client_filter != "Full Portfolio" else "Full Portfolio"
        self.ax.set_title(
            f"Monthly payout schedule — {effort_level} effort ({title_suffix}, {n_months} mo)",
            fontsize=11,
        )
        self.ax.set_ylabel("Value (EUR)", fontsize=9)
        self.ax.legend(loc="upper right", fontsize=9)
        self.ax.grid(axis="y", linestyle="--", alpha=0.3)

        payout_labels: List[str] = []
        remaining_labels: List[str] = []
        for payout_val, remaining_val in zip(
            breakdown_df["monthly_payout"], breakdown_df["remaining_total"]
        ):
            payout_f = float(payout_val)
            remaining_f = float(remaining_val)
            if total_case_value > 0:
                payout_labels.append(
                    f"{payout_f:,.0f}\n({payout_f / total_case_value * 100:.1f}%)"
                )
                remaining_labels.append(
                    f"{remaining_f:,.0f}\n({remaining_f / total_case_value * 100:.1f}%)"
                )
            else:
                payout_labels.append(f"{payout_f:,.0f}")
                remaining_labels.append(f"{remaining_f:,.0f}")

        y_max = max(
            float(breakdown_df["monthly_payout"].max()),
            float(breakdown_df["remaining_total"].max()),
            1.0,
        )
        extra_top = y_max * 0.15 if any("\n" in label for label in payout_labels) else y_max * 0.08
        self.ax.set_ylim(bottom=0, top=y_max + extra_top)
        self.ax.bar_label(bars_payout, labels=payout_labels, padding=3, fontsize=6)
        self.ax.bar_label(bars_remaining, labels=remaining_labels, padding=3, fontsize=6)

        self.fig.tight_layout(pad=1.0)

        self._hover_bars = []
        self._hover_meta = []
        self.canvas.draw_idle()

    def _update_graph_for_filter(self, value: str) -> None:
        df = getattr(self, "_results_df", None)
        if df is None:
            return
        self._refresh_chart()

    def _normalize_action_costs(self, costs: Dict[str, float]) -> Dict[str, float]:
        defaults = {"Calls": 0.2, "Letters": 1.22, "SMS": 0.03, "Emails": 0.0}
        normalized = {}
        for action in self.action_features:
            value = costs.get(action, defaults.get(action, 0.0))
            try:
                normalized[action] = float(value)
            except (TypeError, ValueError):
                normalized[action] = 0.0
        return normalized

    def _actions_grid_median(self) -> Dict[str, float]:
        grid = self.config_dict.get("actions_grid", {})
        medians = {}
        for action in self.action_features:
            values = grid.get(action)
            medians[action] = float(np.median(values)) if values else 0.0
        return medians

    def _compute_training_baseline(self) -> Dict[str, float]:
        if self._training_baseline_actions is not None:
            return self._training_baseline_actions

        data_cfg = self.config_dict.get("data", {})
        train_path = data_cfg.get("train_path")

        if train_path and os.path.exists(train_path):
            try:
                df = read_table_file(train_path)
                weekly_cols = {
                    action: f"{action}_per_week" for action in self.action_features
                }
                date_cfg = self.config_dict.get("columns", {}).get("date_features", {})
                import_col = date_cfg.get("import_date", "Import date")
                end_col = date_cfg.get("end_date", "Date End")
                if import_col in df.columns and end_col in df.columns:
                    df = compute_weekly_rates(df, self.config_dict)

                if all(col in df.columns for col in weekly_cols.values()):
                    self._training_baseline_actions = {
                        action: float(pd.to_numeric(df[weekly_col], errors="coerce").mean())
                        for action, weekly_col in weekly_cols.items()
                    }
                    return self._training_baseline_actions
            except Exception:
                pass

        self._training_baseline_actions = self._actions_grid_median()
        return self._training_baseline_actions

    def _normalized_client_key(self, value: object) -> str:
        return str(value).strip().casefold()

    def _compute_client_baseline_actions(self, df: pd.DataFrame) -> Optional[Dict[str, float]]:
        cols_cfg = self.config_dict.get("columns", {})
        cat_features = cols_cfg.get("categorical_features", [])
        if not cat_features:
            return None
        client_col = cat_features[0]
        if client_col not in df.columns:
            return None

        client_series = df[client_col].astype(str).str.strip()
        if client_series.empty:
            return None
        dominant_client = client_series.mode(dropna=True)
        if dominant_client.empty:
            return None
        client_name = str(dominant_client.iloc[0]).strip()
        client_rows = df[client_series == client_name]
        if client_rows.empty:
            return None

        return self._compute_baseline_actions_from_df(client_rows)

    def _apply_client_action_constraints(
        self,
        actions: Dict[str, int],
        client_filter: str,
    ) -> Dict[str, int]:
        adjusted = {action: int(actions.get(action, 0)) for action in self.action_features}
        if not self._actions_never_by_client:
            return adjusted
        if client_filter.startswith("Full Portfolio"):
            return adjusted

        client_name = client_filter.rsplit(" (", 1)[0] if " (" in client_filter else client_filter
        normalized_client = self._normalized_client_key(client_name)
        for action, clients_never in self._actions_never_by_client.items():
            if action in adjusted and normalized_client in clients_never:
                adjusted[action] = 0
        return adjusted

    def _toggle_section(self, section: str) -> None:
        """Toggle visibility of collapsible sections (settings or costs)."""
        if section == "settings":
            expanded = self.settings_expanded.get()
            self.settings_expanded.set(not expanded)
            if not expanded:
                self.settings_content_frame.pack(fill="x", padx=5, pady=(5, 8))
                self.settings_toggle_btn.configure(text="▼ Effort Settings")
            else:
                self.settings_content_frame.pack_forget()
                self.settings_toggle_btn.configure(text="▶ Effort Settings")
        elif section == "costs":
            expanded = self.costs_expanded.get()
            self.costs_expanded.set(not expanded)
            if not expanded:
                self.costs_content_frame.pack(fill="x", padx=5, pady=(5, 8))
                self.costs_toggle_btn.configure(text="▼ Action Costs (EUR / action)")
            else:
                self.costs_content_frame.pack_forget()
                self.costs_toggle_btn.configure(text="▶ Action Costs (EUR / action)")
        elif section == "lengths":
            expanded = self.lengths_expanded.get()
            self.lengths_expanded.set(not expanded)
            if not expanded:
                self.lengths_content_frame.pack(fill="x", padx=5, pady=(5, 8))
                self.lengths_toggle_btn.configure(text="▼ Portfolio Lengths")
            else:
                self.lengths_content_frame.pack_forget()
                self.lengths_toggle_btn.configure(text="▶ Portfolio Lengths")

    def _portfolio_length_clients_for_ui(self) -> List[str]:
        clients = set(list_portfolio_length_clients(self.config_dict))
        df = self.uploaded_df
        if df is not None and not df.empty:
            client_col = self.config_dict["columns"]["categorical_features"][0]
            if client_col in df.columns:
                for name in df[client_col].dropna().astype(str).str.strip().unique():
                    clients.add(name)
        return sorted(clients, key=lambda s: s.casefold())

    def _lookup_portfolio_length_value(self, client: str) -> object:
        pl_cfg = portfolio_lengths_config(self.config_dict)
        key = self._normalized_client_key(client)
        for name, value in (pl_cfg.get("by_client") or {}).items():
            if self._normalized_client_key(name) == key:
                return value
        return None

    def _populate_portfolio_length_fields(self) -> None:
        if not hasattr(self, "portfolio_length_scroll"):
            return
        for child in self.portfolio_length_scroll.winfo_children():
            child.destroy()
        self.portfolio_length_entries.clear()

        for client in self._portfolio_length_clients_for_ui():
            row = ctk.CTkFrame(self.portfolio_length_scroll, fg_color="transparent")
            row.pack(fill="x", pady=1)
            label_text = client if len(client) <= 22 else client[:19] + "..."
            ctk.CTkLabel(row, text=label_text, width=130, anchor="w").pack(side="left")
            entry = ctk.CTkEntry(row, width=72)
            value = self._lookup_portfolio_length_value(client)
            if value is not None:
                entry.insert(0, format_portfolio_length_for_ui(value))
            entry.pack(side="right", padx=2)
            self.portfolio_length_entries[client] = entry

    def _save_portfolio_lengths(self) -> None:
        by_client: Dict[str, object] = {}
        for client, entry in self.portfolio_length_entries.items():
            text = entry.get()
            try:
                by_client[client] = parse_portfolio_length_input(text)
            except ValueError as exc:
                messagebox.showerror(
                    "Invalid portfolio length",
                    f"{client}: {exc}",
                )
                return

        config = load_config()
        pl_section = dict(config.get("portfolio_lengths") or {})
        pl_section["by_client"] = by_client
        if "no_limit_months" not in pl_section:
            pl_section["no_limit_months"] = portfolio_lengths_config(config)["no_limit_months"]
        if "default_months" not in pl_section:
            pl_section["default_months"] = portfolio_lengths_config(config)["default_months"]
        config["portfolio_lengths"] = pl_section

        try:
            with open("config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
        except Exception as exc:
            messagebox.showerror("Save error", f"Failed to save config.json:\n{exc}")
            return

        self.config_dict = config
        messagebox.showinfo("Saved", "Portfolio lengths saved to config.json.")
        if getattr(self, "_results_df", None) is not None and self._chart_view == "monthly":
            self._refresh_chart()

    def _toggle_baseline_mode(self, value: str) -> None:
        state = "normal" if value == "Custom" else "disabled"
        for entry in self.custom_baseline_entries.values():
            entry.configure(state=state)
        if value == "Default":
            self._sync_baseline_display_fields()
        elif value != "Custom":
            for entry in self.custom_baseline_entries.values():
                entry.delete(0, "end")

    def _parse_float_entry(
        self, entry: ctk.CTkEntry, default: Optional[float] = None
    ) -> Optional[float]:
        text = entry.get().strip()
        if not text:
            return default
        try:
            return float(text.replace(",", "."))
        except ValueError:
            return default

    def _get_baseline_actions(self) -> Dict[str, float]:
        mode = self.baseline_mode_var.get()
        if mode == "Custom":
            return {
                action: self._parse_float_entry(entry, 0.0) or 0.0
                for action, entry in self.custom_baseline_entries.items()
            }
        if mode == "Default":
            if self._portfolio_baseline_actions is not None:
                return dict(self._portfolio_baseline_actions)
            if self.uploaded_df is not None and not self.uploaded_df.empty:
                return self._compute_baseline_actions_from_df(self.uploaded_df)
            return dict(self._compute_training_baseline())
        return dict(self._compute_training_baseline())

    def _avg_weeks_open_from_df(self, df: pd.DataFrame) -> float:
        """Mean case duration in weeks (valid Import/End rows only)."""
        weeks = case_duration_weeks_series(df, self.config_dict)
        if weeks.notna().any():
            return max(float(weeks.mean(skipna=True)), 1.0 / 7.0)
        return 1.0

    @staticmethod
    def _format_actions_per_week_label(
        actions: Dict[str, int],
        avg_weeks_open: float,
        *,
        round_up_fractional: bool = False,
    ) -> str:
        """Format per-case action totals as per-week rates for chart labels."""
        weeks = max(float(avg_weeks_open), 1.0 / 7.0)
        parts: List[str] = []
        for action, total in actions.items():
            weekly = float(total) / weeks
            if weekly <= 0:
                parts.append(f"{action[0]}:0")
            elif abs(weekly - round(weekly)) < 1e-6:
                parts.append(f"{action[0]}:{int(round(weekly))}")
            elif round_up_fractional:
                parts.append(f"{action[0]}:{int(math.ceil(weekly - 1e-9))}")
            else:
                parts.append(f"{action[0]}:{weekly:.1f}")
        return ", ".join(parts)

    @staticmethod
    def _format_weekly_rates_label(weekly_rates: Dict[str, float]) -> str:
        """Format already-computed per-week action rates for chart labels."""
        parts: List[str] = []
        for action, rate in weekly_rates.items():
            rate_f = float(rate)
            if abs(rate_f - round(rate_f)) < 0.05:
                parts.append(f"{action[0]}:{int(round(rate_f))}")
            else:
                parts.append(f"{action[0]}:{rate_f:.1f}")
        return ", ".join(parts)

    def _ensure_weekly_rate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive *_per_week columns from action totals and case dates.
        Never requires these columns to exist in uploaded files.
        """
        date_cfg = self.config_dict.get("columns", {}).get("date_features", {})
        import_col = date_cfg.get("import_date", "Import date")
        end_col = date_cfg.get("end_date", "Date End")
        if import_col in df.columns and end_col in df.columns:
            return compute_weekly_rates(df.copy(), self.config_dict)
        return df

    def _compute_baseline_actions_from_df(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Mean per-case weekly action rate for rows in df (one client slice or filter).

        For each case: actions / weeks_active, then average across cases with valid dates.
        """
        if df.empty:
            return {action: 0.0 for action in self.action_features}

        working = self._ensure_weekly_rate_columns(df)
        weekly_cols = {action: f"{action}_per_week" for action in self.action_features}

        if all(col in working.columns for col in weekly_cols.values()):
            baseline: Dict[str, float] = {}
            for action, weekly_col in weekly_cols.items():
                series = pd.to_numeric(working[weekly_col], errors="coerce")
                baseline[action] = float(series.mean(skipna=True)) if series.notna().any() else 0.0
            return baseline

        return {action: 0.0 for action in self.action_features}

    def _apply_default_effort_settings(self, df: pd.DataFrame) -> None:
        """After upload: portfolio-average baseline, 100% spread, target = sum(Case Value)."""
        self._portfolio_baseline_actions = self._compute_baseline_actions_from_df(df)
        self.percent_change_var.set("100")

        case_value_col = "Case Value"
        if case_value_col in df.columns:
            total_cv = float(
                pd.to_numeric(df[case_value_col], errors="coerce").fillna(0).sum()
            )
            if total_cv == int(total_cv):
                self.target_paid_value_var.set(str(int(total_cv)))
            else:
                self.target_paid_value_var.set(f"{total_cv:.2f}")
        else:
            self.target_paid_value_var.set("")

        if self.baseline_mode_var.get() == "Default":
            self._sync_baseline_display_fields()

    def _sync_baseline_display_fields(self) -> None:
        """Show portfolio baseline averages in the action fields (read-only unless Custom)."""
        if self.baseline_mode_var.get() != "Default":
            return
        baseline = self._get_baseline_actions()
        for action, entry in self.custom_baseline_entries.items():
            entry.configure(state="normal")
            entry.delete(0, "end")
            value = baseline.get(action, 0.0)
            if value == int(value):
                entry.insert(0, str(int(value)))
            else:
                entry.insert(0, f"{value:.4g}")
            entry.configure(state="disabled")

    def _get_percent_change(self) -> float:
        """
        Return the user‑configured percentage change as a fraction.

        The UI accepts values like `20` for 20%. To avoid accidental
        extreme values (e.g. typing `2000` instead of `20`), clamp the
        absolute percentage to a reasonable range.
        """
        percent_value = self._parse_float_entry(self.percent_entry, 0.0) or 0.0
        # Clamp to [-300%, 300%] to avoid nonsensical action levels.
        if percent_value > 300:
            percent_value = 300.0
        elif percent_value < -300:
            percent_value = -300.0
        return percent_value / 100.0

    def _get_target_paid_value(self) -> Optional[float]:
        value = self._parse_float_entry(self.target_value_entry, None)
        if value is None or value <= 0:
            return None
        return value

    def _pct_from_target(self, baseline_total: float, target_total: float) -> float:
        """
        Compute required fractional change to reach target_total
        starting from baseline_total, with safety clamping to avoid
        exploding action levels when the target is unrealistically far
        from the baseline.
        """
        if baseline_total <= 0:
            return 0.0
        pct = (target_total - baseline_total) / baseline_total
        # Clamp to [-300%, 300%] in fractional form.
        if pct > 3.0:
            pct = 3.0
        elif pct < -3.0:
            pct = -3.0
        return pct

    def _update_computed_percent_label(self, pct_change: Optional[float]) -> None:
        if pct_change is None:
            self.computed_percent_label.configure(text="Computed % from target: -")
        else:
            self.computed_percent_label.configure(
                text=f"Computed % from target: {pct_change * 100:.1f}%"
            )

    def _save_action_costs(self) -> None:
        updated_costs = {}
        for action, entry in self.cost_entries.items():
            value = self._parse_float_entry(entry, None)
            if value is None or value < 0:
                messagebox.showerror(
                    "Invalid cost",
                    f"Please enter a valid non-negative number for {action}.",
                )
                return
            updated_costs[action] = float(value)

        config = load_config()
        config["action_costs"] = updated_costs
        try:
            with open("config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
        except Exception as exc:
            messagebox.showerror("Save error", f"Failed to save config.json:\n{exc}")
            return

        self.config_dict = config
        self.action_costs = updated_costs
        messagebox.showinfo("Saved", "Action costs saved to config.json.")

    def _update_hover_meta(self, bars, totals: List[float], level_names: Optional[List[str]] = None) -> None:
        self._hover_bars = bars
        self._hover_meta = []
        if level_names is None:
            level_names = ["Low", "Medium", "High", "Suggested"]
        for level, total in zip(level_names, totals):
            actions = self._effective_effort_levels.get(level, self.effort_levels.get(level, {}))
            total_cost = sum(
                actions.get(action, 0) * self.action_costs.get(action, 0.0)
                for action in self.action_features
            )
            self._hover_meta.append(
                {
                    "level": level if level != "Suggested" else "Suggested (Historical)",
                    "total": total,
                    "actions": actions,
                    "total_cost": total_cost,
                }
            )

    def _update_hover_meta_grouped(
        self,
        bars_brut,
        brut_totals: List[float],
        level_names: List[str],
        n_cases: int,
        baseline_paid: Optional[float] = None,
        baseline_actions_per_case: Optional[Dict[str, float]] = None,
    ) -> None:
        """Hover metadata for BRUT-only portfolio chart."""
        self._hover_bars = list(bars_brut)
        self._hover_meta = []
        baseline_actions_per_case = baseline_actions_per_case or {}
        for i, level in enumerate(level_names):
            actions = self._effective_effort_levels.get(level, self.effort_levels.get(level, {}))
            cost_per_case = sum(
                actions.get(action, 0) * self.action_costs.get(action, 0.0)
                for action in self.action_features
            )
            total_cost = n_cases * cost_per_case
            level_label = level if level != "Suggested" else "Suggested (Historical)"
            pct_paid: Optional[float] = None
            if baseline_paid is not None and baseline_paid != 0:
                pct_paid = (brut_totals[i] - baseline_paid) / baseline_paid * 100.0
            pct_actions: Dict[str, float] = {}
            for action in self.action_features:
                base_val = baseline_actions_per_case.get(action, 0.0)
                level_val = float(actions.get(action, 0))
                if base_val != 0:
                    pct_actions[action] = (level_val - base_val) / base_val * 100.0
                elif level_val != 0:
                    pct_actions[action] = 100.0
            self._hover_meta.append(
                {
                    "level": f"{level_label} BRUT",
                    "total": brut_totals[i],
                    "actions": actions,
                    "total_cost": total_cost,
                    "value_type": "BRUT",
                    "pct_paid": pct_paid,
                    "pct_actions": pct_actions,
                }
            )

    def _format_hover_text(self, meta: Dict) -> str:
        value_type = meta.get("value_type", "")
        if value_type:
            lines = [f"{meta['level']}"]
        else:
            lines = [f"{meta['level']} Effort"]
        lines.append(f"Value: EUR {meta['total']:,.2f}")
        pct_paid = meta.get("pct_paid")
        if pct_paid is not None:
            sign = "+" if pct_paid >= 0 else ""
            lines.append(f"Paid vs baseline: {sign}{pct_paid:.1f}%")
        pct_actions = meta.get("pct_actions") or {}
        if pct_actions:
            parts = [f"{a}: {'+' if pct_actions[a] >= 0 else ''}{pct_actions[a]:.1f}%" for a in self.action_features if a in pct_actions]
            if parts:
                lines.append("Actions vs baseline: " + ", ".join(parts))
        for action in self.action_features:
            count = meta["actions"].get(action, 0)
            cost = count * self.action_costs.get(action, 0.0)
            lines.append(f"{action}: {count} (EUR {cost:.2f} per case)")
        lines.append(f"Total Action Cost (all cases): EUR {meta['total_cost']:,.2f}")
        return "\n".join(lines)

    def _on_hover(self, event) -> None:
        if self._chart_view == "monthly":
            if self._hover_annot is not None and self._hover_annot.get_visible():
                self._hover_annot.set_visible(False)
                self.canvas.draw_idle()
            return
        if not self._hover_bars or self._hover_annot is None:
            return
        if event.inaxes != self.ax:
            if self._hover_annot.get_visible():
                self._hover_annot.set_visible(False)
                self.canvas.draw_idle()
            return
        for bar, meta in zip(self._hover_bars, self._hover_meta):
            contains, _ = bar.contains(event)
            if contains:
                self._hover_annot.xy = (bar.get_x() + bar.get_width() / 2, bar.get_height())
                self._hover_annot.set_text(self._format_hover_text(meta))
                self._hover_annot.set_visible(True)
                self.canvas.draw_idle()
                return
        if self._hover_annot.get_visible():
            self._hover_annot.set_visible(False)
            self.canvas.draw_idle()


class InsightsTab(ctk.CTkFrame):
    """Tab for displaying training data insights with histograms."""

    def __init__(self, master, auto_load: bool = True):
        super().__init__(master)
        self.pack(expand=True, fill="both")
        
        self.config_dict = load_config()
        self.action_features = list(self.config_dict["columns"].get("action_features", []))
        
        self.fig: Optional[Figure] = None
        self.ax = None
        self.canvas: Optional[FigureCanvasTkAgg] = None
        self._is_loaded = False
        
        self._build_ui()
        if auto_load:
            self.ensure_loaded()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # Header
        header = ctk.CTkLabel(
            self,
            text="Training Data Insights",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        header.grid(row=0, column=0, pady=(10, 5))

        # Main container with left (text) and right (histogram) panels
        main_container = ctk.CTkFrame(self)
        main_container.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        main_container.columnconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=2)
        main_container.rowconfigure(0, weight=1)

        # Left panel - Text insights
        left_frame = ctk.CTkFrame(main_container, corner_radius=10)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=0)
        left_frame.rowconfigure(1, weight=1)
        left_frame.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            left_frame,
            text="Statistics Summary",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, pady=(10, 5), padx=10)

        self.insights_text = ctk.CTkTextbox(
            left_frame,
            font=ctk.CTkFont(family="Consolas", size=11),
            wrap="word",
        )
        self.insights_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        refresh_btn = ctk.CTkButton(
            left_frame,
            text="Refresh Data",
            command=lambda: self._load_and_display_insights(force_refresh=True),
            fg_color="#8B0000",
            hover_color="#B22222",
        )
        refresh_btn.grid(row=2, column=0, pady=(0, 10), padx=10)

        # Right panel - Histograms
        right_frame = ctk.CTkFrame(main_container, corner_radius=10)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=0)
        right_frame.rowconfigure(1, weight=1)
        right_frame.columnconfigure(0, weight=1)

        # Histogram type selector
        selector_frame = ctk.CTkFrame(right_frame, fg_color="transparent")
        selector_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))

        ctk.CTkLabel(
            selector_frame,
            text="Histogram:",
            font=ctk.CTkFont(size=12),
        ).pack(side="left", padx=(0, 10))

        self.histogram_var = ctk.StringVar(value="Paid Value")
        histogram_options = ["Paid Value"] + self.action_features + ["All Actions"]
        self.histogram_menu = ctk.CTkOptionMenu(
            selector_frame,
            values=histogram_options,
            variable=self.histogram_var,
            command=self._update_histogram,
            width=150,
        )
        self.histogram_menu.pack(side="left")

        # Matplotlib figure for histogram
        graph_container = ctk.CTkFrame(right_frame)
        graph_container.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 10))
        graph_container.rowconfigure(0, weight=1)
        graph_container.columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.fig.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.15)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_container)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Toolbar for histogram
        toolbar = NavigationToolbar2Tk(
            self.canvas,
            graph_container,
            pack_toolbar=False,
        )
        toolbar.update()
        toolbar.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 5))

        # Store training data for histogram
        self._training_df: Optional[pd.DataFrame] = None

    def ensure_loaded(self) -> None:
        if self._is_loaded:
            return
        self._load_and_display_insights(force_refresh=False)

    def _load_and_display_insights(self, force_refresh: bool = False) -> None:
        """Load training data and display insights."""
        try:
            if force_refresh:
                self._training_df = None
            insights = get_training_data_insights(include_training_df=True)
            self._training_df = insights.get("training_df")

            # Build text summary
            text = "=== TRAINING DATA SUMMARY ===\n\n"
            text += f"Total Samples: {insights['total_samples']:,}\n\n"

            # Paid Value Stats
            pv = insights["paid_value_stats"]
            text += "PAID VALUE STATISTICS:\n"
            text += f"  Mean: {pv['mean']:,.2f} EUR\n"
            text += f"  Median: {pv['median']:,.2f} EUR\n"
            text += f"  Std Dev: {pv['std']:,.2f} EUR\n"
            text += f"  Min: {pv['min']:,.2f} EUR\n"
            text += f"  Max: {pv['max']:,.2f} EUR\n\n"

            # Action Stats (per case and per week, when available)
            text += "ACTION STATISTICS (Per Case Totals):\n"
            for action, stats in insights["action_stats"].items():
                text += f"  {action}:\n"
                text += f"    Mean: {stats['mean']:.1f}\n"
                text += f"    Median: {stats['median']:.1f}\n"
                text += f"    Range: [{stats['min']:.0f} - {stats['max']:.0f}]\n"
            text += "\n"

            per_week_stats = insights.get("action_stats_per_week") or {}
            if per_week_stats:
                text += "ACTION STATISTICS (Per Week):\n"
                for action, stats in per_week_stats.items():
                    text += f"  {action} per week:\n"
                    text += f"    Mean: {stats['mean']:.2f}\n"
                    text += f"    Median: {stats['median']:.2f}\n"
                    text += f"    Range: [{stats['min']:.2f} - {stats['max']:.2f}]\n"
                text += "\n"

            # Effort Buckets
            avg_weeks_open = insights.get("avg_weeks_open")
            text += "EFFORT LEVEL ANALYSIS:\n"
            if avg_weeks_open:
                text += f"(Based on total actions per week; avg case duration ≈ {avg_weeks_open:.1f} weeks)\n\n"
            else:
                text += "(Based on total actions per case)\n\n"

            for bucket, stats in insights["effort_buckets"].items():
                text += f"  {bucket} Effort:\n"
                text += f"    Cases: {stats['count']:,}\n"
                text += f"    Avg Paid Value: {stats['avg_paid_value']:,.2f} EUR\n"

                # Per-case totals (backwards compatible)
                actions_per_case = stats.get("avg_actions", {})
                if actions_per_case:
                    actions_str = ", ".join(
                        f"{k}={v:.1f}" for k, v in actions_per_case.items()
                    )
                    text += f"    Avg Actions per case: {actions_str}\n"

                # Per-week view (preferred when available)
                actions_per_week = stats.get("avg_actions_per_week", {})
                if actions_per_week:
                    actions_week_str = ", ".join(
                        f"{k}={v:.2f}" for k, v in actions_per_week.items()
                    )
                    text += f"    Avg Actions per week: {actions_week_str}\n"

                text += "\n"

            self.insights_text.configure(state="normal")
            self.insights_text.delete("1.0", "end")
            self.insights_text.insert("1.0", text)
            self.insights_text.configure(state="disabled")

            # Update histogram
            self._update_histogram(self.histogram_var.get())
            self._is_loaded = True

        except Exception as e:
            self.insights_text.configure(state="normal")
            self.insights_text.delete("1.0", "end")
            self.insights_text.insert(
                "1.0",
                f"Could not load training insights:\n{e}\n\n"
                "Make sure training data exists at the configured path."
            )
            self.insights_text.configure(state="disabled")
            self._is_loaded = False

    def _update_histogram(self, selection: str) -> None:
        """Update the histogram based on the selected data."""
        if self.fig is None or self.ax is None or self.canvas is None:
            return

        if self._training_df is None:
            self.ax.clear()
            self.ax.set_title("No training data loaded")
            self.canvas.draw_idle()
            return

        self.ax.clear()

        if selection == "Paid Value":
            target_col = self.config_dict["columns"]["target_column"]
            if target_col in self._training_df.columns:
                data = self._training_df[target_col].dropna()
                # Filter out extreme outliers for better visualization
                q99 = data.quantile(0.99)
                data_filtered = data[data <= q99]
                
                self.ax.hist(
                    data_filtered,
                    bins=50,
                    color="#8B0000",
                    edgecolor="black",
                    alpha=0.7,
                )
                self.ax.set_title(
                    f"Distribution of Paid Value (n={len(data):,})",
                    fontsize=12,
                )
                self.ax.set_xlabel("Paid Value (EUR)", fontsize=10)
                self.ax.set_ylabel("Frequency", fontsize=10)
                
                # Add statistics annotation (mean, median, quartiles, std)
                mean_val = float(data.mean())
                median_val = float(data.median())
                std_val = float(data.std())
                q1 = float(data.quantile(0.25))
                q3 = float(data.quantile(0.75))

                # Vertical reference lines
                self.ax.axvline(
                    mean_val,
                    color="#DAA520",
                    linestyle="--",
                    linewidth=2,
                    label=f"Mean: {mean_val:,.2f}",
                )
                self.ax.axvline(
                    median_val,
                    color="#00BFFF",
                    linestyle="-.",
                    linewidth=2,
                    label=f"Median: {median_val:,.2f}",
                )

                # Small info box in the top-left corner of the plot
                stats_text = (
                    f"Mean: {mean_val:,.2f} EUR\n"
                    f"Median: {median_val:,.2f} EUR\n"
                    f"Q1–Q3: {q1:,.2f} – {q3:,.2f} EUR\n"
                    f"Std: {std_val:,.2f} EUR\n"
                    f"Min / Max: {data.min():,.2f} / {data.max():,.2f} EUR"
                )
                self.ax.text(
                    0.02,
                    0.98,
                    stats_text,
                    transform=self.ax.transAxes,
                    fontsize=9,
                    va="top",
                    ha="left",
                    bbox=dict(boxstyle="round", facecolor="#222222", alpha=0.8),
                    color="white",
                )

                self.ax.legend(loc="upper right", fontsize=9)

        elif selection == "All Actions":
            # Show combined histogram for all actions
            data_list = []
            labels = []
            colors = ["#8B0000", "#B22222", "#DAA520", "#9370DB", "#20B2AA", "#FF8C00"]
            
            for i, action in enumerate(self.action_features):
                if action in self._training_df.columns:
                    data = self._training_df[action].dropna()
                    data_list.append(data)
                    labels.append(action)

            if data_list:
                self.ax.hist(data_list, bins=30, color=colors[:len(data_list)], 
                           label=labels, edgecolor="black", alpha=0.7)
                self.ax.set_title("Distribution of All Actions", fontsize=12)
                self.ax.set_xlabel("Action Count", fontsize=10)
                self.ax.set_ylabel("Frequency", fontsize=10)
                self.ax.legend(loc="upper right", fontsize=9)

        elif selection in self.action_features:
            if selection in self._training_df.columns:
                data = self._training_df[selection].dropna()
                color_map = {"Calls": "#8B0000", "Letters": "#B22222", "SMS": "#DAA520", "Emails": "#9370DB"}
                color = color_map.get(selection, "#8B0000")
                
                self.ax.hist(data, bins=30, color=color, edgecolor="black", alpha=0.7)
                self.ax.set_title(f"Distribution of {selection} (n={len(data):,})", fontsize=12)
                self.ax.set_xlabel(f"Number of {selection}", fontsize=10)
                self.ax.set_ylabel("Frequency", fontsize=10)
                
                # Add statistics
                mean_val = data.mean()
                median_val = data.median()
                self.ax.axvline(mean_val, color="#00BFFF", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.1f}")
                self.ax.axvline(median_val, color="#FFD700", linestyle="-.", linewidth=2, label=f"Median: {median_val:.1f}")
                self.ax.legend(loc="upper right", fontsize=9)

        self.ax.grid(axis="y", linestyle="--", alpha=0.3)
        self.fig.tight_layout(pad=1.0)
        self.canvas.draw_idle()


class TrainingTab(ctk.CTkFrame):
    """Admin-only training tab with progress bar and live loss graph."""

    def __init__(self, master, enabled: bool):
        super().__init__(master)
        self.pack(expand=True, fill="both")

        self.enabled = enabled
        self.running = False
        self.loss_history = {"epoch": [], "train": [], "val": []}
        self.phase_history: List[tuple[str, float]] = []
        self.model_metrics: Dict[str, Dict[str, List[float]]] = {}
        self.fig: Optional[Figure] = None
        self.ax = None
        self.canvas: Optional[FigureCanvasTkAgg] = None
        self.selected_train_file: Optional[str] = None

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)  # Graph area gets the expansion weight

        header = ctk.CTkLabel(
            self,
            text="Model Training",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        header.grid(row=0, column=0, pady=10)

        if not self.enabled:
            self.rowconfigure(1, weight=1)
            disabled_label = ctk.CTkLabel(
                self,
                text="Training is disabled in User Mode.\nLogin as admin to enable.",
                justify="center",
                wraplength=500,
            )
            disabled_label.grid(row=1, column=0, pady=30, padx=10, sticky="n")
            return

        top_frame = ctk.CTkFrame(self, corner_radius=10)
        top_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(10, 5))
        top_frame.columnconfigure(0, weight=0)
        top_frame.columnconfigure(1, weight=1)

        self.new_split_var = ctk.BooleanVar(value=False)
        self.new_split_checkbox = ctk.CTkCheckBox(
            top_frame,
            text="Generate new data split",
            variable=self.new_split_var,
            command=self._toggle_file_selection,
        )
        self.new_split_checkbox.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")

        self.select_btn = ctk.CTkButton(
            top_frame,
            text="Select Training File",
            command=self._select_training_file,
            state="disabled",
        )
        self.select_btn.grid(row=1, column=0, padx=10, pady=(5, 5), sticky="w")

        self.file_label = ctk.CTkLabel(
            top_frame,
            text="Using existing splits from data/train/, data/valid/, data/test/",
            anchor="w",
        )
        self.file_label.grid(row=1, column=1, padx=10, pady=(5, 5), sticky="ew")

        self.start_btn = ctk.CTkButton(
            top_frame,
            text="Start Training",
            command=self._start_training_thread,
        )
        self.start_btn.grid(row=2, column=0, padx=10, pady=10)

        self.progress_var = ctk.DoubleVar(value=0.0)
        self.progress_bar = ctk.CTkProgressBar(
            top_frame,
            variable=self.progress_var,
            width=400,
        )
        self.progress_bar.grid(row=2, column=1, padx=10, pady=10, sticky="ew")

        self.status_label = ctk.CTkLabel(
            top_frame,
            text="Idle.",
            anchor="w",
        )
        self.status_label.grid(row=3, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="w")

        # Graph area.
        graph_container = ctk.CTkFrame(self, corner_radius=10)
        graph_container.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))
        graph_container.rowconfigure(0, weight=1)
        graph_container.columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(7, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_container)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._render_loss_graph()

    def _toggle_file_selection(self) -> None:
        if self.new_split_var.get():
            self.select_btn.configure(state="normal")
            self.file_label.configure(text="No file selected.")
        else:
            self.select_btn.configure(state="disabled")
            self.file_label.configure(
                text="Using existing splits from data/train/, data/valid/, data/test/"
            )
            self.selected_train_file = None

    def _start_training_thread(self) -> None:
        if not self.enabled or self.running:
            return

        generate_new_split = self.new_split_var.get()

        if generate_new_split:
            if not self.selected_train_file or not os.path.exists(self.selected_train_file):
                messagebox.showwarning(
                    "Select training file",
                    "Please select a training file before starting.",
                )
                return
        else:
            config = load_config()
            data_cfg = config["data"]
            required_files = [
                data_cfg["train_path"],
                data_cfg["valid_path"],
                data_cfg["test_path"],
            ]
            artifacts = [
                os.path.join(data_cfg["artifacts_dir"], "encoder.pkl"),
                os.path.join(data_cfg["artifacts_dir"], "scaler.pkl"),
                os.path.join(data_cfg["artifacts_dir"], "target_scaler.pkl"),
            ]
            missing = [f for f in required_files + artifacts if not os.path.exists(f)]
            if missing:
                messagebox.showwarning(
                    "Missing files",
                    "Existing splits or artifacts not found.\n"
                    "Please check 'Generate new data split' and select a training file.\n\n"
                    f"Missing: {', '.join(os.path.basename(f) for f in missing[:3])}...",
                )
                return

        self.running = True
        self.start_btn.configure(state="disabled")
        self.status_label.configure(text="Training started...")
        threading.Thread(
            target=self._run_training,
            args=(generate_new_split,),
            daemon=True,
        ).start()

    def _select_training_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select training file",
            initialdir="data",
            filetypes=[
                ("Supported files", "*.xlsx *.xls *.csv"),
                ("Excel files", "*.xlsx *.xls"),
                ("CSV files", "*.csv"),
            ],
        )
        if not path:
            return
        self.selected_train_file = path
        self.file_label.configure(text=os.path.basename(path))

    def _prepare_training_data(self, input_path: str) -> dict:
        """
        Prepare global train/valid/test splits and preprocessing artifacts.

        This step does NOT train client-specific models; it just creates
        a cleaned, split dataset and shared encoder/scalers in the base
        artifacts directory. Client-specific models are trained later
        using filtered views of these splits.
        """
        config = load_config()
        data_cfg = config["data"]
        data_cfg["input_path"] = input_path

        self.after(0, lambda: self.status_label.configure(text="Splitting data..."))

        if input_path.lower().endswith((".xlsx", ".xls")):
            stream_split_excel(input_path, config)
            train_df = read_table_file(data_cfg["train_path"])
            valid_df = read_table_file(data_cfg["valid_path"])
            test_df = read_table_file(data_cfg["test_path"])
        else:
            raw_df = load_raw_data(input_path)
            cleaned_df = basic_cleaning(raw_df, config)
            train_df, valid_df, test_df = split_dataset(cleaned_df, config)

        train_df, valid_df, test_df = prepare_splits(train_df, valid_df, test_df, config)
        encoder, scaler, target_scaler = fit_preprocessors(train_df, config)
        save_splits_and_artifacts(
            train_df,
            valid_df,
            test_df,
            encoder,
            scaler,
            target_scaler,
            config,
        )

        return config

    def _run_training(self, generate_new_split: bool = True) -> None:
        try:
            if generate_new_split:
                config = self._prepare_training_data(self.selected_train_file)
            else:
                config = load_config()
                self.after(0, lambda: self.status_label.configure(text="Loading existing splits..."))

            self.loss_history = {"epoch": [], "train": [], "val": []}
            self.phase_history = []
            self.model_metrics = {}
            self.after(0, self._render_loss_graph)

            def progress(phase: str, frac: float) -> None:
                def upd() -> None:
                    if not self.running:
                        return
                    self.phase_history.append((phase, float(frac)))
                    self.progress_var.set(frac)
                    self.status_label.configure(text=f"GBM training: {phase}…")
                    self._render_loss_graph()

                self.after(0, upd)

            def metrics(phase: str, train_vals: List[float], val_vals: List[float]) -> None:
                def upd() -> None:
                    if not self.running:
                        return
                    self.model_metrics[phase] = {
                        "train": list(train_vals),
                        "val": list(val_vals),
                    }
                    self._render_loss_graph()

                self.after(0, upd)

            if not self.running:
                return
            train_all_gbm(config, progress_callback=progress, metrics_callback=metrics)

            self.running = False
            self.after(
                0,
                lambda: self.status_label.configure(
                    text="Training finished." if self.enabled else "Training stopped."
                ),
            )
            self.after(0, lambda: self.start_btn.configure(state="normal"))
            self.after(0, self._render_loss_graph)

        except Exception as exc:
            self.running = False
            print(f"Training error: {exc}")
            self.after(
                0, lambda: messagebox.showerror("Training error", str(exc))
            )
            self.after(0, lambda: self.start_btn.configure(state="normal"))

    def _update_training_ui(
        self,
        epoch: int,
        n_epochs: int,
        train_loss: float,
        val_loss: float,
    ) -> None:
        progress = epoch / max(n_epochs, 1)

        def _update():
            self.progress_var.set(progress)
            self.status_label.configure(
                text=f"Epoch {epoch}/{n_epochs} - Train MSE: {train_loss:.4f}, Val MSE: {val_loss:.4f}"
            )
            self._render_loss_graph()

        self.after(0, _update)

    def _render_loss_graph(self) -> None:
        if self.fig is None or self.ax is None or self.canvas is None:
            return

        self.ax.clear()
        if self.model_metrics:
            plotted = False
            if "catboost_paid" in self.model_metrics:
                cat = self.model_metrics["catboost_paid"]
                train_vals = cat.get("train", [])
                val_vals = cat.get("val", [])
                if train_vals:
                    self.ax.plot(
                        range(1, len(train_vals) + 1),
                        train_vals,
                        color="#B22222",
                        linewidth=1.3,
                        label="CatBoost train",
                    )
                    plotted = True
                if val_vals:
                    self.ax.plot(
                        range(1, len(val_vals) + 1),
                        val_vals,
                        color="#DAA520",
                        linewidth=1.3,
                        label="CatBoost val",
                    )
                    plotted = True
            if "lightgbm_paid" in self.model_metrics:
                lgbm = self.model_metrics["lightgbm_paid"]
                val_vals = lgbm.get("val", [])
                if val_vals:
                    self.ax.plot(
                        range(1, len(val_vals) + 1),
                        val_vals,
                        color="#1f6aa5",
                        linewidth=1.2,
                        linestyle="--",
                        label="LightGBM val",
                    )
                    plotted = True
            if plotted:
                self.ax.set_title("Training Metrics by Iteration")
                self.ax.set_xlabel("Iteration")
                self.ax.set_ylabel("Metric")
                self.ax.grid(True, linestyle="--", alpha=0.3)
                self.ax.legend()
                self.canvas.draw_idle()
                return

        if self.phase_history:
            phases = [p for p, _ in self.phase_history]
            values = [v for _, v in self.phase_history]
            x = np.arange(len(phases))
            self.ax.plot(x, values, marker="o", color="#1f6aa5", linewidth=2)
            self.ax.set_ylim(0.0, 1.05)
            self.ax.set_xticks(x)
            self.ax.set_xticklabels(phases, rotation=20, ha="right", fontsize=9)
            self.ax.set_yticks(np.linspace(0, 1, 6))
            self.ax.set_yticklabels([f"{int(t * 100)}%" for t in np.linspace(0, 1, 6)])
            self.ax.set_title("GBM Training Phase Progress")
            self.ax.set_xlabel("Phase")
            self.ax.set_ylabel("Completion")
            self.ax.grid(True, linestyle="--", alpha=0.3)
            self.fig.tight_layout(pad=1.0)
            self.canvas.draw_idle()
            return

        if not self.loss_history["epoch"]:
            self.ax.text(
                0.5,
                0.5,
                "GBM training has no per-epoch curve;\nwatch the status bar for phases.",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
                fontsize=10,
                color="#888888",
            )
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.set_title("Training progress")
            self.canvas.draw_idle()
            return

        self.ax.plot(
            self.loss_history["epoch"],
            self.loss_history["train"],
            marker="o",
            color="#B22222",
            label="Train Loss",
        )
        self.ax.plot(
            self.loss_history["epoch"],
            self.loss_history["val"],
            marker="o",
            color="#DAA520",
            label="Val Loss",
        )
        self.ax.set_title("Training & Validation Loss (MSE)")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.grid(True, linestyle="--", alpha=0.3)
        self.ax.legend()
        self.canvas.draw_idle()


class TestingTab(ctk.CTkFrame):
    """Admin-only testing tab that evaluates the model on a provided test.xlsx."""

    def __init__(self, master, enabled: bool):
        super().__init__(master)
        self.pack(expand=True, fill="both")
        self.enabled = enabled
        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)  # Results area gets the expansion weight

        header = ctk.CTkLabel(
            self,
            text="Model Testing",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        header.grid(row=0, column=0, pady=10)

        if not self.enabled:
            self.rowconfigure(1, weight=1)
            disabled_label = ctk.CTkLabel(
                self,
                text="Testing is disabled in User Mode.\nLogin as admin to enable.",
                justify="center",
                wraplength=500,
            )
            disabled_label.grid(row=1, column=0, pady=30, padx=10, sticky="n")
            return

        btn = ctk.CTkButton(
            self,
            text="Select test.xlsx and Evaluate",
            command=self._run_testing,
        )
        btn.grid(row=1, column=0, pady=(10, 5))

        # Portfolio test helpers: generate test portfolios and run portfolio tests
        portfolio_btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        portfolio_btn_frame.grid(row=2, column=0, pady=(0, 5))
        ctk.CTkButton(
            portfolio_btn_frame,
            text="Generate test portfolios",
            command=self._generate_test_portfolios,
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            portfolio_btn_frame,
            text="Run portfolio tests",
            command=self._run_portfolio_tests,
        ).pack(side="left")

        self.results_text = ctk.CTkTextbox(self, height=200)
        self.results_text.grid(row=3, column=0, pady=10, padx=10, sticky="nsew")

        # Progress bar for portfolio tests
        self.test_progress_bar = ctk.CTkProgressBar(self, mode="determinate")
        self.test_progress_bar.grid(row=4, column=0, pady=(0, 5), padx=10, sticky="ew")
        self.test_progress_bar.set(0)
        self.test_progress_label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=10),
            text_color="#888888",
        )
        self.test_progress_label.grid(row=5, column=0, pady=(0, 5), padx=10)

    def _update_test_progress(self, value: float, text: str = "") -> None:
        """Update test progress bar and label."""
        self.test_progress_bar.set(value)
        if text:
            self.test_progress_label.configure(text=text)
        self.update_idletasks()

    def _generate_test_portfolios(self) -> None:
        """Generate 10 test portfolios (worst to best) in data/test_portfolios/."""
        try:
            scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
            if scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)
            import generate_test_portfolios as gen_module  # type: ignore[import-not-found]
            config = load_config()
            out_dir = os.path.join("data", "test_portfolios")
            os.makedirs(out_dir, exist_ok=True)
            labels = [
                "worst", "very_bad", "bad", "below_avg", "below_avg2",
                "avg", "above_avg", "good", "very_good", "best",
            ]
            n_cases = 50
            lines = [f"Generating {len(labels)} test portfolios in {out_dir}..."]
            for idx, label in enumerate(labels):
                df = gen_module.generate_portfolio(n_cases, label, seed=42 + idx, config=config)
                path = os.path.join(out_dir, f"portfolio_{idx + 1:02d}_{label}.xlsx")
                df.to_excel(path, index=False)
                lines.append(f"  Wrote {path} ({len(df)} cases)")
            lines.append(f"Done. Run 'Run portfolio tests' to evaluate.")
            self.results_text.delete("1.0", "end")
            self.results_text.insert("end", "\n".join(lines))
        except Exception as exc:
            messagebox.showerror("Generate test portfolios", str(exc))

    def _run_portfolio_tests(self) -> None:
        """Run inference on test portfolios and report errors (predicted vs actual)."""
        portfolios_dir = os.path.join("data", "test_portfolios")
        if not os.path.isdir(portfolios_dir):
            messagebox.showwarning(
                "Run portfolio tests",
                f"Directory not found: {portfolios_dir}\nRun 'Generate test portfolios' first.",
            )
            return
        xlsx_files = sorted(
            [f for f in os.listdir(portfolios_dir) if f.endswith(".xlsx") and not f.startswith("report")]
        )
        if not xlsx_files:
            messagebox.showwarning("Run portfolio tests", f"No .xlsx files in {portfolios_dir}")
            return
        self._update_test_progress(0.05, "Loading model artifacts...")
        try:
            load_artifacts_for_inference()
        except Exception as e:
            self._update_test_progress(0.0, "")
            messagebox.showerror(
                "Run portfolio tests",
                f"Failed to load model artifacts: {e}\nTrain the model first.",
            )
            return
        rows = []
        total_files = len(xlsx_files)
        config = load_config()
        action_costs = config.get("action_costs", {})

        for file_idx, f in enumerate(xlsx_files):
            progress_base = 0.1 + (file_idx / total_files) * 0.85
            self._update_test_progress(progress_base, f"Processing portfolio {file_idx + 1}/{total_files}: {f}")
            path = os.path.join(portfolios_dir, f)
            name = os.path.splitext(f)[0]
            try:
                df = pd.read_excel(path)
            except Exception as e:
                rows.append({"portfolio": name, "status": f"read error: {e}"})
                continue
            if "Paid Value" not in df.columns:
                rows.append({"portfolio": name, "status": "no 'Paid Value' column"})
                continue
            total_actual = float(pd.to_numeric(df["Paid Value"], errors="coerce").fillna(0).sum())

            def portfolio_progress_callback(current: int, total: int, idx=file_idx) -> None:
                # Progress within a single portfolio (0-85% of file progress)
                file_progress = 0.1 + (idx / total_files) * 0.85
                portfolio_progress = (current / total) * 0.85
                overall = file_progress + portfolio_progress
                self._update_test_progress(overall, f"Portfolio {idx + 1}/{total_files}: {current}/{total} cases")

            try:
                result = recommend_portfolio_strategy(df, config=config, action_costs=action_costs, progress_callback=portfolio_progress_callback)
            except Exception as e:
                rows.append({"portfolio": name, "total_actual": total_actual, "status": str(e)})
                continue
            if "Optimal_Predicted_Value" not in result.columns:
                rows.append({"portfolio": name, "status": "no Optimal_Predicted_Value in result"})
                continue
            total_pred = float(result["Optimal_Predicted_Value"].sum())
            error = total_pred - total_actual
            rel_pct = (error / total_actual * 100) if total_actual != 0 else None
            rows.append({
                "portfolio": name,
                "n_cases": len(df),
                "total_actual": total_actual,
                "total_predicted": total_pred,
                "error": error,
                "rel_error_pct": rel_pct,
                "status": "ok",
            })
        report_df = pd.DataFrame(rows)
        ok = report_df[report_df["status"] == "ok"] if "status" in report_df.columns else pd.DataFrame()
        lines = [f"Portfolio tests ({len(xlsx_files)} files)", ""]
        for _, r in report_df.iterrows():
            if r.get("status") == "ok":
                pct = f" ({r['rel_error_pct']:.1f}%)" if pd.notna(r.get("rel_error_pct")) else ""
                lines.append(f"  {r['portfolio']}: actual={r['total_actual']:,.0f}, pred={r['total_predicted']:,.0f}, error={r['error']:,.0f}{pct}")
            else:
                lines.append(f"  {r['portfolio']}: {r.get('status', '?')}")
        if len(ok) > 0:
            mae = ok["error"].abs().mean()
            rmse = (ok["error"] ** 2).mean() ** 0.5
            lines.append("")
            lines.append(f"MAE: {mae:,.2f}  |  RMSE: {rmse:,.2f}")
            if ok["rel_error_pct"].notna().any():
                lines.append(f"Mean |rel error| %: {ok['rel_error_pct'].abs().mean():.1f}%")
        self._update_test_progress(0.95, "Generating report...")
        report_path = os.path.join(portfolios_dir, "report.csv")
        try:
            report_df.to_csv(report_path, index=False)
            lines.append(f"\nReport saved to {report_path}")
        except Exception:
            pass
        self._update_test_progress(1.0, "Complete!")
        self.after(500, lambda: self._update_test_progress(0.0, ""))
        self.results_text.delete("1.0", "end")
        self.results_text.insert("end", "\n".join(lines))

    def _run_testing(self) -> None:
        path = filedialog.askopenfilename(
            title="Select test file",
            initialdir="data/test",
            filetypes=[
                ("Supported files", "*.xlsx *.xls *.csv"),
                ("Excel files", "*.xlsx *.xls"),
                ("CSV files", "*.csv"),
            ],
        )
        if not path:
            return

        try:
            df = read_table_file(path)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load test file:\n{exc}")
            return

        try:
            from sklearn.metrics import mean_absolute_error, mean_squared_error

            config = load_config()
            data_cfg = config["data"]
            artifacts_dir = data_cfg["artifacts_dir"]
            cols_cfg = config["columns"]
            target_col = cols_cfg["target_column"]
            gbm_cfg = config.get("gbm", {})
            paid_blend = gbm_cfg.get("paid_blend", {})
            tolerance_pct = float(paid_blend.get("relative_error_tolerance_pct", 20))

            encoder = joblib.load(os.path.join(artifacts_dir, "encoder.pkl"))
            scaler = joblib.load(os.path.join(artifacts_dir, "scaler.pkl"))
            target_scaler = joblib.load(os.path.join(artifacts_dir, "target_scaler.pkl"))

            bundle = load_paid_bundle(config)
            _, _, _, _, _, routing = bundle

            if target_col not in df.columns:
                messagebox.showerror(
                    "Testing error",
                    f"Test file missing target column '{target_col}'.",
                )
                return

            y_true = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0).values
            y_pred = predict_paid_values_for_dataframe(
                df, config, encoder, scaler, target_scaler, bundle=bundle
            )

            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = mse ** 0.5

            lines = [
                f"Test file: {os.path.basename(path)}",
                f"Samples: {len(y_true)}",
                "",
                f"MAE:  {mae:.4f}",
                f"RMSE: {rmse:.4f}",
                "",
            ]

            cv_col = str(routing.get("case_value_column", "Case Value"))
            qmap = routing.get("case_value_quantiles_train", {})
            q50 = float(qmap.get("p50", float("nan")))
            q75 = float(qmap.get("p75", float("nan")))
            q90 = float(qmap.get("p90", float("nan")))

            if cv_col in df.columns:
                case_values = pd.to_numeric(df[cv_col], errors="coerce").fillna(0.0).values
            else:
                case_values = np.zeros(len(df), dtype=float)

            thr = float(routing.get("case_value_threshold", 0.0))
            high_mask = case_values >= thr
            lines.append(
                f"Routing: {100.0 * high_mask.mean():.1f}% rows used extended LightGBM "
                f"(CV >= {thr:.2f})."
            )
            lines.append("")

            if np.isfinite(q50) and np.isfinite(q75) and np.isfinite(q90):
                tol = tolerance_pct / 100.0
                eps = 1e-9
                rel_err = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), eps)
                within = rel_err <= tol
                buckets: Dict[str, List[bool]] = {
                    "≤p50": [],
                    "p50–p75": [],
                    "p75–p90": [],
                    ">p90": [],
                }
                for cv, ok in zip(case_values, within):
                    cvf = float(cv)
                    if cvf <= q50:
                        b = "≤p50"
                    elif cvf <= q75:
                        b = "p50–p75"
                    elif cvf <= q90:
                        b = "p75–p90"
                    else:
                        b = ">p90"
                    buckets[b].append(bool(ok))
                lines.append(f"Within ±{tolerance_pct}% of actual (train CV buckets):")
                for name, flags in buckets.items():
                    if not flags:
                        lines.append(f"  {name}: (no rows)")
                    else:
                        lines.append(
                            f"  {name}: {100.0 * np.mean(flags):.1f}%  (n={len(flags)})"
                        )
            else:
                lines.append(
                    "Business buckets: train quantiles missing in blend_routing.json "
                    "(re-train GBM to refresh routing)."
                )

            self.results_text.delete("1.0", "end")
            self.results_text.insert("end", "\n".join(lines))
        except Exception as exc:
            messagebox.showerror("Testing error", str(exc))


def main() -> None:
    """
    Entry point for launching the GUI.
    """
    app = MainApp()
    app.mainloop()


if __name__ == "__main__":
    main()
