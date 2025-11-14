import customtkinter as ctk
from tkinter import messagebox, filedialog
import numpy as np
import csv
import math

# -------------------- Appearance --------------------
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# color palette (tweak to taste)
BG = "#0F1113"             # app background (very dark)
PANEL = "#1F2426"          # main panels (dark gray)
CARD = "#232628"           # content card color
ACCENT = "#2C5F8A"         # teal/blue accent (buttons, info circles)
PRIMARY_BTN = ACCENT
DANGER = "#D9534F"         # red for Exit button
TEXT_PRIMARY = "#ECEFF1"   # primary text color (off-white)
TEXT_MUTED = "#BFC9CC"     # muted text

# -------------------- App Window --------------------
app = ctk.CTk()
app.title("AHP — Groundwater Recharge Potential")
app.geometry("1300x850")

# -------------------- Fonts --------------------
HEADER_FONT = ("Arial", 24, "bold")
SUBHEADER_FONT = ("Arial", 18, "bold")
# Slightly smaller body font to avoid clipping on narrow containers
BODY_FONT = ("Arial", 14)
ITALIC_FONT = ("Arial", 13, "italic")
SMALL_FONT = ("Arial", 12)
HOVER_FONT = ("Arial", 13, "italic")

# -------------------- Parameters & Descriptions --------------------
PARAMETERS = [
    "Geology",
    "Land Cover",
    "Slope",
    "Lineament Density",
    "Rainfall",
    "Drainage Density",
    "Soil"
]

PARAMETER_DESCRIPTIONS = {
    "Geology": (
        "Geology plays a fundamental role in determining groundwater recharge because the type and structure of underlying rocks control the movement and storage of water in the subsurface. "
        "Rocks with high porosity and permeability, such as sandstones and other sedimentary formations, allow more water to infiltrate and percolate into aquifers. "
        "In contrast, metamorphic and plutonic rocks typically act as barriers due to their compact structure and low permeability, restricting groundwater flow. "
        "Volcanic rocks may contribute variably, depending on the presence of fractures and vesicular materials that can serve as secondary pathways for infiltration."
    ),
    "Land Cover": (
        "Land cover directly affects the amount of water that infiltrates the ground versus the amount that becomes surface runoff. "
        "Vegetated areas like forests, shrubs, and croplands enhance infiltration by reducing overland flow, improving soil structure, and increasing the time water remains on the surface. "
        "The root systems of vegetation help loosen the soil, allowing water to percolate more easily into recharge zones. "
        "Conversely, impervious surfaces such as urban or built-up areas hinder infiltration and increase surface runoff, significantly reducing groundwater recharge potential."
    ),
    "Slope": (
        "Slope determines the rate at which rainfall runs off the surface versus the amount that infiltrates into the ground. "
        "Gentle slopes favor infiltration by slowing down surface flow, allowing more time for water to seep into the subsurface layers. "
        "Steeper slopes, however, accelerate runoff, reduce the residence time of water, and limit infiltration. "
        "Therefore, flatter terrains generally correspond to higher groundwater recharge potential, while steep terrains are often associated with low recharge capacity."
    ),
    "Lineament Density": (
        "Lineament density reflects the frequency of linear features such as fractures, joints, and faults within an area. "
        "These structural features often act as natural conduits that facilitate the downward and lateral movement of water. "
        "Areas with high lineament density typically exhibit greater recharge potential because water can easily infiltrate through these openings into deeper subsurface zones. "
        "Conversely, areas with low lineament density have fewer pathways for infiltration, limiting groundwater recharge."
    ),
    "Rainfall": (
        "Rainfall serves as the primary input to groundwater systems, supplying the water necessary for infiltration and recharge. "
        "The amount, intensity, and temporal distribution of rainfall determine how much water percolates through the soil to reach aquifers. "
        "Moderate to high rainfall supports greater recharge potential, provided the soil and land surface conditions favor infiltration. "
        "However, excessive rainfall intensity may instead promote surface runoff, especially in areas with steep slopes or impervious soils, reducing effective recharge."
    ),
    "Drainage Density": (
        "Drainage density describes the total length of stream channels per unit area and serves as an indicator of surface runoff characteristics. "
        "Areas with high drainage density tend to have well-developed stream networks that quickly convey rainfall to rivers, minimizing the opportunity for infiltration. "
        "In contrast, areas with low drainage density retain more water on the surface, allowing it to infiltrate and recharge the subsurface. "
        "Thus, drainage density is inversely related to groundwater recharge potential."
    ),
    "Soil": (
        "Soil influences groundwater recharge primarily through its texture, structure, and permeability, which govern the rate of water infiltration. "
        "Sandy and loamy soils (classified as Hydrologic Soil Group A and B) have large pore spaces that allow water to pass through easily, supporting high recharge potential. "
        "Clayey soils (Group C and D), on the other hand, have fine particles that retain water and impede infiltration, resulting in lower recharge rates. "
        "Soil properties also affect runoff behavior, with more permeable soils encouraging percolation and less permeable soils promoting surface flow."
    )
}

SAATY_PHRASES = {
    1: "are equally important.",
    2: "is slightly more important than",
    3: "is moderately more important than",
    4: "is moderately-to-strongly more important than",
    5: "is strongly more important than",
    6: "is strongly-to-very strongly more important than",
    7: "is very strongly more important than",
    8: "is very strongly-to-extremely more important than",
    9: "is extremely more important than"
}

# -------------------- State --------------------
state = {
    'status': 'pending',
    'rank_vars': None,
    'ranked': [],
    'comparisons': [],
    'pairwise': {},
    'index': 0,
    'weights': None,
    'CR': None,
    'matrix': None,
    'correction_mode': False,
    'correction_pair': None,
    'correction_details': None
}

# -------------------- Math Helpers --------------------
def build_ranked_order(rank_vars):
    items = []
    for p, var in rank_vars.items():
        try:
            r = int(var.get())
        except Exception:
            r = 99
        items.append((p, r))
    items.sort(key=lambda x: x[1])
    return [p for p, _ in items]

def build_comparisons(ranked):
    comps = []
    n = len(ranked)
    for i in range(n):
        for j in range(i + 1, n):
            comps.append((ranked[i], ranked[j]))
    return comps

def build_matrix(params, pairwise):
    n = len(params)
    A = np.ones((n, n), dtype=float)
    idx = {p: i for i, p in enumerate(params)}
    for (p, q), v in pairwise.items():
        if p in idx and q in idx:
            try:
                val = float(v)
            except Exception:
                val = 1.0
            A[idx[p], idx[q]] = val
            A[idx[q], idx[p]] = 1.0 / val if val != 0 else 1.0
    return A

def compute_eigen_weights_cr(A):
    vals, vecs = np.linalg.eig(A)
    max_idx = np.argmax(vals.real)
    eigval = vals.real[max_idx]
    eigvec = np.abs(vecs[:, max_idx].real)
    w = eigvec / np.sum(eigvec)
    n = A.shape[0]
    CI = (eigval - n) / (n - 1) if n > 1 else 0.0
    RI_table = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32}
    RI = RI_table.get(n, 1.12)
    CR = CI / RI if RI != 0 else 0.0
    return w, CR, eigval

# --------------------
# Suggestion helper (creates the suggested integer and range + reason)
# --------------------
def suggest_allowed_range(a, b, ranked_list, implied, given, error, n_params):
    eps = 1e-12
    implied_clamped = max(1.0, min(9.0, implied if np.isfinite(implied) else 9.0))
    suggested_int = int(round(implied_clamped))
    suggested_int = max(1, min(9, suggested_int))

    E_scale = 1.5
    E = min((error if error is not None else 0.0) / E_scale, 1.0)
    try:
        pos_a, pos_b = ranked_list.index(a), ranked_list.index(b)
        rank_distance = abs(pos_a - pos_b)
    except ValueError:
        rank_distance = 0
    R_scale = max(1, min(3, n_params - 1))
    R = min(rank_distance / R_scale, 1.0)

    midness = 1.0 - (abs(implied_clamped - 5.0) / 4.0)

    influence = 0.25 * E + 0.20 * R + 0.55 * midness
    if influence <= 0.35:
        k = 1
    elif influence <= 0.70:
        k = 2
    else:
        k = 3

    low = max(1, suggested_int - k)
    high = min(9, suggested_int + k)

    reason = (f"Implied ≈ {implied_clamped:.2f}, error={error:.3f}, influence={influence:.2f} → "
              f"suggested integer {suggested_int}, range [{low}, {high}]")

    return {
        'suggested_int': suggested_int,
        'low': low,
        'high': high,
        'reason': reason
    }

def top_inconsistencies(A, params, w, k=4):
    entries = []
    n = len(params)
    for i in range(n):
        for j in range(i + 1, n):
            aij = A[i, j]
            implied = w[i] / w[j] if w[j] != 0 else float('inf')
            diff = abs(np.log(aij + 1e-12) - np.log(implied + 1e-12))
            entries.append(((params[i], params[j]), diff, aij, implied))
    entries.sort(key=lambda x: x[1], reverse=True)
    return entries[:k]

def suggest_range_for_pair(a, b, ranked_list):
    try:
        pos_a, pos_b = ranked_list.index(a), ranked_list.index(b)
    except ValueError:
        return "No ranking guidance available."
    diff = abs(pos_a - pos_b)
    if diff == 0:
        return "Advisory: items ranked similarly — consider 1 to 3."
    elif diff == 1:
        return "Advisory: adjacent ranks — consider 2 to 5."
    else:
        return "Advisory: different ranks — consider 3 to 9."

def scale_phrase(val, a, b):
    if val == 1:
        return f"{a} and {b} {SAATY_PHRASES[val]}"
    else:
        return f"{a} {SAATY_PHRASES[val]} {b}."

# -------------------- UI Helpers --------------------
def clear_screen():
    for w in main_content.winfo_children():
        w.destroy()

# -------------------- Layout --------------------
header = ctk.CTkFrame(app, corner_radius=0)
header.pack(fill="x")
header.grid_columnconfigure(0, weight=1)
title_lbl = ctk.CTkLabel(header, text="Groundwater Recharge AHP", font=HEADER_FONT)
title_lbl.grid(row=0, column=0, sticky="w", padx=20, pady=18)
sub_lbl = ctk.CTkLabel(header, text="AHP weights · consistency checks", font=SMALL_FONT)
sub_lbl.grid(row=0, column=1, sticky="e", padx=20)

container = ctk.CTkFrame(app, fg_color="transparent")
container.pack(fill="both", expand=True, padx=16, pady=12)
container.grid_rowconfigure(0, weight=1)
container.grid_columnconfigure(1, weight=1)

left_panel = ctk.CTkFrame(container, width=260, corner_radius=8)
left_panel.grid(row=0, column=0, sticky="nsw", padx=(0,12), pady=6)
main_content = ctk.CTkFrame(container, corner_radius=8)
main_content.grid(row=0, column=1, sticky="nsew", pady=6)

ctk.CTkLabel(left_panel, text="Controls", font=SUBHEADER_FONT).pack(pady=(12,6))
ctk.CTkButton(left_panel, text="Start AHP", command=lambda: start_ranking(), width=220).pack(pady=6)
ctk.CTkButton(left_panel, text="Save Results", command=lambda: save_results(), width=220).pack(pady=6)
ctk.CTkButton(left_panel, text="Exit", command=app.destroy, width=220, fg_color="#D9534F").pack(side="bottom", pady=20)

# -------------------- Dashboard --------------------
def dashboard():
    # restore left panel default controls
    for w in left_panel.winfo_children():
        w.destroy()
    ctk.CTkLabel(left_panel, text="Controls", font=SUBHEADER_FONT).pack(pady=(12,6))
    ctk.CTkButton(left_panel, text="Start AHP", command=lambda: start_ranking(), width=220).pack(pady=6)
    ctk.CTkButton(left_panel, text="Save Results", command=lambda: save_results(), width=220).pack(pady=6)
    ctk.CTkButton(left_panel, text="Exit", command=app.destroy, width=220, fg_color=DANGER).pack(side="bottom", pady=20)

    clear_screen()

    # main dark card (matches the 'Geology' look)
    dash = ctk.CTkFrame(main_content, fg_color=CARD, corner_radius=8)
    dash.pack(fill="both", expand=True, padx=12, pady=12)

    # Title + subtitle
    ctk.CTkLabel(
        dash,
        text="Identify Potential Groundwater Recharge Zones",
        font=("Arial", 28, "bold"),
        text_color=TEXT_PRIMARY,
        anchor="w"
    ).pack(anchor="w", pady=(16, 6), padx=20)

    ctk.CTkLabel(
        dash,
        text="Perform AHP to derive parameter weights and check consistency ratio (CR ≤ 0.1).",
        font=("Arial", 18),
        text_color=TEXT_MUTED,
        anchor="w"
    ).pack(anchor="w", padx=20)

    # ---------------- Definitions section (tighter wrap + padding) ----------------
    defs_frame = ctk.CTkFrame(dash, corner_radius=8, fg_color=CARD)
    defs_frame.pack(fill="x", padx=20, pady=(12,10))
    defs_frame.grid_columnconfigure(0, weight=1)

    # Wrap length chosen conservatively to avoid clipping at the right border.
    DEF_WRAP = 880
    DEF_PAD_X = 16
    DEF_PAD_Y = (10, 12)

    def create_def_card(parent, title, text):
        card = ctk.CTkFrame(parent, corner_radius=8, fg_color=CARD)
        card.pack(fill="x", padx=8, pady=8)
        # title
        ctk.CTkLabel(card, text=title, font=("Arial", 14, "bold"), text_color=TEXT_PRIMARY).pack(anchor="w", padx=12, pady=(8,4))
        # description label with conservative wraplength and extra internal padding
        lbl = ctk.CTkLabel(card, text=text, wraplength=DEF_WRAP, justify="left", anchor="w", font=BODY_FONT, text_color=TEXT_MUTED)
        # ensure the label takes available space and has internal padding to keep text away from borders
        lbl.pack(fill="both", expand=True, padx=DEF_PAD_X, pady=DEF_PAD_Y)
        return card

    gw_text = (
        "Groundwater recharge occurs when rainfall infiltrates the ground and moves through the unsaturated zone to replenish aquifers. "
        "Recharge can also come from local percolation of streamflow or water seeping through joints, cracks, and fissures. "
        "The effectiveness of recharge depends on geologic attributes and soil characteristics that control infiltration and subsurface flow."
    )
    create_def_card(defs_frame, "Groundwater Recharge — definition", gw_text)

    ahp_text = (
        "AHP is a structured multi-criteria decision-making approach used to determine the relative importance of parameters through pairwise comparisons. "
        "In this application, AHP is used to assign weights to groundwater recharge parameters via expert judgment, with consistency assessed using the Consistency Ratio (CR)."
    )
    create_def_card(defs_frame, "AHP (Analytic Hierarchy Process) — definition", ahp_text)

    cr_text = (
        "The consistency ratio in AHP checks whether your pairwise comparisons are logically aligned. "
        "It measures how well your judgments fit together by comparing your actual consistency to what would be expected from random choices. "
        "A consistency ratio ≤ 0.1 indicates that comparisons are acceptably consistent and the resulting weights are reliable."
    )
    create_def_card(defs_frame, "Consistency Ratio — definition", cr_text)
    # ------------------------------------------------------------

    # If weights present show latest CR and button
    if state['weights'] is not None:
        cr_color = "#1E8449" if state['CR'] <= 0.1 else "#C0392B"
        ctk.CTkLabel(dash, text=f"Latest CR: {state['CR']:.4f}", text_color=cr_color, font=("Arial", 16)).pack(anchor="w", pady=10, padx=20)
        ctk.CTkButton(dash, text="View Results", command=show_results, width=140).pack(pady=8, padx=20, anchor="w")

    tips = (
        "1. Rank parameters 1 (most important) to 7 (least important).\n"
        "2. Proceed through pairwise comparisons following Saaty's scale.\n"
        "3. If CR > 0.1, review suggested inconsistent pairs."
    )
    ctk.CTkLabel(dash, text=tips, justify="left", font=("Arial", 16), text_color=TEXT_MUTED).pack(anchor="w", pady=12, padx=20)


# -------------------- Ranking Screen --------------------
def start_ranking():
    clear_screen()
    top_frame = ctk.CTkFrame(main_content)
    top_frame.pack(fill="x", padx=12, pady=(8,6))
    ctk.CTkLabel(top_frame, text="Rank the parameters from 1 (most important) to 7 (least important):", font=BODY_FONT).pack(anchor="w")
    ctk.CTkLabel(top_frame, text="Hover or click the info (i) button to read each parameter’s description.", font=ITALIC_FONT).pack(anchor="w", pady=(6,0))

    rank_vars = {}

    # renamed to avoid confusion with tkinter.grid method
    rank_grid = ctk.CTkFrame(main_content)
    rank_grid.pack(fill="x", padx=12, pady=(6,12))

    # Static bottom description box (unchanged)
    desc_frame = ctk.CTkFrame(main_content, corner_radius=8, width=880, height=220)
    desc_frame.pack(fill="x", padx=12, pady=(12,16))
    desc_frame.pack_propagate(False)

    inner = ctk.CTkFrame(desc_frame, fg_color="transparent")
    inner.pack(fill="both", expand=True, padx=16, pady=10)

    desc_title = ctk.CTkLabel(inner, text=PARAMETERS[0], font=("Arial", 15, "bold"), anchor="w")
    desc_title.pack(anchor="nw", padx=2, pady=(0,6))

    desc_label = ctk.CTkLabel(inner, text=PARAMETER_DESCRIPTIONS[PARAMETERS[0]], wraplength=840, justify="left", anchor="nw", font=BODY_FONT)
    desc_label.pack(fill="both", expand=True, padx=2, pady=(0,4))

    def update_description(param):
        desc_title.configure(text=param)
        desc_label.configure(text=PARAMETER_DESCRIPTIONS[param])

    # create rows for each parameter
    for i, p in enumerate(PARAMETERS):
        row = ctk.CTkFrame(rank_grid)
        row.pack(fill="x", pady=6)
        var = ctk.StringVar(value="1")
        # OptionMenu values should be strings; explicitly set width to prevent squeezing
        ctk.CTkOptionMenu(row, values=[str(i) for i in range(1,8)], variable=var, width=100, font=BODY_FONT).pack(side="left", padx=(4,8))
        info_btn = ctk.CTkButton(row, text="i", width=36, height=36, corner_radius=18, fg_color="#2C5F8A", font=("Arial", 12, "bold"), command=lambda param=p: update_description(param))
        info_btn.pack(side="left", padx=(4,8))
        info_btn.bind("<Enter>", lambda e, param=p: update_description(param))
        ctk.CTkLabel(row, text=p, anchor="w", font=BODY_FONT).pack(side="left", padx=(8,0))
        rank_vars[p] = var

    state['rank_vars'] = rank_vars

    control_frame = ctk.CTkFrame(main_content)
    control_frame.pack(fill="x", padx=12, pady=8)
    ctk.CTkButton(control_frame, text="Confirm Ranking", command=confirm_ranking, width=180).pack(side="left", padx=(0,12))
    ctk.CTkButton(control_frame, text="Back", command=dashboard, width=120).pack(side="left")

# -------------------- Pairwise Comparison --------------------
def confirm_ranking():
    ranked = build_ranked_order(state['rank_vars'])
    state['ranked'] = ranked
    state['comparisons'] = build_comparisons(ranked)
    state['index'] = 0
    state['pairwise'] = {}
    open_pairwise()

def open_pairwise():
    clear_screen()
    # rebuild left_panel to include Ranking Order (only during pairwise)
    for w in left_panel.winfo_children():
        w.destroy()
    ctk.CTkLabel(left_panel, text="Controls", font=SUBHEADER_FONT).pack(pady=(12,6))
    ctk.CTkButton(left_panel, text="Start AHP", command=lambda: start_ranking(), width=220).pack(pady=6)
    ctk.CTkButton(left_panel, text="Save Results", command=lambda: save_results(), width=220).pack(pady=6)

    # Insert ranking list under Save Results (simple static list)
    rank_frame = ctk.CTkFrame(left_panel)
    rank_frame.pack(fill="x", padx=8, pady=(8,12))
    ctk.CTkLabel(rank_frame, text="Ranking Order", font=("Arial", 13, "bold")).pack(anchor="w", padx=8, pady=(6,4))
    for i, p in enumerate(state.get('ranked', []), start=1):
        ctk.CTkLabel(rank_frame, text=f"{i}. {p}", anchor="w", font=("Arial", 12)).pack(anchor="w", padx=8)

    ctk.CTkButton(left_panel, text="Exit", command=app.destroy, width=220, fg_color="#D9534F").pack(side="bottom", pady=20)

    header_bar = ctk.CTkFrame(main_content)
    header_bar.pack(fill="x", padx=12, pady=(6, 8))
    header_bar.grid_columnconfigure(0, weight=1)

    # Left: Compare title
    if state.get('correction_mode') and state.get('correction_pair') is not None:
        a, b = state['correction_pair']
        title_text = f"Re-evaluate: {a} vs {b}"
    else:
        comps = state.get('comparisons', [])
        idx = state.get('index', 0)
        if idx >= len(comps):
            finalize()
            return
        a, b = comps[idx]
        title_text = f"Compare: {a} vs {b}"

    title_lbl = ctk.CTkLabel(header_bar, text=title_text, font=SUBHEADER_FONT)
    title_lbl.pack(side="left", anchor="w")

    # Right: Comparison progress indicator (upper-right)
    total = len(state.get('comparisons', []))
    if state.get('correction_mode'):
        progress_text = "Correction Mode"
    else:
        progress_text = f"Comparison {state.get('index', 0) + 1} of {total}" if total > 0 else "Comparison 0 of 0"
    progress_lbl = ctk.CTkLabel(header_bar, text=progress_text, font=BODY_FONT)
    progress_lbl.pack(side="right", anchor="e", padx=8)

    # Hover instruction label (moved below Compare)
    hover_inst = ctk.CTkLabel(main_content, text="Hover over a number to read meaning", font=HOVER_FONT)
    hover_inst.pack(anchor="w", padx=12, pady=(4,4))

    # Saaty suggestion
    suggestion = suggest_range_for_pair(a, b, state.get('ranked', []))
    ctk.CTkLabel(main_content, text=f"{suggestion}", font=("Arial", 12)).pack(anchor="w", padx=12, pady=(0,8))

    saaty_instruction = "Assign a value between the two parameters based on how much more important one is than the other using the Saaty scale."
    ctk.CTkLabel(main_content, text=saaty_instruction, wraplength=950, justify="left", font=SMALL_FONT).pack(anchor="w", padx=12, pady=(2,6))

    # grid of choices (1..9)
    choices = ctk.CTkFrame(main_content)
    choices.pack(pady=6, padx=12)

    def make_on_enter(text):
        def _on_enter(event=None):
            hover_inst.configure(text=text)
        return _on_enter

    def make_on_leave():
        def _on_leave(event=None):
            hover_inst.configure(text="Hover over a number to read meaning")
        return _on_leave

    for val in range(1, 10):
        txt = scale_phrase(val, a, b)
        btn = ctk.CTkButton(choices, text=str(val), width=44, height=44, command=lambda v=val: confirm_pair(a, b, v))
        btn.grid(row=0, column=val-1, padx=6)
        btn.bind("<Enter>", make_on_enter(txt))
        btn.bind("<Leave>", make_on_leave())

    # Layout: left = two stacked description cards / right = Saaty grid table
    layout = ctk.CTkFrame(main_content)
    layout.pack(fill="both", expand=True, padx=12, pady=(10, 0))
    layout.grid_columnconfigure(0, weight=2)
    layout.grid_columnconfigure(1, weight=1)

    left_frame = ctk.CTkFrame(layout)
    left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
    # ensure there are enough rows configured so grid behaves predictably
    left_frame.grid_rowconfigure(0, weight=1)
    left_frame.grid_rowconfigure(1, weight=0)
    left_frame.grid_rowconfigure(2, weight=1)

    # ---------- PADDING/WRAP FIXES APPLIED ONLY TO THESE TWO STACKED CARDS ----------
    card_top = ctk.CTkFrame(left_frame, corner_radius=8)
    card_top.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
    ctk.CTkLabel(card_top, text=a, font=("Arial", 12, "bold"), anchor="w").pack(anchor="nw", pady=(8, 0), padx=12)
    top_inner = ctk.CTkFrame(card_top, fg_color="transparent")
    top_inner.pack(fill="both", expand=True, padx=12, pady=(6, 12))
    desc_top = ctk.CTkLabel(
        top_inner,
        text=PARAMETER_DESCRIPTIONS[a],
        wraplength=520,
        justify="left",
        anchor="nw",
        font=BODY_FONT
    )
    desc_top.pack(fill="both", expand=True)

    # spacer (row 1)
    spacer = ctk.CTkFrame(left_frame, height=6, fg_color="transparent")
    spacer.grid(row=1, column=0, sticky="ew")

    card_bottom = ctk.CTkFrame(left_frame, corner_radius=8)
    card_bottom.grid(row=2, column=0, sticky="nsew", padx=6, pady=6)
    ctk.CTkLabel(card_bottom, text=b, font=("Arial", 12, "bold"), anchor="w").pack(anchor="nw", pady=(8, 0), padx=12)
    bottom_inner = ctk.CTkFrame(card_bottom, fg_color="transparent")
    bottom_inner.pack(fill="both", expand=True, padx=12, pady=(6, 12))
    desc_bottom = ctk.CTkLabel(
        bottom_inner,
        text=PARAMETER_DESCRIPTIONS[b],
        wraplength=520,
        justify="left",
        anchor="nw",
        font=BODY_FONT
    )
    desc_bottom.pack(fill="both", expand=True)
    # --------------------------------------------------------------------------------

    # Right side: Saaty grid table
    right_frame = ctk.CTkFrame(layout)
    right_frame.grid(row=0, column=1, rowspan=3, sticky="nsew", padx=(8, 0))
    right_frame.grid_rowconfigure(0, weight=0)
    right_frame.grid_rowconfigure(1, weight=1)

    ctk.CTkLabel(right_frame, text="Saaty Scale", font=("Arial", 13, "bold")).pack(anchor="nw", pady=(8, 4), padx=10)

    table = ctk.CTkFrame(right_frame)
    table.pack(fill="both", expand=True, padx=10, pady=6)

    # Table headers
    ctk.CTkLabel(table, text="Scale", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky="w", padx=(4, 12), pady=(4, 4))
    ctk.CTkLabel(table, text="Numerical Rating", font=("Arial", 12, "bold")).grid(row=0, column=1, sticky="e", padx=(4, 12), pady=(4, 4))

    scales = [
        ("Extreme Importance", 9),
        ("Very Strong to Extreme", 8),
        ("Very Strong Importance", 7),
        ("Strong to Very Strong", 6),
        ("Strong Importance", 5),
        ("Moderate to Strong", 4),
        ("Moderate Importance", 3),
        ("Equal to Moderate", 2),
        ("Equal Importance", 1)
    ]
    for i, (text, val) in enumerate(scales, start=1):
        ctk.CTkLabel(table, text=text, anchor="w", font=SMALL_FONT).grid(row=i, column=0, sticky="w", padx=(4, 12), pady=4)
        ctk.CTkLabel(table, text=str(val), anchor="e", font=SMALL_FONT).grid(row=i, column=1, sticky="e", padx=(4, 12), pady=4)

    # If in correction mode, show suggested allowed range and reason (no quick picks)
    if state.get('correction_mode') and state.get('correction_pair') is not None:
        pair = state['correction_pair']
        p_a, p_b = pair
        details = state.get('correction_details', {})
        # compute up-to-date implied, error, etc using current matrix/weights if available
        if state['matrix'] is not None and state['weights'] is not None:
            try:
                idx = {p: i for i, p in enumerate(state['ranked'])}
                i_idx = idx[p_a]
                j_idx = idx[p_b]
                aij = float(state['matrix'][i_idx, j_idx])
                implied = float(state['weights'][i_idx] / (state['weights'][j_idx] + 1e-12))
                err = abs(math.log(aij + 1e-12) - math.log(implied + 1e-12))
            except Exception:
                aij = float(details.get('given', 1))
                implied = float(details.get('implied', 1))
                err = float(details.get('error', 0))
        else:
            aij = float(details.get('given', 1))
            implied = float(details.get('implied', 1))
            err = float(details.get('error', 0))

        suggestion_info = suggest_allowed_range(p_a, p_b, state.get('ranked', []), implied, aij, err, n_params=len(state.get('ranked', [])))
        sugg_frame = ctk.CTkFrame(right_frame, corner_radius=8)
        sugg_frame.pack(fill="x", padx=8, pady=(8, 8))
        ctk.CTkLabel(sugg_frame, text=f"Suggested integer: {suggestion_info['suggested_int']}", font=("Arial", 14, "bold")).pack(anchor="w", padx=8, pady=(6,2))
        ctk.CTkLabel(sugg_frame, text=f"Consider values between {suggestion_info['low']} and {suggestion_info['high']}", font=("Arial", 12)).pack(anchor="w", padx=8, pady=(0,6))
        ctk.CTkLabel(sugg_frame, text=f"Reason: {suggestion_info['reason']}", font=("Arial", 11), wraplength=320, justify="left").pack(anchor="w", padx=8, pady=(0,8))

    # Centered Go Back button below the layout
    back_frame = ctk.CTkFrame(main_content)
    back_frame.pack(pady=12)
    if state.get('index', 0) == 0:
        ctk.CTkButton(back_frame, text="Go Back", command=start_ranking).pack()
    else:
        ctk.CTkButton(back_frame, text="Go Back", command=go_back_pairwise).pack()

# -------------------- Pairwise Confirmation & Corrections --------------------
def confirm_pair(a, b, val):
    desc = scale_phrase(val, a, b)
    if not messagebox.askyesno("Confirm Choice", f"You chose:\n\n{desc}\n\nConfirm?"):
        return

    state['pairwise'][(a, b)] = val
    if (b, a) in state['pairwise']:
        del state['pairwise'][(b, a)]

    if state.get('correction_mode'):
        ranked = state['ranked']
        A = build_matrix(ranked, state['pairwise'])
        w, CR, _ = compute_eigen_weights_cr(A)
        state['weights'], state['CR'], state['matrix'] = w, CR, A
        state['correction_mode'] = False
        state['correction_pair'] = None
        state['correction_details'] = None
        show_results()
    else:
        state['index'] = state.get('index', 0) + 1
        open_pairwise()

def go_back_pairwise():
    if state['index'] > 0:
        state['index'] -= 1
    open_pairwise()

def cancel_correction():
    state['correction_mode'] = False
    state['correction_pair'] = None
    state['correction_details'] = None
    show_results()

# -------------------- Finalization & Results --------------------
def finalize():
    ranked = state['ranked']
    A = build_matrix(ranked, state['pairwise'])
    w, CR, _ = compute_eigen_weights_cr(A)
    state['weights'], state['CR'], state['matrix'] = w, CR, A
    show_results()

def show_results():
    clear_screen()
    ctk.CTkLabel(main_content, text="Computed Weights", font=SUBHEADER_FONT).pack(anchor="w", pady=8, padx=12)
    if state['weights'] is None:
        ctk.CTkLabel(main_content, text="No results yet. Complete the comparisons to compute weights.", font=BODY_FONT).pack(anchor="w", padx=36)
        ctk.CTkButton(main_content, text="Back to Dashboard", command=dashboard, width=160).pack(pady=12)
        return

    for p, w in zip(state['ranked'], state['weights']):
        ctk.CTkLabel(main_content, text=f"{p}: {w:.4f}", font=BODY_FONT).pack(anchor="w", padx=36)

    color = "#1E8449" if state['CR'] <= 0.1 else "#C0392B"
    ctk.CTkLabel(main_content, text=f"Consistency Ratio: {state['CR']:.4f}", font=("Arial", 12, "bold"), text_color=color).pack(anchor="w", padx=36, pady=8)

    if state['CR'] > 0.1:
        ctk.CTkLabel(main_content, text="Most inconsistent pairs (click to edit):", font=("Arial", 12, "bold")).pack(anchor="w", pady=(8,4), padx=12)
        bad = top_inconsistencies(state['matrix'], state['ranked'], state['weights'], k=6)
        scroll_frame = ctk.CTkScrollableFrame(main_content, height=240)
        scroll_frame.pack(fill="x", padx=36, pady=6)
        for (p, q), err, given_val, implied in bad:
            user_given = state['pairwise'].get((p, q), state['pairwise'].get((q, p), None))
            displayed_given = user_given if user_given is not None else f"{given_val:.3g}"
            short_txt = f"{p} vs {q} — given: {displayed_given}, implied: {implied:.3g}, err: {err:.3f}"

            def make_callback(pair, given_v, implied_v, err_v):
                return lambda: open_correction_pair(pair, given_v, implied_v, err_v)

            btn = ctk.CTkButton(scroll_frame, text=short_txt, anchor="w", command=make_callback((p, q), displayed_given, implied, err))
            btn.pack(fill="x", pady=6, padx=6)

        ctk.CTkLabel(main_content, text="Tip: Click a pair to re-evaluate that comparison. Suggested integer and a conservative range are shown when you re-evaluate.", font=ITALIC_FONT).pack(anchor="w", padx=36, pady=6)
    else:
        ctk.CTkLabel(main_content, text="All main inconsistencies resolved. CR is acceptable (≤ 0.1).", font=("Arial", 11, "italic")).pack(anchor="w", padx=36, pady=6)

    btns = ctk.CTkFrame(main_content)
    btns.pack(pady=12)
    ctk.CTkButton(btns, text="Revisit All Comparisons", command=reopen_all_pairs, width=180).pack(side="left", padx=8)
    ctk.CTkButton(btns, text="Save Results", command=save_results, width=140, fg_color="#27AE60").pack(side="left", padx=8)
    ctk.CTkButton(btns, text="Back to Dashboard", command=dashboard, width=160).pack(side="left", padx=8)

def open_correction_pair(pair, given, implied, err):
    state['correction_mode'] = True
    state['correction_pair'] = pair
    state['correction_details'] = {'given': given, 'implied': implied, 'error': err}
    open_pairwise()

def reopen_all_pairs():
    state['index'] = 0
    state['correction_mode'] = False
    state['correction_pair'] = None
    state['correction_details'] = None
    open_pairwise()

# -------------------- Save Results --------------------
def save_results():
    if state['weights'] is None:
        messagebox.showwarning("No results", "No results to save. Run the AHP first.")
        return

    fname = filedialog.asksaveasfilename(defaultextension=".csv",
                                         filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt")])
    if not fname:
        return
    try:
        with open(fname, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Parameter", "Weight"])
            for p, w in zip(state['ranked'], state['weights']):
                writer.writerow([p, f"{w:.4f}"])
            writer.writerow([])
            writer.writerow(["Consistency Ratio", f"{state['CR']:.4f}"])
            writer.writerow([])
            writer.writerow(["Pairwise judgments (a compared to b)", "Value"])
            for (a, b), v in state['pairwise'].items():
                writer.writerow([f"{a} vs {b}", f"{v}"])
        messagebox.showinfo("Saved", f"Results saved to {fname}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# -------------------- Start App --------------------
if __name__ == '__main__':
    dashboard()
    app.mainloop()
