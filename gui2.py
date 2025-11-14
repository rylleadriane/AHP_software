# gui2.py — Multi-level AHP GUI for Water Security
# Full, expanded version with indented bordered cards for parameter/domain definitions
# Requires: numpy
# Run: python gui2.py

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import csv

# ------------------------------------------------------------------------------
# Window setup
# ------------------------------------------------------------------------------
window = tk.Tk()
window.title("Multi-level AHP — Water Security")
window.geometry("1150x850")
window.title("Water Security AHP")
window.configure(bg="#F9F9F9")

# Global fonts used across the UI. Chosen to be legible and consistent.
HEADER_FONT = ("Arial", 20, "bold")
SUBHEADER_FONT = ("Arial", 14, "bold")
BODY_FONT = ("Arial", 13)
ITALIC_FONT = ("Arial", 11, "italic")
SMALL_FONT = ("Arial", 11)

# ------------------------------------------------------------------------------
# Data: domains, parameters, and descriptions
# ------------------------------------------------------------------------------
DOMAINS = {
    "Water Services and Public Health": [
        "Connection to Water Supply System",
        "Water-borne Disease Factor",
        "Drinking Water Samples Meeting Safety Standards",
        "Access to Improved Sanitation Services"
    ],
    "Water Resources and Environment": [
        "Vegetative Cover",
        "Groundwater Recharge Zone Index",
        "Water Demand-to-Supply Ratio",
        "Precipitation"
    ],
    "Socio-economics and Water Management": [
        "Number of Illegal Water Use",
        "Performance Factor",
        "Service Quality Factor",
        "Willingness to Pay for Improved Water Supply Services"
    ]
}

# Domain descriptions: match keys exactly with DOMAINS keys
DOMAIN_DESCRIPTIONS = {
    "Water Services and Public Health": (
        "This domain reflects the adequacy, safety, and accessibility of water and sanitation services, "
        "as well as their influence on public health outcomes. It captures how effectively the water service "
        "provider supports human well-being and safeguards residents from water-related diseases. "
        "A secure water system ensures reliable supply, safe drinking water, and proper sanitation — essential "
        "components of overall water security."
    ),
    "Water Resources and Environment": (
        "This domain evaluates the natural and environmental dimensions of water security, emphasizing the "
        "sustainability and resilience of the city’s water resources. It integrates ecological factors such as "
        "vegetation, rainfall, and recharge capacity that influence water availability, quality, and ecosystem health."
    ),
    "Socio-economics and Water Management": (
        "This domain captures the management, institutional, and social dimensions of water security. It considers "
        "how management performance, consumer behavior, and economic willingness shape the city’s capacity to "
        "sustain reliable and equitable water services. Effective management and public participation are vital "
        "for long-term water system resilience."
    )
}

# Parameter descriptions — updated and polished (your requested wording)
PARAMETER_DESCRIPTIONS = {
    "Connection to Water Supply System": (
        "Represents the population’s level of access to safe and reliable piped water. "
        "A higher connection rate reflects broader service coverage, improved distribution efficiency, "
        "and enhanced accessibility of water services to households."
    ),

    "Water-borne Disease Factor": (
        "Reflects the prevalence of diseases linked to unsafe or contaminated water sources. "
        "It indicates the effectiveness of water quality management and the degree of public health protection "
        "associated with the city’s water services."
    ),

    "Drinking Water Samples Meeting Safety Standards": (
        "Measures the compliance of drinking water with established microbial and chemical safety standards. "
        "It directly reflects the quality and safety dimension of water security."
    ),

    "Access to Improved Sanitation Services": (
        "Indicates the proportion of the population with access to adequate and safely managed sanitation facilities, "
        "which helps prevent contamination of water sources and supports public health resilience."
    ),

    "Vegetative Cover": (
        "Represents the extent of natural or managed vegetation that supports watershed protection, soil stability, "
        "and groundwater recharge. High vegetative cover strengthens the environmental pillar of water security "
        "by maintaining ecosystem functions."
    ),

    "Groundwater Recharge Zone Index": (
        "Quantifies the potential of land areas to facilitate infiltration and replenish groundwater reserves. "
        "It serves as an environmental indicator of the sustainability and availability of water resources."
    ),

    "Water Demand-to-Supply Ratio": (
        "Expresses the balance between total water consumption and available water resources. "
        "A high ratio indicates stress or potential scarcity, making it a key measure of water availability security."
    ),

    "Precipitation": (
        "Represents the amount of rainfall contributing to the city’s surface and groundwater resources. "
        "As a climatic determinant, it influences water availability, recharge, and the overall hydrologic balance."
    ),

    "Number of Illegal Water Use": (
        "Indicates the prevalence of unauthorized or non-revenue water connections. "
        "It serves as a proxy for governance efficiency, reflecting management control and institutional integrity "
        "in the water sector."
    ),

    "Performance Factor": (
        "A composite measure of the water service provider’s operational efficiency, including collection rate, "
        "non-revenue water reduction, and service continuity. High performance signifies institutional reliability "
        "and strong water governance."
    ),

    "Service Quality Factor": (
        "Reflects consumer satisfaction and the reliability of water supply in terms of pressure, continuity, "
        "and responsiveness. It represents the social and service delivery dimension of water security."
    ),

    "Willingness to Pay for Improved Water Supply Services": (
        "Captures the public’s perceived value of water services and their readiness to support service enhancement "
        "through higher tariffs. It reflects economic resilience and social engagement in sustaining water security improvements."
    )
}

# Saaty phrase dictionary (used for hover text)
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

# ------------------------------------------------------------------------------
# State storage (keeps the progress and temporary selections)
# ------------------------------------------------------------------------------
domain_progress = {
    d: {
        'status': 'pending',          # pending | done
        'temp_rank_vars': None,       # temporary StringVar map during ranking
        'ranked_params': [],          # ordered list after confirm
        'comparisons': [],            # list of pairs to compare
        'pairwise': {},               # dict: (p,q) -> value
        'index': 0,                   # current comparison index
        'weights': None,              # final weights array
        'CR': None,                   # consistency ratio
        'matrix': None                # AHP matrix used
    } for d in DOMAINS
}

domain_level_result = {
    'temp_rank_vars': None,
    'ranked_domains': [],
    'comparisons': [],
    'pairwise': {},
    'index': 0,
    'weights': None,
    'CR': None,
    'matrix': None
}

# ------------------------------------------------------------------------------
# AHP math helpers: building matrices, computing eigenvector weights and CR
# ------------------------------------------------------------------------------
def build_ranked_order(rank_vars):
    """
    Convert a mapping param -> Tk StringVar into a sorted list of parameters
    in ascending rank order (1 = most important).
    """
    items = []
    for p, var in rank_vars.items():
        try:
            r = int(var.get())
        except Exception:
            r = 99
        items.append((p, r))
    items.sort(key=lambda x: x[1])
    return [p for p, _ in items]


def build_comparisons_from_ranked(ranked):
    """
    Given a ranked list [p1, p2, p3, ...], produce the list of unique pairs
    for pairwise comparisons in the rank-based order:
        (p1,p2), (p1,p3), ..., (p2,p3), ...
    """
    comps = []
    n = len(ranked)
    for i in range(n):
        for j in range(i+1, n):
            comps.append((ranked[i], ranked[j]))
    return comps


def build_matrix(params, pairwise):
    """
    Build the reciprocal AHP matrix from parameters list and pairwise dict.
    pairwise keys are tuples (p, q) where p was judged against q with value v.
    """
    n = len(params)
    A = np.ones((n, n), dtype=float)
    idx = {p: i for i, p in enumerate(params)}
    for (p, q), v in pairwise.items():
        try:
            val = float(v)
        except Exception:
            val = 1.0
        A[idx[p], idx[q]] = val
        A[idx[q], idx[p]] = 1.0 / val
    return A


def compute_eigen_weights_cr(A):
    """
    Compute eigenvector weights and Consistency Ratio (CR) from matrix A.
    Returns (w, CR, eigval).
    """
    vals, vecs = np.linalg.eig(A)
    # choose eigenvalue with the largest real part
    max_idx = np.argmax(vals.real)
    eigval = vals.real[max_idx]
    eigvec = np.abs(vecs[:, max_idx].real)
    # normalize eigenvector
    w = eigvec / np.sum(eigvec)
    n = A.shape[0]
    CI = (eigval - n) / (n - 1) if n > 1 else 0.0
    RI_table = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32}
    RI = RI_table.get(n, 1.12)
    CR = CI / RI if RI != 0 else 0.0
    return w, CR, eigval


def scale_phrase(val, a, b, level="parameter"):
    """
    Produce a readable phrase for a given Saaty scale val comparing a to b.
    level can be 'parameter' or 'domain' to alter wording slightly.
    """
    term = "domain" if level == "domain" else "parameter"
    if val == 1:
        return f"The {term} {a} and the {term} {b} {SAATY_PHRASES[val]}"
    else:
        return f"The {term} {a} {SAATY_PHRASES[val]} the {term} {b}."


def top_inconsistencies(A, params, w, k=4):
    """
    Return the top-k most inconsistent pairs (highest log-difference between given a_ij and implied w_i/w_j).
    Each entry: ((param_i, param_j), diff, given_value, implied_value)
    """
    entries = []
    n = len(params)
    for i in range(n):
        for j in range(i+1, n):
            aij = A[i, j]
            implied = w[i] / w[j]
            # log-diff metric to capture relative multiplicative error
            diff = abs(np.log(aij + 1e-12) - np.log(implied + 1e-12))
            entries.append(((params[i], params[j]), diff, aij, implied))
    entries.sort(key=lambda x: x[1], reverse=True)
    return entries[:k]

# ------------------------------------------------------------------------------
# UI helpers
# ------------------------------------------------------------------------------
def clear_screen():
    """
    Remove all widgets from the main window — used to render each screen from scratch.
    """
    for w in window.winfo_children():
        w.destroy()


def header_bar(title="Water Security AHP", subtitle="Analytic Hierarchy Process"):
    """
    Create a header bar across the top of the window.
    """
    header = tk.Frame(window, bg="#2C5F8A", height=80)
    header.pack(fill="x")
    tk.Label(header, text=title, bg="#2C5F8A", fg="white", font=HEADER_FONT).pack(anchor="w", padx=20, pady=(10, 0))
    tk.Label(header, text=subtitle, bg="#2C5F8A", fg="white", font=SMALL_FONT).pack(anchor="w", padx=20)

# ------------------------------------------------------------------------------
# Dashboard screen
# ------------------------------------------------------------------------------
def dashboard_screen():
    """
    The main dashboard that lists each domain and allows the user to start or continue
    the AHP workflow for each domain.
    """
    clear_screen()
    header_bar()
    frame = tk.Frame(window, bg="#F9F9F9")
    frame.pack(fill="both", expand=True, padx=20, pady=14)

    # Title and instruction
    tk.Label(frame, text="Domains Dashboard", bg="#F9F9F9", font=SUBHEADER_FONT).pack(anchor="w", pady=(8, 10))
    tk.Label(frame,
             text="Complete each domain. Consistency Ratio shown for completed domains (acceptable if ≤ 0.1).",
             bg="#F9F9F9", font=BODY_FONT).pack(anchor="w", pady=(0, 12))

    # Domain cards
    for d in DOMAINS:
        dp = domain_progress[d]

        card = tk.Frame(frame, bg="#FFFFFF", bd=1, relief="solid")
        card.pack(fill="x", pady=8, padx=8)

        left = tk.Frame(card, bg="#FFFFFF")
        left.pack(side="left", fill="both", expand=True)

        tk.Label(left, text=d, bg="#FFFFFF", font=("Arial", 14, "bold")).pack(anchor="w", padx=14, pady=10)

        status = dp['status']
        cr_text = ""
        cr_color = "#666666"
        if dp['CR'] is not None:
            if dp['CR'] <= 0.1:
                cr_text = f"Consistency Ratio = {dp['CR']:.4f} (acceptable)"
                cr_color = "#1E8449"
            else:
                cr_text = f"Consistency Ratio = {dp['CR']:.4f} (inconsistent)"
                cr_color = "#C0392B"

        tk.Label(left, text=f"Status: {status.upper()}    {cr_text}", bg="#FFFFFF", fg=cr_color, font=BODY_FONT).pack(anchor="w", padx=14, pady=(0, 8))

        right = tk.Frame(card, bg="#FFFFFF")
        right.pack(side="right", padx=12)

        # Start/Continue button
        tk.Button(right, text="Start / Continue", bg="#3498DB", fg="white", font=("Arial", 12, "bold"),
                  command=lambda dn=d: start_domain_ranking(dn)).pack(side="right", padx=6, pady=12)

        # View results button only if domain done
        if status == 'done':
            tk.Button(right, text="View Results", bg="#EAEAEA", font=("Arial", 11),
                      command=lambda dn=d: show_domain_results(dn)).pack(side="right", padx=6, pady=12)

    # If all domains done, allow proceeding to domain-level AHP
    if all(domain_progress[d]['status'] == 'done' for d in DOMAINS):
        tk.Button(frame, text="Proceed to Domain-level AHP", bg="#27AE60", fg="white", font=("Arial", 13, "bold"),
                  command=start_domain_level_ranking).pack(pady=16)

    # Exit button pinned to bottom
    tk.Button(window, text="Exit", font=("Arial", 12, "bold"), command=window.destroy).pack(side="bottom", pady=14)


# ------------------------------------------------------------------------------
# Domain ranking screen (per domain) — contains parameter definitions cards
# ------------------------------------------------------------------------------
def start_domain_ranking(domain):
    """
    Shows the ranking form for parameters in a domain and displays parameter descriptions
    below the Confirm/Back buttons as indented bordered cards.
    """
    clear_screen()
    header_bar(title=f"AHP — {domain}", subtitle="Rank parameters 1 (most) → 4 (least)")
    frame = tk.Frame(window, bg="#F9F9F9")
    frame.pack(fill="both", expand=True, padx=18, pady=12)

    # Title and instruction
    tk.Label(frame, text=f"AHP — {domain}", bg="#F9F9F9", font=SUBHEADER_FONT).pack(anchor="w", pady=8)
    tk.Label(frame, text="Rank parameters from 1 (most important) to 4 (least important):", bg="#F9F9F9",
             font=BODY_FONT).pack(anchor="w", pady=(0, 8))

    # Ranking form
    rank_vars = {}
    form = tk.Frame(frame, bg="#F9F9F9")
    form.pack(anchor="w", padx=12, pady=6)
    for p in DOMAINS[domain]:
        row = tk.Frame(form, bg="#F9F9F9")
        row.pack(anchor="w", pady=6)
        tk.Label(row, text=p, bg="#F9F9F9", width=70, anchor="w",
                 wraplength=700, font=BODY_FONT).pack(side="left")
        var = tk.StringVar(value="1")
        cb = ttk.Combobox(row, textvariable=var, values=["1", "2", "3", "4"], width=4, state="readonly")
        cb.pack(side="left", padx=8)
        rank_vars[p] = var

    # keep the temp vars so the confirm screen can read them
    domain_progress[domain]['temp_rank_vars'] = rank_vars

    # Buttons: Confirm and Back
    btn_frame = tk.Frame(frame, bg="#F9F9F9")
    btn_frame.pack(pady=14)
    tk.Button(btn_frame, text="Confirm Ranking", bg="#27AE60", fg="white", font=("Arial", 12, "bold"),
              command=lambda dn=domain: confirm_domain_ranking(dn)).pack(side="left", padx=6)
    tk.Button(btn_frame, text="Back", font=("Arial", 11), command=dashboard_screen).pack(side="left", padx=6)

    # --------------------------------------------------------------------------
    # Parameter definitions section: show each parameter in an indented, bordered card
    # The visual effect is a narrow blue accent at the left and a white card body.
    # --------------------------------------------------------------------------
    defs_frame = tk.Frame(frame, bg="#F9F9F9")
    defs_frame.pack(fill="x", padx=20, pady=(10, 20))

    tk.Label(defs_frame, text="Parameter Descriptions", bg="#F9F9F9", font=("Arial", 12, "bold")).pack(anchor="w", pady=(4, 8))

    for param in DOMAINS[domain]:
        desc = PARAMETER_DESCRIPTIONS.get(param, "")

        # container: left accent + card
        container = tk.Frame(defs_frame, bg="#F9F9F9")
        container.pack(fill="x", padx=10, pady=6, anchor="w")

        # left accent narrow frame
        accent = tk.Frame(container, bg="#2C5F8A", width=6, height=1)
        accent.pack(side="left", fill="y", padx=(10, 0), pady=0)

        # card body (white)
        card = tk.Frame(container, bg="white")
        card.pack(side="left", fill="x", expand=True, padx=(8, 12))

        # Title (parameter name) and description
        tk.Label(card, text=param, bg="white", font=("Arial", 11, "bold"), anchor="w", justify="left").pack(anchor="w", pady=(6, 0))
        tk.Label(card, text=desc, bg="white", wraplength=980, justify="left", font=("Arial", 11)).pack(anchor="w", pady=(4, 8))

# ------------------------------------------------------------------------------
# Confirm rankings screen (per domain)
# ------------------------------------------------------------------------------
def confirm_domain_ranking(domain):
    """
    After user selects ranks, show a confirmation screen listing parameters
    highest to lowest. From here they can proceed to pairwise comparisons.
    """
    rank_vars = domain_progress[domain].get('temp_rank_vars')
    if not rank_vars:
        messagebox.showerror("Error", "No ranking data found.")
        start_domain_ranking(domain)
        return

    ranked = build_ranked_order(rank_vars)
    domain_progress[domain]['ranked_params'] = ranked

    clear_screen()
    header_bar(title=f"Confirm Ranking — {domain}")
    frame = tk.Frame(window, bg="#F9F9F9")
    frame.pack(fill="both", expand=True, padx=18, pady=12)

    tk.Label(frame, text="Confirm Ranking (highest → lowest)", bg="#F9F9F9", font=SUBHEADER_FONT).pack(anchor="w", pady=8)

    # show ranked list
    for i, p in enumerate(ranked, start=1):
        tk.Label(frame, text=f"{i}. {p}", bg="#F9F9F9", font=BODY_FONT).pack(anchor="w", padx=20)

    # navigation
    btn_frame = tk.Frame(frame, bg="#F9F9F9")
    btn_frame.pack(pady=14)
    tk.Button(btn_frame, text="Proceed to Pairwise Comparison", bg="#27AE60", fg="white",
              font=("Arial", 12, "bold"),
              command=lambda dn=domain: start_pairwise_for_domain(dn)).pack(side="left", padx=6)
    tk.Button(btn_frame, text="Go Back", font=("Arial", 11), command=lambda dn=domain: start_domain_ranking(dn)).pack(side="left", padx=6)


# ------------------------------------------------------------------------------
# Pairwise comparison flow (per domain)
# ------------------------------------------------------------------------------
def start_pairwise_for_domain(domain):
    """
    Prepare comparisons from the ranked parameters and open the first pairwise screen.
    """
    dp = domain_progress[domain]
    ranked = dp['ranked_params']
    comps = build_comparisons_from_ranked(ranked)
    dp['comparisons'] = comps
    dp['pairwise'] = {}
    dp['index'] = 0
    open_pairwise_screen(domain)


def open_pairwise_screen(domain):
    """
    Show the current pairwise comparison screen for domain,
    with buttons 1..9 and hover text showing Saaty interpretation.
    """
    dp = domain_progress[domain]
    comps = dp['comparisons']
    idx = dp['index']
    ranked = dp['ranked_params']

    clear_screen()
    header_bar(title=f"AHP — {domain}", subtitle="Pairwise comparisons (rank-based order)")

    # if we've finished all comparisons, finalize the domain-level AHP
    if idx >= len(comps):
        finalize_domain_ahp(domain)
        return

    a, b = comps[idx]

    frame = tk.Frame(window, bg="#F9F9F9")
    frame.pack(fill="both", expand=True, padx=18, pady=12)

    # short instruction
    tk.Label(frame, text=f"Compare the following pair (based on rank):", bg="#F9F9F9", font=BODY_FONT).pack(anchor="w", pady=(6, 6))

    # comparison card
    card = tk.Frame(frame, bg="#ECF0F1", bd=1, relief="solid", padx=16, pady=12)
    card.pack(pady=10, padx=8, fill="x")

    tk.Label(card, text=a, font=("Arial", 14, "bold"), fg="#2E86C1", bg="#ECF0F1").pack()
    tk.Label(card, text="vs", bg="#ECF0F1").pack()
    tk.Label(card, text=b, font=("Arial", 14, "bold"), fg="#E67E22", bg="#ECF0F1").pack(pady=(0, 8))

    # show parameter descriptions above the Saaty scale
    tk.Label(frame, text=f"{a}: {PARAMETER_DESCRIPTIONS.get(a, '')}", wraplength=980, justify="left",
             bg="#F9F9F9", font=BODY_FONT).pack(anchor="w", padx=6)
    tk.Label(frame, text=f"{b}: {PARAMETER_DESCRIPTIONS.get(b, '')}", wraplength=980, justify="left",
             bg="#F9F9F9", font=BODY_FONT).pack(anchor="w", padx=6, pady=(0, 8))

    # hover label to show Saaty phrase
    hover = tk.Label(frame, text="Hover over a number to read meaning", bg="#F9F9F9", font=ITALIC_FONT)
    hover.pack(pady=6, anchor="w", padx=6)

    # buttons 1..9
    btn_frame = tk.Frame(frame, bg="#F9F9F9")
    btn_frame.pack(pady=8, padx=6, anchor="w")
    for col, val in enumerate(range(1, 10)):
        txt = scale_phrase(val, a, b, level="parameter")
        btn = tk.Button(btn_frame, text=str(val), width=4, height=2, font=("Arial", 11),
                        command=lambda v=val: confirm_pairwise(domain, a, b, v))
        btn.grid(row=0, column=col, padx=6, pady=6)
        # bind mouse enter/leave for hover description
        btn.bind("<Enter>", lambda e, t=txt: hover.config(text=t))
        btn.bind("<Leave>", lambda e: hover.config(text="Hover over a number to read meaning"))

    # show hierarchy order at right (or below)
    tk.Label(frame, text="Order of importance (highest → lowest):", bg="#F9F9F9",
             font=("Arial", 11, "bold")).pack(pady=(12, 4), anchor="w", padx=6)
    hframe = tk.Frame(frame, bg="#F9F9F9")
    hframe.pack(anchor="w", padx=20)
    for j, param in enumerate(ranked, start=1):
        tk.Label(hframe, text=f"{j}. {param}", bg="#F9F9F9", anchor="w", justify="left", wraplength=900,
                 font=BODY_FONT).pack(anchor="w")

    # back navigation (to previous comparison or confirmation)
    back_frame = tk.Frame(frame, bg="#F9F9F9")
    back_frame.pack(pady=12)
    if idx == 0:
        tk.Button(back_frame, text="Go Back", font=("Arial", 11), command=lambda dn=domain: confirm_domain_ranking(dn)).pack()
    else:
        tk.Button(back_frame, text="Go Back", font=("Arial", 11), command=lambda dn=domain: go_back_pairwise(dn)).pack()


def confirm_pairwise(domain, a, b, val):
    """
    Confirm user selection for a pairwise comparison, store it, and advance index.
    """
    desc = scale_phrase(val, a, b, level="parameter")
    if messagebox.askyesno("Confirm Choice", f"You chose:\n\n{desc}\n\nConfirm?"):
        domain_progress[domain]['pairwise'][(a, b)] = val
        domain_progress[domain]['index'] += 1
        open_pairwise_screen(domain)


def go_back_pairwise(domain):
    """
    Move back one comparison index and reopen the pairwise screen.
    """
    if domain_progress[domain]['index'] > 0:
        domain_progress[domain]['index'] -= 1
    open_pairwise_screen(domain)


def finalize_domain_ahp(domain):
    """
    Build matrix, compute eigenvector weights and CR for the domain, save results,
    and mark domain status as done.
    """
    dp = domain_progress[domain]
    ranked = dp['ranked_params']
    A = build_matrix(ranked, dp['pairwise'])
    w, CR, _ = compute_eigen_weights_cr(A)
    dp['weights'] = w
    dp['CR'] = CR
    dp['matrix'] = A
    dp['status'] = 'done'

    # show results screen
    show_domain_results(domain)


# ------------------------------------------------------------------------------
# Domain results and utilities
# ------------------------------------------------------------------------------
def show_domain_results(domain):
    """
    Display parameter weights and CR for a completed domain. If inconsistent,
    show the top inconsistent pairs as clickable buttons allowing the user to
    edit that pair.
    """
    dp = domain_progress[domain]
    clear_screen()
    header_bar(title=f"AHP Results — {domain}")

    frame = tk.Frame(window, bg="#F9F9F9")
    frame.pack(fill="both", expand=True, padx=18, pady=12)

    tk.Label(frame, text=f"AHP Results — {domain}", bg="#F9F9F9", font=SUBHEADER_FONT).pack(anchor="w", pady=8)

    if dp['weights'] is None:
        tk.Label(frame, text="No results yet.", bg="#F9F9F9", font=BODY_FONT).pack(anchor="w")
        tk.Button(frame, text="Back", font=("Arial", 11), command=dashboard_screen).pack(pady=10)
        return

    # Display weights with index
    for i, (p, w) in enumerate(zip(dp['ranked_params'], dp['weights']), start=1):
        tk.Label(frame, text=f"{i}. {p}: {w:.4f}", bg="#F9F9F9", font=BODY_FONT).pack(anchor="w", padx=36)

    # Show consistency ratio
    cr_color = "#1E8449" if dp['CR'] <= 0.1 else "#C0392B"
    tk.Label(frame, text=f"Consistency Ratio: {dp['CR']:.4f}", bg="#F9F9F9",
             font=("Arial", 12, "bold"), fg=cr_color).pack(pady=10, anchor="w", padx=36)

    # If inconsistent, show top inconsistent pairs (click to edit)
    if dp['CR'] > 0.1:
        bad = top_inconsistencies(dp['matrix'], dp['ranked_params'], dp['weights'], k=4)
        tk.Label(frame, text="Most inconsistent pairs (click to edit):", bg="#F9F9F9",
                 font=("Arial", 12, "bold")).pack(anchor="w", pady=(6, 4))
        for (p, q), err, given, implied in bad:
            txt = f"{p} vs {q} — given: {given:.3g}, implied: {implied:.3g}, error: {err:.3f}"
            b = tk.Button(frame, text=txt, bg="#FAD7A0", wraplength=900, justify="left", font=SMALL_FONT,
                          command=lambda pair=(p, q), dn=domain: open_specific_pair_for_domain(dn, pair))
            b.pack(fill="x", padx=40, pady=4)

    # Buttons: revisit comparisons and back to dashboard
    btns = tk.Frame(frame, bg="#F9F9F9")
    btns.pack(pady=12)
    tk.Button(btns, text="Revisit Comparisons", bg="#F39C12", fg="white", font=("Arial", 11, "bold"),
              command=lambda dn=domain: reopen_domain_pairwise(dn)).pack(side="left", padx=8)
    tk.Button(btns, text="Back to Domains", font=("Arial", 11), command=dashboard_screen).pack(side="left", padx=8)


def open_specific_pair_for_domain(domain, pair):
    """
    If a top inconsistent pair is clicked, set the index to match that pair and open pairwise editing.
    """
    comps = domain_progress[domain]['comparisons']
    try:
        idx = comps.index(pair)
    except ValueError:
        try:
            idx = comps.index((pair[1], pair[0]))
        except ValueError:
            messagebox.showerror("Not found", "Comparison not found.")
            return
    domain_progress[domain]['index'] = idx
    open_pairwise_screen(domain)


def reopen_domain_pairwise(domain):
    """
    Reset index to zero and reopen pairwise comparison sequence.
    """
    domain_progress[domain]['index'] = 0
    open_pairwise_screen(domain)


# ------------------------------------------------------------------------------
# Domain-level AHP (rank the 3 domains, show domain description cards)
# ------------------------------------------------------------------------------
def start_domain_level_ranking():
    """
    Let the user rank the three domains. Under the Confirm/Back buttons display
    domain descriptions in indented bordered cards similar to parameter cards.
    """
    clear_screen()
    header_bar(title="Domain-level AHP", subtitle="Rank the domains 1 (most) → 3 (least)")
    frame = tk.Frame(window, bg="#F9F9F9")
    frame.pack(fill="both", expand=True, padx=18, pady=12)

    tk.Label(frame, text="Rank the three domains from 1 (most important) to 3 (least important):",
             bg="#F9F9F9", font=BODY_FONT).pack(anchor="w", pady=8)

    rank_vars = {}
    form = tk.Frame(frame, bg="#F9F9F9")
    form.pack(anchor="w", padx=12, pady=6)

    for d in DOMAINS.keys():
        row = tk.Frame(form, bg="#F9F9F9")
        row.pack(anchor="w", pady=6)
        tk.Label(row, text=d, bg="#F9F9F9", width=60, anchor="w", font=BODY_FONT).pack(side="left")
        var = tk.StringVar(value="1")
        cb = ttk.Combobox(row, textvariable=var, values=["1", "2", "3"], width=4, state="readonly")
        cb.pack(side="left", padx=8)
        rank_vars[d] = var

    domain_level_result['temp_rank_vars'] = rank_vars

    # Confirm / Back buttons
    btn_frame = tk.Frame(frame, bg="#F9F9F9")
    btn_frame.pack(pady=12)
    tk.Button(btn_frame, text="Confirm Ranking", bg="#27AE60", fg="white", font=("Arial", 12, "bold"),
              command=confirm_domain_level_ranking).pack(side="left", padx=6)
    tk.Button(btn_frame, text="Back", font=("Arial", 11), command=dashboard_screen).pack(side="left", padx=6)

    # --------------------------------------------------------------------------
    # Domain descriptions shown as indented cards with left blue accent
    # --------------------------------------------------------------------------
    defs_frame = tk.Frame(frame, bg="#F9F9F9")
    defs_frame.pack(fill="x", padx=20, pady=(10, 20))

    tk.Label(defs_frame, text="Domain Descriptions", bg="#F9F9F9", font=("Arial", 12, "bold")).pack(anchor="w", pady=(4, 8))

    for d in DOMAINS.keys():
        desc = DOMAIN_DESCRIPTIONS.get(d, "")

        container = tk.Frame(defs_frame, bg="#F9F9F9")
        container.pack(fill="x", padx=10, pady=6, anchor="w")

        accent = tk.Frame(container, bg="#2C5F8A", width=6)
        accent.pack(side="left", fill="y", padx=(10, 0))

        card = tk.Frame(container, bg="white")
        card.pack(side="left", fill="x", expand=True, padx=(8, 12))

        tk.Label(card, text=d, bg="white", font=("Arial", 11, "bold"), anchor="w", justify="left").pack(anchor="w", pady=(6, 0))
        tk.Label(card, text=desc, bg="white", wraplength=980, justify="left", font=("Arial", 11)).pack(anchor="w", pady=(4, 8))


def confirm_domain_level_ranking():
    """
    Confirm the domain ranking and proceed to pairwise comparisons for domain-level AHP.
    """
    rank_vars = domain_level_result.get('temp_rank_vars')
    if not rank_vars:
        messagebox.showerror("Error", "No ranking data.")
        start_domain_level_ranking()
        return

    ranked = build_ranked_order(rank_vars)
    domain_level_result['ranked_domains'] = ranked

    # Confirm ranking screen
    clear_screen()
    header_bar(title="Confirm Domain Ranking")
    frame = tk.Frame(window, bg="#F9F9F9")
    frame.pack(fill="both", expand=True, padx=18, pady=12)

    tk.Label(frame, text="Confirm Domain Ranking", bg="#F9F9F9", font=SUBHEADER_FONT).pack(anchor="w", pady=8)

    for i, d in enumerate(ranked, start=1):
        tk.Label(frame, text=f"{i}. {d}", bg="#F9F9F9", font=BODY_FONT).pack(anchor="w", padx=20)

    btn_frame = tk.Frame(frame, bg="#F9F9F9")
    btn_frame.pack(pady=12)
    tk.Button(btn_frame, text="Proceed to Pairwise", bg="#27AE60", fg="white", font=("Arial", 12, "bold"),
              command=start_domain_level_pairwise).pack(side="left", padx=6)
    tk.Button(btn_frame, text="Go Back", font=("Arial", 11), command=start_domain_level_ranking).pack(side="left", padx=6)


def start_domain_level_pairwise():
    """
    Build the pairwise comparison list for domains and open the first pair for comparison.
    """
    ranked = domain_level_result['ranked_domains']
    comps = build_comparisons_from_ranked(ranked)
    domain_level_result['comparisons'] = comps
    domain_level_result['pairwise'] = {}
    domain_level_result['index'] = 0
    open_domain_level_pairwise()


def open_domain_level_pairwise():
    """
    Show the pairwise comparison screen for domains (1..9 scale), with hover text.
    """
    idx = domain_level_result['index']
    comps = domain_level_result['comparisons']
    ranked = domain_level_result['ranked_domains']

    clear_screen()
    header_bar(title="Domain-level Pairwise")

    if idx >= len(comps):
        finalize_domain_level()
        return

    a, b = comps[idx]

    frame = tk.Frame(window, bg="#F9F9F9")
    frame.pack(fill="both", expand=True, padx=18, pady=12)

    tk.Label(frame, text=f"Compare the domains: {a} vs {b}", bg="#F9F9F9", font=SUBHEADER_FONT).pack(anchor="w", pady=(6, 8))

    hover = tk.Label(frame, text="Hover over a number to read meaning", bg="#F9F9F9", font=ITALIC_FONT)
    hover.pack(anchor="w", padx=8, pady=(0, 8))

    btn_frame = tk.Frame(frame, bg="#F9F9F9")
    btn_frame.pack(anchor="w", padx=8, pady=6)
    for col, val in enumerate(range(1, 10)):
        txt = scale_phrase(val, a, b, level="domain")
        btn = tk.Button(btn_frame, text=str(val), width=4, height=2, font=("Arial", 11),
                        command=lambda v=val: confirm_domain_level_choice(a, b, v))
        btn.grid(row=0, column=col, padx=6, pady=6)
        btn.bind("<Enter>", lambda e, t=txt: hover.config(text=t))
        btn.bind("<Leave>", lambda e: hover.config(text="Hover over a number to read meaning"))

    tk.Label(frame, text="Order of importance (highest → lowest):", bg="#F9F9F9",
             font=("Arial", 11, "bold")).pack(pady=(10, 4), anchor="w")
    for i, d in enumerate(ranked, start=1):
        tk.Label(frame, text=f"{i}. {d}", bg="#F9F9F9", font=BODY_FONT).pack(anchor="w", padx=20)

    if idx == 0:
        tk.Button(frame, text="Go Back", font=("Arial", 11), command=confirm_domain_level_ranking).pack(pady=12)
    else:
        tk.Button(frame, text="Go Back", font=("Arial", 11), command=domain_level_back).pack(pady=12)


def confirm_domain_level_choice(a, b, v):
    desc = scale_phrase(v, a, b, level="domain")
    if messagebox.askyesno("Confirm Choice", f"You chose:\n\n{desc}\n\nConfirm?"):
        domain_level_result['pairwise'][(a, b)] = v
        domain_level_result['index'] += 1
        open_domain_level_pairwise()


def domain_level_back():
    if domain_level_result['index'] > 0:
        domain_level_result['index'] -= 1
    open_domain_level_pairwise()


def finalize_domain_level():
    """
    Compute the domain-level weights and CR; then show the domain-level results.
    """
    ranked = domain_level_result['ranked_domains']
    A = build_matrix(ranked, domain_level_result['pairwise'])
    w, CR, _ = compute_eigen_weights_cr(A)
    domain_level_result['weights'] = w
    domain_level_result['CR'] = CR
    domain_level_result['matrix'] = A
    show_domain_level_results()


def show_domain_level_results():
    """
    Display computed domain weights and consistency ratio for the 3 domains.
    """
    clear_screen()
    header_bar(title="AHP Results — Water Security Domains")
    frame = tk.Frame(window, bg="#F9F9F9")
    frame.pack(fill="both", expand=True, padx=18, pady=12)

    tk.Label(frame, text="AHP Results — Water Security Domains", bg="#F9F9F9", font=SUBHEADER_FONT).pack(anchor="w", pady=8)

    if domain_level_result['weights'] is None:
        tk.Label(frame, text="No domain-level result yet.", bg="#F9F9F9", font=BODY_FONT).pack(anchor="w", pady=6)
        tk.Button(frame, text="Back to Dashboard", font=("Arial", 11), command=dashboard_screen).pack(pady=8)
        return

    # show domain weights
    for d, w in zip(domain_level_result['ranked_domains'], domain_level_result['weights']):
        tk.Label(frame, text=f"{d}: {w:.4f}", bg="#F9F9F9", font=BODY_FONT).pack(anchor="w", padx=36)

    cr_color = "#1E8449" if domain_level_result['CR'] <= 0.1 else "#C0392B"
    tk.Label(frame, text=f"Consistency Ratio: {domain_level_result['CR']:.4f}",
             bg="#F9F9F9", font=("Arial", 12, "bold"), fg=cr_color).pack(pady=10, anchor="w", padx=36)

    btns = tk.Frame(frame, bg="#F9F9F9")
    btns.pack(pady=12)
    tk.Button(btns, text="AHP Summary", bg="#3498DB", fg="white", font=("Arial", 12, "bold"),
              command=ahp_summary_screen).pack(side="left", padx=8)
    tk.Button(btns, text="Back to Dashboard", font=("Arial", 11), command=dashboard_screen).pack(side="left", padx=8)


# ------------------------------------------------------------------------------
# AHP Summary and Save (CSV / TXT)
# ------------------------------------------------------------------------------
def ahp_summary_screen():
    """
    Summarize all domain-level and parameter-level weights and CR values.
    Allow user to save results to CSV or TXT.
    """
    clear_screen()
    header_bar(title="AHP Summary — All Results")
    frame = tk.Frame(window, bg="#F9F9F9")
    frame.pack(fill="both", expand=True, padx=18, pady=12)

    tk.Label(frame, text="AHP Summary — All Results", bg="#F9F9F9", font=SUBHEADER_FONT).pack(anchor="w", pady=8)

    # Show each domain and its parameter weights
    for d in DOMAINS:
        dp = domain_progress[d]
        tk.Label(frame, text=f"{d} — {dp['status'].upper()}", bg="#F9F9F9", font=("Arial", 12, "bold")).pack(anchor="w", padx=20, pady=(6, 2))
        if dp['status'] == 'done' and dp['weights'] is not None:
            for p, w in zip(dp['ranked_params'], dp['weights']):
                tk.Label(frame, text=f"    {p}: {w:.4f}", bg="#F9F9F9", font=BODY_FONT).pack(anchor="w", padx=48)
            tk.Label(frame, text=f"    Consistency Ratio: {dp['CR']:.4f}", bg="#F9F9F9", font=BODY_FONT).pack(anchor="w", padx=48, pady=(0, 6))
        else:
            tk.Label(frame, text="    (not completed)", bg="#F9F9F9", font=BODY_FONT).pack(anchor="w", padx=48, pady=(0, 6))

    # Domain-level results
    tk.Label(frame, text="", bg="#F9F9F9").pack()
    tk.Label(frame, text="Water Security Domains", bg="#F9F9F9", font=("Arial", 13, "bold")).pack(anchor="w", pady=(8, 4))
    if domain_level_result['weights'] is not None:
        for d, w in zip(domain_level_result['ranked_domains'], domain_level_result['weights']):
            tk.Label(frame, text=f"    {d}: {w:.4f}", bg="#F9F9F9", font=BODY_FONT).pack(anchor="w", padx=48)
        tk.Label(frame, text=f"    Consistency Ratio: {domain_level_result['CR']:.4f}", bg="#F9F9F9", font=BODY_FONT).pack(anchor="w", padx=48, pady=(0, 6))
    else:
        tk.Label(frame, text="    (domain-level AHP not yet completed)", bg="#F9F9F9", font=BODY_FONT).pack(anchor="w", padx=48, pady=(0, 6))

    btns = tk.Frame(frame, bg="#F9F9F9")
    btns.pack(pady=12)
    tk.Button(btns, text="Save Results", bg="#27AE60", fg="white", font=("Arial", 12, "bold"), command=save_results_dialog).pack(side="left", padx=8)
    tk.Button(btns, text="Back to Dashboard", font=("Arial", 11), command=dashboard_screen).pack(side="left", padx=8)


def save_results_dialog():
    """
    Present file dialog and call appropriate save function based on extension.
    """
    opts = [("CSV file", "*.csv"), ("Text file", "*.txt")]
    fname = filedialog.asksaveasfilename(title="Save AHP Summary", defaultextension=".csv", filetypes=opts)
    if not fname:
        return
    if fname.lower().endswith(".csv"):
        save_as_csv(fname)
    else:
        save_as_txt(fname)


def save_as_csv(path):
    """
    Save the full AHP summary to CSV. Format is simple and readable.
    """
    try:
        with open(path, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Item", "Weight", "Consistency Ratio"])
            for d in DOMAINS:
                dp = domain_progress[d]
                cr_val = f"{dp['CR']:.4f}" if dp['status'] == 'done' and dp['CR'] is not None else ""
                writer.writerow([d, "", cr_val])
                if dp['status'] == 'done' and dp['weights'] is not None:
                    for p, w in zip(dp['ranked_params'], dp['weights']):
                        writer.writerow([f"    {p}", f"{w:.4f}", ""])
                else:
                    writer.writerow(["    (not completed)", "", ""])
                writer.writerow([])

            # Domain-level section
            if domain_level_result['weights'] is not None:
                writer.writerow(["Water Security Domains", "", f"{domain_level_result['CR']:.4f}"])
                for d, w in zip(domain_level_result['ranked_domains'], domain_level_result['weights']):
                    writer.writerow([f"    {d}", f"{w:.4f}", ""])
            else:
                writer.writerow(["Water Security Domains", "", ""])
                writer.writerow(["    (not completed)", "", ""])

        messagebox.showinfo("Saved", f"Results saved to {path}")
    except Exception as e:
        messagebox.showerror("Save Error", str(e))


def save_as_txt(path):
    """
    Save the summary as a simple tab-separated text file.
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("Item\tWeight\tConsistency Ratio\n\n")
            for d in DOMAINS:
                dp = domain_progress[d]
                cr_val = f"{dp['CR']:.4f}" if dp['status'] == 'done' and dp['CR'] is not None else ""
                f.write(f"{d}\t\t{cr_val}\n")
                if dp['status'] == 'done' and dp['weights'] is not None:
                    for p, w in zip(dp['ranked_params'], dp['weights']):
                        f.write(f"    {p}\t{w:.4f}\t\n")
                else:
                    f.write("    (not completed)\t\t\n")
                f.write("\n")

            if domain_level_result['weights'] is not None:
                f.write(f"Water Security Domains\t\t{domain_level_result['CR']:.4f}\n")
                for d, w in zip(domain_level_result['ranked_domains'], domain_level_result['weights']):
                    f.write(f"    {d}\t{w:.4f}\t\n")
            else:
                f.write("Water Security Domains\t\t\n")
                f.write("    (not completed)\t\t\n")

        messagebox.showinfo("Saved", f"Results saved to {path}")
    except Exception as e:
        messagebox.showerror("Save Error", str(e))


# ------------------------------------------------------------------------------
# Start app
# ------------------------------------------------------------------------------
# Render the dashboard and enter Tk mainloop
dashboard_screen()
window.mainloop()
