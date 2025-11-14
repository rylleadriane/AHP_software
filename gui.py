# gui2.py (PART 1 of 2)
# Multi-level AHP GUI — Water Security
# Part 1/2: setup, data, per-domain flow
# Requires: numpy
# Save as gui2.py (concatenate Part 1 + Part 2) and run: python gui2.py

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import csv

# -----------------------
# Window setup
# -----------------------
window = tk.Tk()
window.title("Multi-level AHP — Water Security")
window.geometry("1150x850")
window.configure(bg="#F9F9F9")

# Global font sizes (adjusted slightly larger)
HEADER_FONT = ("Arial", 20, "bold")
SUBHEADER_FONT = ("Arial", 14, "bold")
BODY_FONT = ("Arial", 13)
ITALIC_FONT = ("Arial", 11, "italic")
SMALL_FONT = ("Arial", 11)

# -----------------------
# Data
# -----------------------
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

PARAMETER_DESCRIPTIONS = {
    "Connection to Water Supply System": "Represents the population’s level of access to safe and reliable piped water.",
    "Water-borne Disease Factor": "Reflects the prevalence of diseases linked to unsafe or contaminated water sources.",
    "Drinking Water Samples Meeting Safety Standards": "Measures compliance of drinking water with established microbial and chemical safety standards.",
    "Access to Improved Sanitation Services": "Indicates the proportion of the population with access to adequate and safely managed sanitation facilities.",
    "Vegetative Cover": "Represents the extent of natural or managed vegetation that supports watershed protection, soil stability, and groundwater recharge.",
    "Groundwater Recharge Zone Index": "Quantifies the potential of land areas to facilitate infiltration and replenish groundwater reserves.",
    "Water Demand-to-Supply Ratio": "Expresses the balance between total water consumption and available water resources.",
    "Precipitation": "Represents the amount of rainfall contributing to the city’s surface and groundwater resources.",
    "Number of Illegal Water Use": "Indicates the prevalence of unauthorized or non-revenue water connections.",
    "Performance Factor": "A composite measure of the water service provider’s operational efficiency.",
    "Service Quality Factor": "Reflects consumer satisfaction and the reliability of water supply in terms of pressure, continuity, and responsiveness.",
    "Willingness to Pay for Improved Water Supply Services": "Captures the public’s perceived value of water services and their readiness to support service enhancement through higher tariffs."
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

# -----------------------
# State storage
# -----------------------
domain_progress = {
    d: {
        'status': 'pending',
        'temp_rank_vars': None,
        'ranked_params': [],
        'comparisons': [],
        'pairwise': {},       # (p,q) -> value
        'index': 0,
        'weights': None,
        'CR': None,
        'matrix': None
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

# -----------------------
# AHP math helpers
# -----------------------
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

def build_comparisons_from_ranked(ranked):
    comps = []
    n = len(ranked)
    for i in range(n):
        for j in range(i+1, n):
            comps.append((ranked[i], ranked[j]))
    return comps

def build_matrix(params, pairwise):
    n = len(params)
    A = np.ones((n, n), dtype=float)
    idx = {p: i for i, p in enumerate(params)}
    for (p, q), v in pairwise.items():
        A[idx[p], idx[q]] = float(v)
        A[idx[q], idx[p]] = 1.0 / float(v)
    return A

def compute_eigen_weights_cr(A):
    vals, vecs = np.linalg.eig(A)
    max_idx = np.argmax(vals.real)
    eigval = vals.real[max_idx]
    eigvec = np.abs(vecs[:, max_idx].real)
    # normalize
    w = eigvec / np.sum(eigvec)
    n = A.shape[0]
    CI = (eigval - n) / (n - 1) if n > 1 else 0.0
    RI_table = {1:0.00, 2:0.00, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32}
    RI = RI_table.get(n, 1.12)
    CR = CI / RI if RI != 0 else 0.0
    return w, CR, eigval

def scale_phrase(val, a, b, level="parameter"):
    term = "domain" if level == "domain" else "parameter"
    if val == 1:
        return f"The {term} {a} and the {term} {b} {SAATY_PHRASES[val]}"
    else:
        return f"The {term} {a} {SAATY_PHRASES[val]} the {term} {b}."

def top_inconsistencies(A, params, w, k=4):
    entries = []
    n = len(params)
    for i in range(n):
        for j in range(i+1, n):
            aij = A[i, j]
            implied = w[i] / w[j]
            # use log-diff metric
            diff = abs(np.log(aij + 1e-12) - np.log(implied + 1e-12))
            entries.append(((params[i], params[j]), diff, aij, implied))
    entries.sort(key=lambda x: x[1], reverse=True)
    return entries[:k]

# -----------------------
# UI helpers
# -----------------------
def clear_screen():
    for w in window.winfo_children():
        w.destroy()

def header_bar(title="Water Security AHP", subtitle="Analytic Hierarchy Process"):
    header = tk.Frame(window, bg="#2C5F8A", height=80)
    header.pack(fill="x")
    tk.Label(header, text=title, bg="#2C5F8A", fg="white", font=HEADER_FONT).pack(anchor="w", padx=20, pady=(10, 0))
    tk.Label(header, text=subtitle, bg="#2C5F8A", fg="white", font=SMALL_FONT).pack(anchor="w", padx=20)

# -----------------------
# Dashboard
# -----------------------
def dashboard_screen():
    clear_screen()
    header_bar()
    frame = tk.Frame(window, bg="#F9F9F9")
    frame.pack(fill="both", expand=True, padx=20, pady=14)

    tk.Label(frame, text="Domains Dashboard", bg="#F9F9F9", font=SUBHEADER_FONT).pack(anchor="w", pady=(8, 10))
    tk.Label(frame, text="Complete each domain. Consistency Ratio shown for completed domains (acceptable if ≤ 0.1).",
             bg="#F9F9F9", font=BODY_FONT).pack(anchor="w", pady=(0, 12))

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
        tk.Button(right, text="Start / Continue", bg="#3498DB", fg="white", font=("Arial", 12, "bold"),
                  command=lambda dn=d: start_domain_ranking(dn)).pack(side="right", padx=6, pady=12)
        if status == 'done':
            tk.Button(right, text="View Results", bg="#EAEAEA", font=("Arial", 11),
                      command=lambda dn=d: show_domain_results(dn)).pack(side="right", padx=6, pady=12)

    # proceed
    if all(domain_progress[d]['status'] == 'done' for d in DOMAINS):
        tk.Button(frame, text="Proceed to Domain-level AHP", bg="#27AE60", fg="white", font=("Arial", 13, "bold"),
                  command=start_domain_level_ranking).pack(pady=16)

    tk.Button(window, text="Exit", font=("Arial", 12, "bold"), command=window.destroy).pack(side="bottom", pady=14)

# -----------------------
# Domain ranking & pairwise (per domain)
# -----------------------
def start_domain_ranking(domain):
    clear_screen()
    header_bar(title=f"AHP — {domain}", subtitle="Rank parameters 1 (most) → 4 (least)")
    frame = tk.Frame(window, bg="#F9F9F9")
    frame.pack(fill="both", expand=True, padx=18, pady=12)

    tk.Label(frame, text=f"AHP — {domain}", bg="#F9F9F9", font=SUBHEADER_FONT).pack(anchor="w", pady=8)
    tk.Label(frame, text="Rank parameters from 1 (most important) to 4 (least important):", bg="#F9F9F9", font=BODY_FONT).pack(anchor="w", pady=(0,8))

    rank_vars = {}
    form = tk.Frame(frame, bg="#F9F9F9")
    form.pack(anchor="w", padx=12, pady=6)
    for p in DOMAINS[domain]:
        row = tk.Frame(form, bg="#F9F9F9")
        row.pack(anchor="w", pady=6)
        tk.Label(row, text=p, bg="#F9F9F9", width=70, anchor="w", wraplength=700, font=BODY_FONT).pack(side="left")
        var = tk.StringVar(value="1")
        cb = ttk.Combobox(row, textvariable=var, values=["1","2","3","4"], width=4, state="readonly")
        cb.pack(side="left", padx=8)
        rank_vars[p] = var

    domain_progress[domain]['temp_rank_vars'] = rank_vars

    btn_frame = tk.Frame(frame, bg="#F9F9F9")
    btn_frame.pack(pady=14)
    tk.Button(btn_frame, text="Confirm Ranking", bg="#27AE60", fg="white", font=("Arial", 12, "bold"),
              command=lambda dn=domain: confirm_domain_ranking(dn)).pack(side="left", padx=6)
    tk.Button(btn_frame, text="Back", font=("Arial", 11), command=dashboard_screen).pack(side="left", padx=6)

def confirm_domain_ranking(domain):
    rank_vars = domain_progress[domain].get('temp_rank_vars')
    if not rank_vars:
        messagebox.showerror("Error", "No ranking data found.")
        start_domain_ranking(domain)
        return
    ranked = build_ranked_order(rank_vars)
    domain_progress[domain]['ranked_params'] = ranked
    # show confirmation and proceed
    clear_screen()
    header_bar(title=f"Confirm Ranking — {domain}")
    frame = tk.Frame(window, bg="#F9F9F9")
    frame.pack(fill="both", expand=True, padx=18, pady=12)
    tk.Label(frame, text="Confirm Ranking (highest → lowest)", bg="#F9F9F9", font=SUBHEADER_FONT).pack(anchor="w", pady=8)
    for i, p in enumerate(ranked, start=1):
        tk.Label(frame, text=f"{i}. {p}", bg="#F9F9F9", font=BODY_FONT).pack(anchor="w", padx=20)
    btn_frame = tk.Frame(frame, bg="#F9F9F9"); btn_frame.pack(pady=14)
    tk.Button(btn_frame, text="Proceed to Pairwise Comparison", bg="#27AE60", fg="white", font=("Arial", 12, "bold"),
              command=lambda dn=domain: start_pairwise_for_domain(dn)).pack(side="left", padx=6)
    tk.Button(btn_frame, text="Go Back", font=("Arial", 11), command=lambda dn=domain: start_domain_ranking(dn)).pack(side="left", padx=6)

def start_pairwise_for_domain(domain):
    dp = domain_progress[domain]
    ranked = dp['ranked_params']
    comps = build_comparisons_from_ranked(ranked)
    dp['comparisons'] = comps
    dp['pairwise'] = {}
    dp['index'] = 0
    open_pairwise_screen(domain)

def open_pairwise_screen(domain):
    dp = domain_progress[domain]
    comps = dp['comparisons']
    idx = dp['index']
    ranked = dp['ranked_params']
    clear_screen()
    header_bar(title=f"AHP — {domain}", subtitle="Pairwise comparisons (rank-based order)")
    if idx >= len(comps):
        finalize_domain_ahp(domain)
        return

    a, b = comps[idx]
    frame = tk.Frame(window, bg="#F9F9F9")
    frame.pack(fill="both", expand=True, padx=18, pady=12)

    tk.Label(frame, text=f"Compare the following pair (based on rank):", bg="#F9F9F9", font=BODY_FONT).pack(anchor="w", pady=(6,6))
    card = tk.Frame(frame, bg="#ECF0F1", bd=1, relief="solid", padx=16, pady=12)
    card.pack(pady=10, padx=8, fill="x")

    tk.Label(card, text=a, font=("Arial", 14, "bold"), fg="#2E86C1", bg="#ECF0F1").pack()
    tk.Label(card, text="vs", bg="#ECF0F1").pack()
    tk.Label(card, text=b, font=("Arial", 14, "bold"), fg="#E67E22", bg="#ECF0F1").pack(pady=(0,8))

    tk.Label(frame, text=f"{a}: {PARAMETER_DESCRIPTIONS.get(a,'')}", wraplength=980, justify="left", bg="#F9F9F9", font=BODY_FONT).pack(anchor="w", padx=6)
    tk.Label(frame, text=f"{b}: {PARAMETER_DESCRIPTIONS.get(b,'')}", wraplength=980, justify="left", bg="#F9F9F9", font=BODY_FONT).pack(anchor="w", padx=6, pady=(0,8))

    hover = tk.Label(frame, text="Hover over a number to read meaning", bg="#F9F9F9", font=ITALIC_FONT)
    hover.pack(pady=6, anchor="w", padx=6)

    btn_frame = tk.Frame(frame, bg="#F9F9F9")
    btn_frame.pack(pady=8, padx=6, anchor="w")
    # Create 1..9 buttons
    for col, val in enumerate(range(1, 10)):
        txt = scale_phrase(val, a, b, level="parameter")
        btn = tk.Button(btn_frame, text=str(val), width=4, height=2, font=("Arial", 11),
                        command=lambda v=val: confirm_pairwise(domain, a, b, v))
        btn.grid(row=0, column=col, padx=6, pady=6)
        btn.bind("<Enter>", lambda e, t=txt: hover.config(text=t))
        btn.bind("<Leave>", lambda e: hover.config(text="Hover over a number to read meaning"))

    # Hierarchy (vertical descending)
    tk.Label(frame, text="Order of importance (highest → lowest):", bg="#F9F9F9", font=("Arial", 11, "bold")).pack(pady=(12,4), anchor="w", padx=6)
    hframe = tk.Frame(frame, bg="#F9F9F9")
    hframe.pack(anchor="w", padx=20)
    # vertical list
    for j, param in enumerate(ranked, start=1):
        tk.Label(hframe, text=f"{j}. {param}", bg="#F9F9F9", anchor="w", justify="left", wraplength=900, font=BODY_FONT).pack(anchor="w")

    # Back button: if first pair, go back to confirm ranking; else previous pair
    back_frame = tk.Frame(frame, bg="#F9F9F9")
    back_frame.pack(pady=12)
    if idx == 0:
        tk.Button(back_frame, text="Go Back", font=("Arial", 11), command=lambda dn=domain: confirm_domain_ranking(dn)).pack()
    else:
        tk.Button(back_frame, text="Go Back", font=("Arial", 11), command=lambda dn=domain: go_back_pairwise(dn)).pack()

def confirm_pairwise(domain, a, b, val):
    desc = scale_phrase(val, a, b, level="parameter")
    if messagebox.askyesno("Confirm Choice", f"You chose:\n\n{desc}\n\nConfirm?"):
        domain_progress[domain]['pairwise'][(a, b)] = val
        domain_progress[domain]['index'] += 1
        open_pairwise_screen(domain)

def go_back_pairwise(domain):
    if domain_progress[domain]['index'] > 0:
        domain_progress[domain]['index'] -= 1
    open_pairwise_screen(domain)

def finalize_domain_ahp(domain):
    dp = domain_progress[domain]
    ranked = dp['ranked_params']
    A = build_matrix(ranked, dp['pairwise'])
    w, CR, _ = compute_eigen_weights_cr(A)
    dp['weights'] = w
    dp['CR'] = CR
    dp['matrix'] = A
    dp['status'] = 'done'
    show_domain_results(domain)

# gui2.py (PART 2 of 2)
# Multi-level AHP GUI — Water Security
# Part 2/2: domain results, domain-level flow, summary, save, start app
# (Append this to Part 1)

# -----------------------
# Domain results & utilities
# -----------------------
def show_domain_results(domain):
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

    # Display weights
    for i, (p, w) in enumerate(zip(dp['ranked_params'], dp['weights']), start=1):
        tk.Label(frame, text=f"{i}. {p}: {w:.4f}", bg="#F9F9F9", font=BODY_FONT).pack(anchor="w", padx=36)

    # Consistency Ratio display (spelled out)
    cr_color = "#1E8449" if dp['CR'] <= 0.1 else "#C0392B"
    tk.Label(frame, text=f"Consistency Ratio: {dp['CR']:.4f}", bg="#F9F9F9", font=("Arial", 12, "bold"), fg=cr_color).pack(pady=10, anchor="w", padx=36)

    # If inconsistent, show top inconsistent pairs for editing
    if dp['CR'] > 0.1:
        bad = top_inconsistencies(dp['matrix'], dp['ranked_params'], dp['weights'], k=4)
        tk.Label(frame, text="Most inconsistent pairs (click to edit):", bg="#F9F9F9", font=("Arial", 12, "bold")).pack(anchor="w", pady=(6,4))
        for (p, q), err, given, implied in bad:
            txt = f"{p} vs {q} — given: {given:.3g}, implied: {implied:.3g}, error: {err:.3f}"
            b = tk.Button(frame, text=txt, bg="#FAD7A0", wraplength=900, justify="left", font=SMALL_FONT,
                          command=lambda pair=(p,q), dn=domain: open_specific_pair_for_domain(dn, pair))
            b.pack(fill="x", padx=40, pady=4)

    btns = tk.Frame(frame, bg="#F9F9F9")
    btns.pack(pady=12)
    tk.Button(btns, text="Revisit Comparisons", bg="#F39C12", fg="white", font=("Arial", 11, "bold"),
              command=lambda dn=domain: reopen_domain_pairwise(dn)).pack(side="left", padx=8)
    tk.Button(btns, text="Back to Domains", font=("Arial", 11), command=dashboard_screen).pack(side="left", padx=8)

def open_specific_pair_for_domain(domain, pair):
    comps = domain_progress[domain]['comparisons']
    try:
        idx = comps.index(pair)
    except ValueError:
        # try reverse ordering
        try:
            idx = comps.index((pair[1], pair[0]))
        except ValueError:
            messagebox.showerror("Not found", "Comparison not found.")
            return
    domain_progress[domain]['index'] = idx
    open_pairwise_screen(domain)

def reopen_domain_pairwise(domain):
    domain_progress[domain]['index'] = 0
    open_pairwise_screen(domain)

# -----------------------
# Domain-level (3 domains) AHP
# -----------------------
def start_domain_level_ranking():
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
        cb = ttk.Combobox(row, textvariable=var, values=["1","2","3"], width=4, state="readonly")
        cb.pack(side="left", padx=8)
        rank_vars[d] = var

    domain_level_result['temp_rank_vars'] = rank_vars
    btn_frame = tk.Frame(frame, bg="#F9F9F9")
    btn_frame.pack(pady=12)
    tk.Button(btn_frame, text="Confirm Ranking", bg="#27AE60", fg="white", font=("Arial", 12, "bold"),
              command=confirm_domain_level_ranking).pack(side="left", padx=6)
    tk.Button(btn_frame, text="Back", font=("Arial", 11), command=dashboard_screen).pack(side="left", padx=6)

def confirm_domain_level_ranking():
    rank_vars = domain_level_result.get('temp_rank_vars')
    if not rank_vars:
        messagebox.showerror("Error", "No ranking data.")
        start_domain_level_ranking()
        return
    ranked = build_ranked_order(rank_vars)
    domain_level_result['ranked_domains'] = ranked
    # confirm screen
    clear_screen()
    header_bar(title="Confirm Domain Ranking")
    frame = tk.Frame(window, bg="#F9F9F9")
    frame.pack(fill="both", expand=True, padx=18, pady=12)
    tk.Label(frame, text="Confirm Domain Ranking", bg="#F9F9F9", font=SUBHEADER_FONT).pack(anchor="w", pady=8)
    for i, d in enumerate(ranked, start=1):
        tk.Label(frame, text=f"{i}. {d}", bg="#F9F9F9", font=BODY_FONT).pack(anchor="w", padx=20)
    btn_frame = tk.Frame(frame, bg="#F9F9F9"); btn_frame.pack(pady=12)
    tk.Button(btn_frame, text="Proceed to Pairwise", bg="#27AE60", fg="white", font=("Arial", 12, "bold"),
              command=start_domain_level_pairwise).pack(side="left", padx=6)
    tk.Button(btn_frame, text="Go Back", font=("Arial", 11), command=start_domain_level_ranking).pack(side="left", padx=6)

def start_domain_level_pairwise():
    ranked = domain_level_result['ranked_domains']
    comps = build_comparisons_from_ranked(ranked)
    domain_level_result['comparisons'] = comps
    domain_level_result['pairwise'] = {}
    domain_level_result['index'] = 0
    open_domain_level_pairwise()

def open_domain_level_pairwise():
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
    tk.Label(frame, text=f"Compare the domains: {a} vs {b}", bg="#F9F9F9", font=SUBHEADER_FONT).pack(anchor="w", pady=(6,8))

    hover = tk.Label(frame, text="Hover over a number to read meaning", bg="#F9F9F9", font=ITALIC_FONT)
    hover.pack(anchor="w", padx=8, pady=(0,8))

    btn_frame = tk.Frame(frame, bg="#F9F9F9")
    btn_frame.pack(anchor="w", padx=8, pady=6)
    for col, val in enumerate(range(1, 10)):
        txt = scale_phrase(val, a, b, level="domain")
        btn = tk.Button(btn_frame, text=str(val), width=4, height=2, font=("Arial", 11),
                        command=lambda v=val: confirm_domain_level_choice(a, b, v))
        btn.grid(row=0, column=col, padx=6, pady=6)
        btn.bind("<Enter>", lambda e, t=txt: hover.config(text=t))
        btn.bind("<Leave>", lambda e: hover.config(text="Hover over a number to read meaning"))

    # hierarchy vertical
    tk.Label(frame, text="Order of importance (highest → lowest):", bg="#F9F9F9", font=("Arial", 11, "bold")).pack(pady=(10,4), anchor="w")
    for i, d in enumerate(ranked, start=1):
        tk.Label(frame, text=f"{i}. {d}", bg="#F9F9F9", font=BODY_FONT).pack(anchor="w", padx=20)

    # back button
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
    ranked = domain_level_result['ranked_domains']
    A = build_matrix(ranked, domain_level_result['pairwise'])
    w, CR, _ = compute_eigen_weights_cr(A)
    domain_level_result['weights'] = w
    domain_level_result['CR'] = CR
    domain_level_result['matrix'] = A
    show_domain_level_results()

def show_domain_level_results():
    clear_screen()
    header_bar(title="AHP Results — Water Security Domains")
    frame = tk.Frame(window, bg="#F9F9F9")
    frame.pack(fill="both", expand=True, padx=18, pady=12)
    tk.Label(frame, text="AHP Results — Water Security Domains", bg="#F9F9F9", font=SUBHEADER_FONT).pack(anchor="w", pady=8)
    if domain_level_result['weights'] is None:
        tk.Label(frame, text="No domain-level result yet.", bg="#F9F9F9", font=BODY_FONT).pack(anchor="w", pady=6)
        tk.Button(frame, text="Back to Dashboard", font=("Arial", 11), command=dashboard_screen).pack(pady=8)
        return
    for d, w in zip(domain_level_result['ranked_domains'], domain_level_result['weights']):
        tk.Label(frame, text=f"{d}: {w:.4f}", bg="#F9F9F9", font=BODY_FONT).pack(anchor="w", padx=36)
    cr_color = "#1E8449" if domain_level_result['CR'] <= 0.1 else "#C0392B"
    tk.Label(frame, text=f"Consistency Ratio: {domain_level_result['CR']:.4f}", bg="#F9F9F9",
          font=("Arial", 12, "bold"), fg=cr_color).pack(pady=10, anchor="w", padx=36)

    btns = tk.Frame(frame, bg="#F9F9F9")
    btns.pack(pady=12)
    tk.Button(btns, text="AHP Summary", bg="#3498DB", fg="white", font=("Arial", 12, "bold"), command=ahp_summary_screen).pack(side="left", padx=8)
    tk.Button(btns, text="Back to Dashboard", font=("Arial", 11), command=dashboard_screen).pack(side="left", padx=8)

# -----------------------
# AHP Summary + Save (CSV/TXT)
# -----------------------
def ahp_summary_screen():
    clear_screen()
    header_bar(title="AHP Summary — All Results")
    frame = tk.Frame(window, bg="#F9F9F9")
    frame.pack(fill="both", expand=True, padx=18, pady=12)

    tk.Label(frame, text="AHP Summary — All Results", bg="#F9F9F9", font=SUBHEADER_FONT).pack(anchor="w", pady=8)

    # Level 1 domain results
    for d in DOMAINS:
        dp = domain_progress[d]
        tk.Label(frame, text=f"{d} — {dp['status'].upper()}", bg="#F9F9F9", font=("Arial", 12, "bold")).pack(anchor="w", padx=20, pady=(6,2))
        if dp['status'] == 'done' and dp['weights'] is not None:
            for p, w in zip(dp['ranked_params'], dp['weights']):
                tk.Label(frame, text=f"    {p}: {w:.4f}", bg="#F9F9F9", font=BODY_FONT).pack(anchor="w", padx=48)
            tk.Label(frame, text=f"    Consistency Ratio: {dp['CR']:.4f}", bg="#F9F9F9", font=BODY_FONT).pack(anchor="w", padx=48, pady=(0,6))
        else:
            tk.Label(frame, text="    (not completed)", bg="#F9F9F9", font=BODY_FONT).pack(anchor="w", padx=48, pady=(0,6))

    # Level 2 domain-level
    tk.Label(frame, text="", bg="#F9F9F9").pack()
    tk.Label(frame, text="Water Security Domains", bg="#F9F9F9", font=("Arial", 13, "bold")).pack(anchor="w", pady=(8,4))
    if domain_level_result['weights'] is not None:
        for d, w in zip(domain_level_result['ranked_domains'], domain_level_result['weights']):
            tk.Label(frame, text=f"    {d}: {w:.4f}", bg="#F9F9F9", font=BODY_FONT).pack(anchor="w", padx=48)
        tk.Label(frame, text=f"    Consistency Ratio: {domain_level_result['CR']:.4f}", bg="#F9F9F9", font=BODY_FONT).pack(anchor="w", padx=48, pady=(0,6))
    else:
        tk.Label(frame, text="    (domain-level AHP not yet completed)", bg="#F9F9F9", font=BODY_FONT).pack(anchor="w", padx=48, pady=(0,6))

    btns = tk.Frame(frame, bg="#F9F9F9")
    btns.pack(pady=12)
    tk.Button(btns, text="Save Results", bg="#27AE60", fg="white", font=("Arial", 12, "bold"), command=save_results_dialog).pack(side="left", padx=8)
    tk.Button(btns, text="Back to Dashboard", font=("Arial", 11), command=dashboard_screen).pack(side="left", padx=8)

def save_results_dialog():
    opts = [("CSV file", "*.csv"), ("Text file", "*.txt")]
    fname = filedialog.asksaveasfilename(title="Save AHP Summary", defaultextension=".csv", filetypes=opts)
    if not fname:
        return
    if fname.lower().endswith(".csv"):
        save_as_csv(fname)
    else:
        save_as_txt(fname)

def save_as_csv(path):
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
            # Domain-level
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

# -----------------------
# Start app
# -----------------------
dashboard_screen()
window.mainloop()
