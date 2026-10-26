"""
Schraubradgetriebe-Rechner (Kunststoff) – Streamlit-Anwendung
==============================================================
Vereinfachte Auslegung und Nachweisrechnung von Kunststoff-Schraubradgetrieben
(gekreuzte Achsen) in Anlehnung an VDI 2736 / DIN 3990.

WICHTIG: Die Festigkeitsberechnung ist ein vereinfachter Ansatz für Lehre
und Vorauslegung. Werkstoffkennwerte sind Richtwerte und müssen für die
Bachelorarbeit mit Datenblättern / VDI 2736 Blatt 1-4 abgeglichen werden.

Start:  streamlit run app.py
"""

import math
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# 1. WERKSTOFFDATENBANK (Richtwerte, bei ca. 20 °C, trocken)
# ---------------------------------------------------------------------------
WERKSTOFFE = {
    "POM (Polyoxymethylen)": dict(
        E=2900.0,          # E-Modul [MPa]
        sigma_Flim=36.0,   # Zahnfuß-Dauerfestigkeit (Biegung) [MPa]
        sigma_Hlim=19.0,   # Flankenfestigkeit (Pressung) [MPa]
        T_max=90.0,        # max. Dauereinsatztemperatur [°C]
        nu=0.35,           # Querkontraktionszahl [-]
        mu=0.20,           # Gleitreibwert Kunststoff/Kunststoff [-]
    ),
    "PA66-GF30": dict(
        E=7500.0, sigma_Flim=50.0, sigma_Hlim=30.0,
        T_max=120.0, nu=0.38, mu=0.25,
    ),
    "PEEK": dict(
        E=3700.0, sigma_Flim=65.0, sigma_Hlim=40.0,
        T_max=250.0, nu=0.40, mu=0.20,
    ),
    "PBT": dict(
        E=2600.0, sigma_Flim=30.0, sigma_Hlim=20.0,
        T_max=120.0, nu=0.38, mu=0.22,
    ),
}


# ---------------------------------------------------------------------------
# 2. GEOMETRIE
# ---------------------------------------------------------------------------
def berechne_geometrie(z1, i_soll, m_n, beta1, alpha_n, sigma):
    """Geometrie eines Schraubradpaares (alle Winkel in Grad, Längen in mm).

    z1      : Zähnezahl Rad 1 [-]
    i_soll  : Ziel-Übersetzung [-]
    m_n     : Normalmodul [mm]
    beta1   : Schrägungswinkel Rad 1 [°]
    alpha_n : Normaleingriffswinkel [°]
    sigma   : Achskreuzungswinkel [°]  (beta1 + beta2 = sigma, gleichsinnig)
    """
    z2 = max(1, round(i_soll * z1))          # Zähnezahl Rad 2 [-]
    beta2 = sigma - beta1                    # Schrägungswinkel Rad 2 [°]
    b1, b2, an = map(math.radians, (beta1, beta2, alpha_n))

    # Stirneingriffswinkel je Rad: tan(alpha_t) = tan(alpha_n) / cos(beta)
    at1 = math.atan(math.tan(an) / math.cos(b1))
    at2 = math.atan(math.tan(an) / math.cos(b2))

    # Teilkreisdurchmesser d = m_n * z / cos(beta)  [mm]
    d1 = m_n * z1 / math.cos(b1)
    d2 = m_n * z2 / math.cos(b2)

    # Grundkreisdurchmesser db = d * cos(alpha_t)  [mm]
    db1 = d1 * math.cos(at1)
    db2 = d2 * math.cos(at2)

    # Grundschrägungswinkel: sin(beta_b) = sin(beta) * cos(alpha_n)  [°]
    bb1 = math.asin(math.sin(b1) * math.cos(an))

    return dict(
        z1=z1, z2=z2, beta1=beta1, beta2=beta2,
        alpha_t1=math.degrees(at1), alpha_t2=math.degrees(at2),
        d1=d1, d2=d2, db1=db1, db2=db2,
        a=(d1 + d2) / 2.0,                   # Achsabstand [mm]
        i_ist=z2 / z1,                       # Zähnezahlverhältnis [-]
        beta_b1=math.degrees(bb1),
        zn1=z1 / math.cos(b1) ** 3,          # Ersatzzähnezahl Rad 1 [-]
        zn2=z2 / math.cos(b2) ** 3,          # Ersatzzähnezahl Rad 2 [-]
    )


# ---------------------------------------------------------------------------
# 3. KRÄFTE, GESCHWINDIGKEITEN, WIRKUNGSGRAD
# ---------------------------------------------------------------------------
def berechne_kraefte(p, geo):
    """Zahnkräfte, Geschwindigkeiten und Wirkungsgrad (reibungsfrei für Kräfte)."""
    an = math.radians(p["alpha_n"])
    b1 = math.radians(geo["beta1"])
    b2 = math.radians(geo["beta2"])

    Ft1 = 2000.0 * p["T1"] / geo["d1"]                 # Umfangskraft Rad 1 [N]
    Fn = Ft1 / (math.cos(an) * math.cos(b1))           # Zahnnormalkraft [N]
    Ft2 = Fn * math.cos(an) * math.cos(b2)             # Umfangskraft Rad 2 [N]
    Fa1 = Fn * math.cos(an) * math.sin(b1)             # Axialkraft Rad 1 [N]
    Fa2 = Fn * math.cos(an) * math.sin(b2)             # Axialkraft Rad 2 [N]
    Fr = Fn * math.sin(an)                             # Radialkraft [N]

    n2 = p["n1"] / geo["i_ist"]                        # Abtriebsdrehzahl [1/min]
    v1 = math.pi * geo["d1"] * p["n1"] / 60000.0       # Umfangsgeschw. Rad 1 [m/s]
    v2 = math.pi * geo["d2"] * n2 / 60000.0            # Umfangsgeschw. Rad 2 [m/s]
    sig = math.radians(p["sigma"])
    v_g = math.sqrt(v1**2 + v2**2 - 2 * v1 * v2 * math.cos(sig))  # Gleitgeschw. [m/s]

    # Wirkungsgrad (Schraubrad, Rad 1 treibend): (cos an - mu tan b1)/(cos an + mu tan b2)
    mu = p["mat"]["mu"]
    eta = (math.cos(an) - mu * math.tan(b1)) / (math.cos(an) + mu * math.tan(b2))
    eta = min(max(eta, 0.05), 0.99)

    P1 = p["T1"] * p["n1"] * math.pi / 30.0 / 1000.0   # Antriebsleistung [kW]
    return dict(
        Ft1=Ft1, Ft2=Ft2, Fn=Fn, Fa1=Fa1, Fa2=Fa2, Fr=Fr,
        n2=n2, v1=v1, v2=v2, v_g=v_g, eta=eta,
        P1=P1, P2=P1 * eta, P_V=P1 * (1 - eta),        # [kW]
        T2=Ft2 * geo["d2"] / 2000.0,                   # Abtriebsmoment [Nm]
    )


# ---------------------------------------------------------------------------
# 4. FESTIGKEITSNACHWEIS (vereinfacht, angelehnt an VDI 2736 / DIN 3990)
# ---------------------------------------------------------------------------
def faktor_temperatur(T_umg, T_max):
    """Temperaturfaktor Y_T [-]: lineare Abminderung von 20 °C bis T_max."""
    anteil = (T_umg - 20.0) / max(T_max - 20.0, 1.0)
    return float(np.clip(1.0 - 0.5 * anteil, 0.2, 1.0))


def faktor_lebensdauer(N_L):
    """Lebensdauerfaktor Y_N [-] (Bezug: 1e7 Lastwechsel, Exponent 0,1)."""
    return float(np.clip((1e7 / max(N_L, 1.0)) ** 0.1, 0.6, 1.8))


def formfaktor_fuss(zn):
    """Näherung Formfaktor Y_Fa (alpha_n = 20°, x = 0) über Ersatzzähnezahl [-]."""
    return 2.1 + 14.5 / zn


def spannungskorrekturfaktor(zn):
    """Näherung Spannungskorrekturfaktor Y_Sa über Ersatzzähnezahl [-]."""
    return 1.07 + 0.158 * math.log(zn)


def berechne_festigkeit(p, geo, kr):
    """Zahnfuß- und Flankennachweis für beide Räder."""
    mat = p["mat"]
    Y_T = faktor_temperatur(p["T_umg"], mat["T_max"])
    K_V = 1.0 + 0.05 * kr["v1"]                         # Dynamikfaktor [-] (Näherung)
    K = p["K_A"] * K_V                                  # Gesamt-Lastfaktor ohne Breitenfaktor

    N1 = 60.0 * p["n1"] * p["L"]                        # Lastwechsel Rad 1 [-]
    N2 = 60.0 * kr["n2"] * p["L"]                       # Lastwechsel Rad 2 [-]

    ergebnis = {"Y_T": Y_T, "K_V": K_V}
    for k, (z, zn, Ft, N) in enumerate(
        [(geo["z1"], geo["zn1"], kr["Ft1"], N1), (geo["z2"], geo["zn2"], kr["Ft2"], N2)], start=1
    ):
        Y_N = faktor_lebensdauer(N)
        Y_Fa, Y_Sa = formfaktor_fuss(zn), spannungskorrekturfaktor(zn)
        # Zahnfußspannung [MPa]: Ft/(b*m_n) * Y_Fa * Y_Sa * K_A * K_V * K_Fbeta
        sigma_F = Ft / (p["b"] * p["m_n"]) * Y_Fa * Y_Sa * K * p["K_Fb"]
        sigma_F_zul = mat["sigma_Flim"] * Y_T * Y_N     # zulässig [MPa]
        ergebnis.update({
            f"N{k}": N, f"Y_N{k}": Y_N, f"Y_Fa{k}": Y_Fa, f"Y_Sa{k}": Y_Sa,
            f"sigma_F{k}": sigma_F, f"sigma_F_zul{k}": sigma_F_zul,
            f"S_F{k}": sigma_F_zul / sigma_F,
        })

    # Flankenpressung (Hertz, Ersatz-Stirnradpaar, mit Punktberührungsfaktor Z_P)
    an = math.radians(p["alpha_n"])
    at = math.radians(geo["alpha_t1"])
    bb = math.radians(geo["beta_b1"])
    Z_H = math.sqrt(2.0 * math.cos(bb) / (math.cos(at) * math.sin(at)))   # Zonenfaktor [-]
    Z_E = math.sqrt(1.0 / (math.pi * 2.0 * (1.0 - mat["nu"] ** 2) / mat["E"]))  # [sqrt(MPa)]
    u = geo["z2"] / geo["z1"]
    sigma_H = (Z_H * Z_E * math.sqrt(kr["Ft1"] * K * p["K_Hb"] * (u + 1.0)
               / (p["b"] * geo["d1"] * u)) * p["Z_P"])           # [MPa]
    Y_N_H = faktor_lebensdauer(min(N1, N2))
    sigma_H_zul = mat["sigma_Hlim"] * Y_T * Y_N_H                # [MPa]
    ergebnis.update(Z_H=Z_H, Z_E=Z_E, sigma_H=sigma_H,
                    sigma_H_zul=sigma_H_zul, S_H=sigma_H_zul / sigma_H)
    ergebnis["S_F"] = min(ergebnis["S_F1"], ergebnis["S_F2"])
    return ergebnis


def rechne_alles(p):
    """Komplette Rechenkette (auch für Parameterstudien nutzbar)."""
    geo = berechne_geometrie(p["z1"], p["i"], p["m_n"], p["beta1"], p["alpha_n"], p["sigma"])
    kr = berechne_kraefte(p, geo)
    fe = berechne_festigkeit(p, geo, kr)
    return geo, kr, fe


# ---------------------------------------------------------------------------
# 5. EMPFEHLUNGEN
# ---------------------------------------------------------------------------
def erzeuge_empfehlungen(p, geo, kr, fe):
    """Liste konkreter Hinweise (Typ, Text)."""
    h = []
    if fe["S_F"] < p["SF_min"]:
        h.append(("error", "Zahnfuß unterdimensioniert: **Modul m_n vergrößern**, "
                  "**Zahnbreite b erhöhen** oder **Werkstoff mit höherem σ_Flim** wählen (z. B. PA66-GF30 / PEEK)."))
    if fe["S_H"] < p["SH_min"]:
        h.append(("error", "Flanke überlastet: **Zahnbreite b erhöhen**, **Durchmesser/Zähnezahl z1 vergrößern**, "
                  "Werkstoff mit höherem σ_Hlim wählen oder Drehmoment reduzieren."))
    if p["T_umg"] > p["mat"]["T_max"]:
        h.append(("error", f"Umgebungstemperatur über T_max = {p['mat']['T_max']:.0f} °C: **Temperaturbeständigeren Werkstoff** wählen."))
    elif p["T_umg"] > 0.8 * p["mat"]["T_max"]:
        h.append(("warning", "Betriebstemperatur nahe der Einsatzgrenze – Festigkeit stark reduziert."))
    if kr["v_g"] > 5.0:
        h.append(("warning", f"Gleitgeschwindigkeit {kr['v_g']:.1f} m/s hoch: Verschleiß/Erwärmung prüfen, ggf. schmieren."))
    if min(geo["zn1"], geo["zn2"]) < 17:
        h.append(("warning", "Ersatzzähnezahl < 17: Gefahr von Unterschnitt, Profilverschiebung prüfen."))
    if abs(geo["i_ist"] - p["i"]) / p["i"] > 0.02:
        h.append(("info", f"Ist-Übersetzung {geo['i_ist']:.3f} weicht um mehr als 2 % von i_soll = {p['i']:.3f} ab – z1 anpassen."))
    return h


# ---------------------------------------------------------------------------
# 6. DIAGRAMME
# ---------------------------------------------------------------------------
def kreis(cx, r, n=200):
    """Kreispunkte (x, y) um (cx, 0) mit Radius r [mm]."""
    t = np.linspace(0, 2 * np.pi, n)
    return cx + r * np.cos(t), r * np.sin(t)


def diagramm_geometrie(geo):
    """Prinzipdarstellung: Teil- und Grundkreise, Achsabstand (schematisch)."""
    fig = go.Figure()
    a = geo["a"]
    r1, r2 = geo["d1"] / 2, geo["d2"] / 2
    for cx, r, rb, name, farbe in [(0, r1, geo["db1"] / 2, "Rad 1", "#1f77b4"),
                                    (a, r2, geo["db2"] / 2, "Rad 2", "#d62728")]:
        x, y = kreis(cx, r)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"Teilkreis {name}", line=dict(color=farbe, width=3)))
        x, y = kreis(cx, rb)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"Grundkreis {name}", line=dict(color=farbe, dash="dot")))
    fig.add_trace(go.Scatter(x=[0, a], y=[0, 0], mode="lines+markers+text", name="Achsabstand a",
                             line=dict(color="black", dash="dash"), text=["", f"a = {a:.2f} mm"],
                             textposition="top left"))
    fig.update_layout(title="Prinzipdarstellung der Teil- und Grundkreise (schematische Draufsicht)",
                      xaxis_title="x [mm]", yaxis_title="y [mm]", height=520,
                      yaxis=dict(scaleanchor="x", scaleratio=1))
    return fig


def parameterstudie(p, groesse):
    """Sicherheiten S_F und S_H in Abhängigkeit von Drehzahl, Temperatur oder Moment."""
    konfig = {
        "Drehzahl n1 [1/min]": ("n1", np.linspace(max(p["n1"] * 0.1, 10), p["n1"] * 3, 40)),
        "Umgebungstemperatur T_umg [°C]": ("T_umg", np.linspace(-20, p["mat"]["T_max"] + 20, 40)),
        "Drehmoment T1 [Nm]": ("T1", np.linspace(p["T1"] * 0.2, p["T1"] * 3, 40)),
    }
    key, xs = konfig[groesse]
    sf, sh = [], []
    for x in xs:
        q = dict(p)
        q[key] = float(x)
        _, _, fe = rechne_alles(q)
        sf.append(fe["S_F"])
        sh.append(fe["S_H"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=sf, name="S_F (Zahnfuß)", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=xs, y=sh, name="S_H (Flanke)", line=dict(width=3)))
    fig.add_hline(y=p["SF_min"], line_dash="dash", line_color="#1f77b4", annotation_text="S_F,min")
    fig.add_hline(y=p["SH_min"], line_dash="dash", line_color="#ff7f0e", annotation_text="S_H,min")
    fig.add_vline(x=p[key], line_dash="dot", line_color="gray", annotation_text="Arbeitspunkt")
    fig.update_layout(title=f"Sicherheiten über {groesse}", xaxis_title=groesse,
                      yaxis_title="Sicherheit [-]", height=480)
    return fig


# ---------------------------------------------------------------------------
# 7. EXPORT (CSV / HTML-Protokoll)
# ---------------------------------------------------------------------------
def tabelle_export(p, geo, kr, fe, werkstoff_name):
    """Alle Ein- und Ausgaben als DataFrame (Kategorie, Größe, Wert, Einheit)."""
    z = []
    def add(kat, name, wert, einheit):
        z.append((kat, name, round(wert, 4) if isinstance(wert, float) else wert, einheit))
    add("Eingabe", "Werkstoff", werkstoff_name, "-")
    for name, key, einh in [("Ziel-Übersetzung i", "i", "-"), ("Normalmodul m_n", "m_n", "mm"),
                            ("Schrägungswinkel beta1", "beta1", "°"), ("Eingriffswinkel alpha_n", "alpha_n", "°"),
                            ("Achskreuzungswinkel Sigma", "sigma", "°"), ("Zahnbreite b", "b", "mm"),
                            ("Zähnezahl z1", "z1", "-"), ("Drehzahl n1", "n1", "1/min"),
                            ("Drehmoment T1", "T1", "Nm"), ("Lebensdauer L", "L", "h"),
                            ("Umgebungstemperatur T_umg", "T_umg", "°C"),
                            ("Mindestsicherheit S_F,min", "SF_min", "-"), ("Mindestsicherheit S_H,min", "SH_min", "-")]:
        add("Eingabe", name, p[key], einh)
    for name, key, einh in [("Zähnezahl z2", "z2", "-"), ("Achsabstand a", "a", "mm"),
                            ("Teilkreisdurchmesser d1", "d1", "mm"), ("Teilkreisdurchmesser d2", "d2", "mm"),
                            ("Grundkreisdurchmesser db1", "db1", "mm"), ("Grundkreisdurchmesser db2", "db2", "mm"),
                            ("Schrägungswinkel beta2", "beta2", "°"), ("Ist-Übersetzung i_ist", "i_ist", "-")]:
        add("Geometrie", name, geo[key], einh)
    for name, key, einh in [("Umfangskraft Ft1", "Ft1", "N"), ("Umfangskraft Ft2", "Ft2", "N"),
                            ("Axialkraft Fa1", "Fa1", "N"), ("Radialkraft Fr", "Fr", "N"),
                            ("Gleitgeschwindigkeit v_g", "v_g", "m/s"), ("Wirkungsgrad eta", "eta", "-"),
                            ("Verlustleistung P_V", "P_V", "kW"), ("Abtriebsmoment T2", "T2", "Nm")]:
        add("Kräfte", name, kr[key], einh)
    for name, key, einh in [("Zahnfußspannung sigma_F1", "sigma_F1", "MPa"), ("Zahnfußspannung sigma_F2", "sigma_F2", "MPa"),
                            ("Zahnfußsicherheit S_F1", "S_F1", "-"), ("Zahnfußsicherheit S_F2", "S_F2", "-"),
                            ("Flankenpressung sigma_H", "sigma_H", "MPa"), ("Flankensicherheit S_H", "S_H", "-"),
                            ("Temperaturfaktor Y_T", "Y_T", "-")]:
        add("Festigkeit", name, fe[key], einh)
    return pd.DataFrame(z, columns=["Kategorie", "Größe", "Wert", "Einheit"])


def html_protokoll(df, status_text):
    """Formatiertes HTML-Protokoll (im Browser über Drucken als PDF speicherbar)."""
    zeit = datetime.now().strftime("%d.%m.%Y %H:%M")
    return f"""<!DOCTYPE html><html lang="de"><head><meta charset="utf-8">
<title>Berechnungsprotokoll Schraubradgetriebe</title>
<style>body{{font-family:Arial,sans-serif;margin:30px}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #bbb;padding:6px 10px;text-align:left}}th{{background:#eee}}</style></head><body>
<h1>Berechnungsprotokoll Kunststoff-Schraubradgetriebe</h1>
<p>Erstellt am {zeit}</p><h2>Ergebnis: {status_text}</h2>
{df.to_html(index=False)}
<p><em>Vereinfachte Berechnung in Anlehnung an VDI 2736 / DIN 3990 – Vorauslegung, kein Ersatz für einen vollständigen Nachweis.</em></p>
</body></html>"""


# ---------------------------------------------------------------------------
# 8. STREAMLIT-OBERFLÄCHE
# ---------------------------------------------------------------------------
def sidebar_eingaben():
    """Liest alle Eingaben aus der Sidebar und gibt Parameter-Dict + Werkstoffname zurück."""
    sb = st.sidebar
    sb.header("⚙️ Eingaben")

    sb.subheader("Werkstoff")
    name = sb.selectbox("Kunststoff", list(WERKSTOFFE) + ["Benutzerdefiniert"])
    if name == "Benutzerdefiniert":
        mat = dict(
            E=sb.number_input("E-Modul [MPa]", 500.0, 30000.0, 3000.0, 100.0),
            sigma_Flim=sb.number_input("σ_Flim [MPa]", 5.0, 300.0, 40.0, 1.0),
            sigma_Hlim=sb.number_input("σ_Hlim [MPa]", 5.0, 300.0, 20.0, 1.0),
            T_max=sb.number_input("T_max [°C]", 40.0, 400.0, 100.0, 5.0),
            nu=sb.number_input("Querkontraktion ν [-]", 0.2, 0.5, 0.35, 0.01),
            mu=sb.number_input("Reibwert μ [-]", 0.05, 0.5, 0.2, 0.01),
        )
    else:
        mat = WERKSTOFFE[name]

    sb.subheader("Übersetzung & Geometrie")
    i = sb.number_input("Ziel-Übersetzung i [-]", 0.2, 20.0, 2.0, 0.1)
    m_n = sb.number_input("Normalmodul m_n [mm]", 0.3, 10.0, 2.0, 0.1)
    sigma = sb.slider("Achskreuzungswinkel Σ [°]", 30.0, 90.0, 90.0, 1.0)
    beta1 = sb.slider("Schrägungswinkel β1 [°] (β2 = Σ − β1)", 5.0, sigma - 5.0, min(45.0, sigma - 5.0), 0.5)
    alpha_n = sb.number_input("Eingriffswinkel α_n [°]", 14.5, 25.0, 20.0, 0.5)
    z1 = int(sb.number_input("Zähnezahl z1 [-]", 6, 100, 20, 1))
    b = sb.number_input("Zahnbreite b [mm]", 2.0, 100.0, 20.0, 0.5)

    sb.subheader("Betriebsparameter")
    n1 = sb.number_input("Antriebsdrehzahl n1 [1/min]", 1.0, 20000.0, 1500.0, 50.0)
    modus = sb.radio("Lastvorgabe", ["Drehmoment T1 [Nm]", "Leistung P1 [kW]"], horizontal=True)
    if modus.startswith("Dreh"):
        T1 = sb.number_input("Drehmoment T1 [Nm]", 0.01, 500.0, 0.3, 0.05)
    else:
        P1 = sb.number_input("Leistung P1 [kW]", 0.001, 50.0, 0.3, 0.05)
        T1 = 9549.3 * P1 / n1                       # T [Nm] = 9549,3 * P [kW] / n [1/min]
        sb.caption(f"→ T1 = {T1:.2f} Nm")
    L = sb.number_input("Geforderte Lebensdauer L [h]", 10.0, 100000.0, 5000.0, 100.0)
    T_umg = sb.number_input("Betriebstemperatur T_umg [°C]", -40.0, 250.0, 40.0, 5.0)

    sb.subheader("Ziel-Sicherheiten")
    SF_min = sb.number_input("S_F,min Zahnfuß [-]", 1.0, 5.0, 1.5, 0.1)
    SH_min = sb.number_input("S_H,min Zahnflanke [-]", 1.0, 5.0, 1.2, 0.1)

    with sb.expander("Erweiterte Faktoren"):
        K_A = st.number_input("Anwendungsfaktor K_A [-]", 1.0, 3.0, 1.25, 0.05)
        K_Fb = st.number_input("Breitenfaktor Fuß K_Fβ [-]", 1.0, 2.5, 1.3, 0.05)
        K_Hb = st.number_input("Breitenfaktor Flanke K_Hβ [-]", 1.0, 2.5, 1.3, 0.05)
        Z_P = st.number_input("Punktberührungsfaktor Z_P [-]", 1.0, 3.0, 1.5, 0.1,
                              help="Schraubräder berühren sich punktförmig; Z_P erhöht die Pressung gegenüber Linienberührung.")

    p = dict(mat=mat, i=i, m_n=m_n, sigma=sigma, beta1=beta1, alpha_n=alpha_n, z1=z1, b=b,
             n1=n1, T1=T1, L=L, T_umg=T_umg, SF_min=SF_min, SH_min=SH_min,
             K_A=K_A, K_Fb=K_Fb, K_Hb=K_Hb, Z_P=Z_P)
    return p, name


def main():
    st.set_page_config(page_title="Schraubradgetriebe-Rechner", page_icon="⚙️", layout="wide")
    st.title("⚙️ Auslegung von Kunststoff-Schraubradgetrieben")
    st.caption("Vereinfachte Berechnung in Anlehnung an VDI 2736 / DIN 3990 – Bachelorarbeit Maschinenbau")

    p, werkstoff_name = sidebar_eingaben()
    geo, kr, fe = rechne_alles(p)

    ok = fe["S_F"] >= p["SF_min"] and fe["S_H"] >= p["SH_min"]
    status = "AUSLEGUNG IN ORDNUNG" if ok else "AUSLEGUNG NICHT AUSREICHEND"
    if ok:
        st.success(f"✅ {status}: S_F = {fe['S_F']:.2f} ≥ {p['SF_min']}, S_H = {fe['S_H']:.2f} ≥ {p['SH_min']}")
    else:
        st.error(f"❌ {status}: S_F = {fe['S_F']:.2f} (min. {p['SF_min']}), S_H = {fe['S_H']:.2f} (min. {p['SH_min']})")
    for typ, text in erzeuge_empfehlungen(p, geo, kr, fe):
        {"error": st.error, "warning": st.warning, "info": st.info}[typ](text)

    tab_geo, tab_fest, tab_dia, tab_exp = st.tabs(["📐 Geometrie", "🔩 Festigkeit", "📈 Diagramme", "💾 Export"])

    with tab_geo:
        c = st.columns(4)
        c[0].metric("Zähnezahl z2", f"{geo['z2']}")
        c[1].metric("Achsabstand a", f"{geo['a']:.2f} mm")
        c[2].metric("Ist-Übersetzung i_ist", f"{geo['i_ist']:.3f}", f"{geo['i_ist'] - p['i']:+.3f} ggü. Soll")
        c[3].metric("Schrägungswinkel β2", f"{geo['beta2']:.1f} °")
        c = st.columns(4)
        c[0].metric("Teilkreis d1", f"{geo['d1']:.2f} mm")
        c[1].metric("Teilkreis d2", f"{geo['d2']:.2f} mm")
        c[2].metric("Grundkreis db1", f"{geo['db1']:.2f} mm")
        c[3].metric("Grundkreis db2", f"{geo['db2']:.2f} mm")
        with st.expander("Weitere Geometriegrößen"):
            st.write(pd.DataFrame({
                "Größe": ["Stirneingriffswinkel α_t1", "Stirneingriffswinkel α_t2", "Grundschrägungswinkel β_b1",
                          "Ersatzzähnezahl z_n1", "Ersatzzähnezahl z_n2"],
                "Wert": [geo["alpha_t1"], geo["alpha_t2"], geo["beta_b1"], geo["zn1"], geo["zn2"]],
                "Einheit": ["°", "°", "°", "-", "-"]}).round(3))

    with tab_fest:
        c = st.columns(4)
        c[0].metric("σ_F Rad 1", f"{fe['sigma_F1']:.1f} MPa")
        c[1].metric("σ_F Rad 2", f"{fe['sigma_F2']:.1f} MPa")
        c[2].metric("S_F (min. beider Räder)", f"{fe['S_F']:.2f}", f"{fe['S_F'] - p['SF_min']:+.2f} ggü. Ziel")
        c[3].metric("σ_F,zul Rad 1", f"{fe['sigma_F_zul1']:.1f} MPa")
        c = st.columns(4)
        c[0].metric("σ_H", f"{fe['sigma_H']:.1f} MPa")
        c[1].metric("σ_H,zul", f"{fe['sigma_H_zul']:.1f} MPa")
        c[2].metric("S_H", f"{fe['S_H']:.2f}", f"{fe['S_H'] - p['SH_min']:+.2f} ggü. Ziel")
        c[3].metric("Temperaturfaktor Y_T", f"{fe['Y_T']:.2f}")
        st.subheader("Kräfte und Betriebsgrößen")
        c = st.columns(4)
        c[0].metric("Umfangskraft F_t1", f"{kr['Ft1']:.1f} N")
        c[1].metric("Axialkraft F_a1", f"{kr['Fa1']:.1f} N")
        c[2].metric("Gleitgeschw. v_g", f"{kr['v_g']:.2f} m/s")
        c[3].metric("Wirkungsgrad η", f"{kr['eta'] * 100:.1f} %")
        c = st.columns(4)
        c[0].metric("Abtriebsdrehzahl n2", f"{kr['n2']:.0f} 1/min")
        c[1].metric("Abtriebsmoment T2", f"{kr['T2']:.2f} Nm")
        c[2].metric("Verlustleistung P_V", f"{kr['P_V'] * 1000:.1f} W")
        c[3].metric("Lastwechsel N1", f"{fe['N1']:.2e}")
        with st.expander("Annahmen & Hinweise zum Rechenmodell"):
            st.markdown(
                "- Zahnfuß: Biegung nach Lewis-/DIN-3990-Ansatz mit Ersatzzähnezahl, Y_Fa/Y_Sa als Näherungsfunktionen (α_n = 20°, x = 0).\n"
                "- Flanke: Hertz'sche Pressung am Ersatz-Stirnradpaar, Punktberührung über Faktor Z_P berücksichtigt.\n"
                "- Festigkeitskennwerte werden mit Temperatur- (Y_T) und Lebensdauerfaktor (Y_N) abgemindert.\n"
                "- Werkstoffwerte sind Richtwerte – für die Bachelorarbeit mit Datenblättern/VDI 2736 abgleichen.")

    with tab_dia:
        st.plotly_chart(diagramm_geometrie(geo), use_container_width=True)
        groesse = st.selectbox("Parameterstudie über …", ["Drehzahl n1 [1/min]",
                                                          "Umgebungstemperatur T_umg [°C]", "Drehmoment T1 [Nm]"])
        st.plotly_chart(parameterstudie(p, groesse), use_container_width=True)

    with tab_exp:
        df = tabelle_export(p, geo, kr, fe, werkstoff_name)
        st.dataframe(df, use_container_width=True, hide_index=True)
        c = st.columns(2)
        c[0].download_button("⬇️ CSV herunterladen", df.to_csv(index=False, sep=";").encode("utf-8-sig"),
                             "schraubradgetriebe.csv", "text/csv")
        c[1].download_button("⬇️ HTML-Protokoll (als PDF druckbar)", html_protokoll(df, status).encode("utf-8"),
                             "protokoll_schraubradgetriebe.html", "text/html")


if __name__ == "__main__":
    main()
