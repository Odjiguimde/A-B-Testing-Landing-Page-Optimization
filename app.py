import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, norm
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="A/B Testing Dashboard",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Space Mono', monospace; }

    .stApp { background-color: #0d0f14; color: #e8eaf0; }

    .metric-card {
        background: linear-gradient(135deg, #1a1d26 0%, #1e2130 100%);
        border: 1px solid #2a2d3e;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }
    .metric-val { font-size: 2.2rem; font-weight: 700; font-family: 'Space Mono', monospace; }
    .metric-label { font-size: 0.85rem; color: #8b90a8; text-transform: uppercase; letter-spacing: 1px; }
    .metric-delta { font-size: 0.9rem; margin-top: 4px; }

    div[data-testid="stTabs"] button { font-family: 'Space Mono', monospace; font-size: 0.8rem; }

    [data-testid="stSidebar"] { background-color: #10121a !important; }
    [data-testid="stSidebar"] .stMarkdown { color: #e8eaf0; }

    .insight-box {
        background: #1a1d26;
        border-left: 3px solid #00d4aa;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin: 12px 0;
        font-size: 0.95rem;
    }
    .warning-box {
        background: #1a1d26;
        border-left: 3px solid #f39c12;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin: 12px 0;
        font-size: 0.95rem;
    }
    .info-box {
        background: #1a1d26;
        border-left: 3px solid #6c63ff;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin: 12px 0;
        font-size: 0.9rem;
        font-family: 'Space Mono', monospace;
        color: #8b90a8;
        line-height: 1.8;
    }
    .upload-zone {
        background: #1a1d26;
        border: 1px dashed #2a2d3e;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .col-tag {
        display: inline-block;
        background: #2a2d3e;
        color: #a29bfe;
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 4px;
        margin: 2px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #6c63ff, #a29bfe);
        color: white;
        border: none;
        font-family: 'Space Mono', monospace;
        font-size: 0.8rem;
    }
    .stFileUploader > div {
        background: #1a1d26;
        border-color: #2a2d3e;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
MPL_STYLE = dict(
    fig_bg='#0d0f14',
    ax_bg='#1a1d26',
    grid='#2a2d3e',
    text='#8b90a8',
    spine='#2a2d3e',
    c1='#6c63ff',
    c2='#00d4aa',
)

def style_ax(ax):
    ax.set_facecolor(MPL_STYLE['ax_bg'])
    ax.tick_params(colors=MPL_STYLE['text'])
    for spine in ax.spines.values():
        spine.set_color(MPL_STYLE['spine'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, color=MPL_STYLE['grid'], linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

def make_fig(*args, **kwargs):
    fig, ax = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(MPL_STYLE['fig_bg'])
    return fig, ax

@st.cache_data
def generate_synthetic():
    np.random.seed(42)
    n = 10000
    group = np.random.choice(['control', 'treatment'], size=n, p=[0.5, 0.5])
    conv_rate = np.where(group == 'treatment', 0.127, 0.112)
    converted = np.random.binomial(1, conv_rate)
    time_on_page = np.where(
        group == 'treatment',
        np.random.gamma(shape=3.5, scale=25, size=n),
        np.random.gamma(shape=3.0, scale=22, size=n)
    )
    clicks = np.where(
        group == 'treatment',
        np.random.poisson(lam=4.2, size=n),
        np.random.poisson(lam=3.8, size=n)
    )
    revenue = np.where(converted == 1, np.random.lognormal(mean=3.5, sigma=0.8, size=n), 0)
    device = np.random.choice(['desktop', 'mobile', 'tablet'], size=n, p=[0.55, 0.35, 0.10])
    country = np.random.choice(['FR', 'US', 'UK', 'DE', 'ES'], size=n, p=[0.3, 0.25, 0.2, 0.15, 0.1])
    age_group = np.random.choice(['18-24', '25-34', '35-44', '45-54', '55+'], size=n, p=[0.15, 0.30, 0.25, 0.18, 0.12])
    timestamps = pd.date_range('2024-01-01', periods=n, freq='1min')
    idx = np.arange(n); np.random.shuffle(idx)
    timestamps = timestamps[idx]
    df = pd.DataFrame({
        'user_id': range(1, n+1),
        'timestamp': timestamps,
        'group': group,
        'converted': converted,
        'time_on_page': time_on_page.round(1),
        'clicks': clicks,
        'revenue': revenue.round(2),
        'device': device,
        'country': country,
        'age_group': age_group
    })
    return df

def engineer_features(df):
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
    return df

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'col_map' not in st.session_state:
    st.session_state.col_map = {}
if 'data_ready' not in st.session_state:
    st.session_state.data_ready = False

# ─────────────────────────────────────────────
# SIDEBAR — DATA LOADING
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## A/B Testing Dashboard")
    st.markdown("---")
    st.markdown("### Source de données")

    data_source = st.radio("", ["Dataset synthétique", "Charger un CSV"], label_visibility="collapsed")

    if data_source == "Dataset synthétique":
        df_raw = generate_synthetic()
        df_raw = engineer_features(df_raw)
        st.session_state.df_raw = df_raw
        st.session_state.col_map = {
            'group': 'group',
            'converted': 'converted',
            'revenue': 'revenue',
            'time_on_page': 'time_on_page',
            'clicks': 'clicks',
        }
        st.session_state.data_ready = True
        st.caption(f"Dataset synthétique — {len(df_raw):,} lignes, {df_raw.shape[1]} colonnes")

    else:
        uploaded = st.file_uploader("Fichier CSV", type=["csv"])

        if uploaded is not None:
            try:
                df_raw = pd.read_csv(uploaded)
                df_raw = engineer_features(df_raw)
                st.session_state.df_raw = df_raw
                st.success(f"{len(df_raw):,} lignes · {df_raw.shape[1]} colonnes")

                st.markdown("---")
                st.markdown("### Mappage des colonnes")
                st.caption("Indiquez quelle colonne correspond à chaque variable clé.")

                all_cols = ["(aucune)"] + df_raw.columns.tolist()
                num_cols = ["(aucune)"] + df_raw.select_dtypes(include=np.number).columns.tolist()

                def guess(candidates):
                    for c in candidates:
                        for col in df_raw.columns:
                            if c in col.lower():
                                return col
                    return "(aucune)"

                col_group     = st.selectbox("Colonne groupe (A/B)",     all_cols, index=all_cols.index(guess(['group','variant','test','bucket','arm'])))
                col_converted = st.selectbox("Colonne conversion (0/1)", all_cols, index=all_cols.index(guess(['convert','success','purchase','click','outcome'])))
                col_revenue   = st.selectbox("Colonne revenu (optionnel)", num_cols, index=0)
                col_time      = st.selectbox("Colonne temps (optionnel)", num_cols, index=0)
                col_clicks    = st.selectbox("Colonne clics (optionnel)", num_cols, index=0)

                if col_group != "(aucune)" and col_converted != "(aucune)":
                    # Normaliser les valeurs de groupe
                    unique_groups = df_raw[col_group].dropna().unique().tolist()
                    st.markdown("**Identifier les groupes**")
                    ctrl_val = st.selectbox("Valeur = Contrôle (A)", unique_groups)
                    trt_val  = st.selectbox("Valeur = Traitement (B)", [v for v in unique_groups if v != ctrl_val])

                    st.session_state.col_map = {
                        'group': col_group,
                        'converted': col_converted,
                        'revenue': col_revenue if col_revenue != "(aucune)" else None,
                        'time_on_page': col_time if col_time != "(aucune)" else None,
                        'clicks': col_clicks if col_clicks != "(aucune)" else None,
                        'ctrl_val': ctrl_val,
                        'trt_val': trt_val,
                    }
                    st.session_state.data_ready = True
                else:
                    st.warning("Sélectionnez au minimum les colonnes groupe et conversion.")
                    st.session_state.data_ready = False

            except Exception as e:
                st.error(f"Erreur de lecture : {e}")
                st.session_state.data_ready = False

    if not st.session_state.data_ready:
        st.stop()

    st.markdown("---")
    df = st.session_state.df_raw
    col_map = st.session_state.col_map

    # Normaliser le groupe si CSV custom
    if 'ctrl_val' in col_map:
        df = df.copy()
        df['_group'] = df[col_map['group']].map(
            {col_map['ctrl_val']: 'control', col_map['trt_val']: 'treatment'}
        )
        df = df.dropna(subset=['_group'])
    else:
        df = df.copy()
        df['_group'] = df[col_map['group']]

    df['_converted'] = pd.to_numeric(df[col_map['converted']], errors='coerce').fillna(0).astype(int)

    # Filtres dynamiques
    st.markdown("### Filtres")
    EXCLUDE = {'user_id','timestamp','date','hour','week','_group','_converted',
               col_map['group'], col_map['converted']}
    filter_cols = [c for c in df.columns if c not in EXCLUDE
                   and df[c].dtype == object and 2 <= df[c].nunique() <= 30]

    df_filtered = df.copy()
    for fc in filter_cols:
        vals = ['Tous'] + sorted(df[fc].dropna().unique().tolist())
        chosen = st.selectbox(fc.replace('_',' ').title(), vals)
        if chosen != 'Tous':
            df_filtered = df_filtered[df_filtered[fc] == chosen]

    st.markdown("---")
    st.markdown("### Parametres statistiques")
    alpha = st.slider("Seuil alpha", 0.01, 0.10, 0.05, 0.01)
    st.markdown(f"Intervalle de confiance : **{int((1-alpha)*100)}%**")
    st.markdown("---")
    st.caption("Projet 2 — A/B Testing · Analyse statistique")

# ─────────────────────────────────────────────
# COMPUTED DATA
# ─────────────────────────────────────────────
control   = df_filtered[df_filtered['_group'] == 'control']
treatment = df_filtered[df_filtered['_group'] == 'treatment']

n_control   = len(control)
n_treatment = len(treatment)

if n_control == 0 or n_treatment == 0:
    st.error("Aucune donnée disponible pour l'un des groupes avec les filtres actuels.")
    st.stop()

conv_control   = control['_converted'].mean()
conv_treatment = treatment['_converted'].mean()
lift = (conv_treatment - conv_control) / conv_control * 100 if conv_control > 0 else 0

contingency = pd.crosstab(df_filtered['_group'], df_filtered['_converted'])
chi2_stat, p_chi2, dof, _ = chi2_contingency(contingency)

p_pool  = df_filtered['_converted'].mean()
se      = np.sqrt(p_pool * (1 - p_pool) * (1/n_control + 1/n_treatment)) if p_pool > 0 else 1e-9
z_score = (conv_treatment - conv_control) / se
p_ztest = 2 * (1 - norm.cdf(abs(z_score)))
ci_low  = (conv_treatment - conv_control) - norm.ppf(1 - alpha/2) * se
ci_high = (conv_treatment - conv_control) + norm.ppf(1 - alpha/2) * se

is_significant = p_chi2 < alpha

def get_col(name):
    """Get actual series from col_map if available."""
    mapped = col_map.get(name)
    if mapped and mapped in df_filtered.columns:
        return df_filtered[mapped], control[mapped], treatment[mapped]
    return None, None, None

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# A/B Testing — Analyse expérimentale")
st.markdown(f"**{len(df_filtered):,} utilisateurs** analysés · Seuil α = {alpha} · IC = {int((1-alpha)*100)}%")
st.markdown("---")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tabs = st.tabs(["Vue d'ensemble", "Tests statistiques", "Visualisations", "Analyse segmentée", "Décision"])

# ════════════════════════════════════════════
# TAB 1 — VUE D'ENSEMBLE
# ════════════════════════════════════════════
with tabs[0]:
    st.markdown("### Métriques clés")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Utilisateurs — Contrôle</div>
            <div class="metric-val">{n_control:,}</div>
            <div class="metric-delta" style="color:#8b90a8">Version A (originale)</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Utilisateurs — Traitement</div>
            <div class="metric-val">{n_treatment:,}</div>
            <div class="metric-delta" style="color:#8b90a8">Version B (nouvelle)</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        color = "#00d4aa" if lift > 0 else "#e74c3c"
        sign  = "+" if lift > 0 else ""
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Lift (Uplift)</div>
            <div class="metric-val" style="color:{color}">{sign}{lift:.2f}%</div>
            <div class="metric-delta" style="color:#8b90a8">Variation relative</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        sig_color = "#00d4aa" if is_significant else "#f39c12"
        sig_text  = "Significatif" if is_significant else "Non Significatif"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Résultat statistique</div>
            <div class="metric-val" style="color:{sig_color}; font-size:1.3rem">{sig_text}</div>
            <div class="metric-delta" style="color:#8b90a8">p-value = {p_chi2:.4f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("### Taux de conversion par groupe")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = make_fig(figsize=(7, 4))
        style_ax(ax)
        rates  = [conv_control * 100, conv_treatment * 100]
        bars   = ax.bar(['Contrôle (A)', 'Traitement (B)'], rates,
                        color=[MPL_STYLE['c1'], MPL_STYLE['c2']], width=0.5, edgecolor='none', zorder=3)
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{rate:.2f}%', ha='center', va='bottom',
                    color='white', fontweight='bold', fontsize=13, fontfamily='monospace')
        ax.set_ylabel('Taux de conversion (%)', color=MPL_STYLE['text'])
        ax.set_ylim(0, max(rates) * 1.3)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        summary = pd.DataFrame({
            'Groupe': ['Contrôle (A)', 'Traitement (B)'],
            'Utilisateurs': [f"{n_control:,}", f"{n_treatment:,}"],
            'Convertis': [f"{int(control['_converted'].sum()):,}", f"{int(treatment['_converted'].sum()):,}"],
            'Taux conv.': [f"{conv_control*100:.3f}%", f"{conv_treatment*100:.3f}%"],
        })
        _, ctrl_rev, trt_rev = get_col('revenue')
        if ctrl_rev is not None:
            summary['Revenu moyen'] = [f"€{ctrl_rev.mean():.2f}", f"€{trt_rev.mean():.2f}"]
        _, ctrl_time, trt_time = get_col('time_on_page')
        if ctrl_time is not None:
            summary['Temps moyen (s)'] = [f"{ctrl_time.mean():.1f}", f"{trt_time.mean():.1f}"]

        st.markdown("#### Tableau récapitulatif")
        st.dataframe(summary.set_index('Groupe'), use_container_width=True)

        if is_significant:
            winner = 'Traitement (B)' if conv_treatment > conv_control else 'Contrôle (A)'
            st.markdown(f"""<div class="insight-box">
                <strong>Vainqueur détecté : {winner}</strong><br>
                Lift de <strong>{lift:+.2f}%</strong> — différence statistiquement significative (p={p_chi2:.4f} &lt; α={alpha})
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="warning-box">
                <strong>Pas de conclusion définitive</strong><br>
                La différence observée n'est pas statistiquement significative (p={p_chi2:.4f} &gt; α={alpha}).
                Plus de données sont nécessaires.
            </div>""", unsafe_allow_html=True)

    # Dataset info
    with st.expander("Apercu du dataset chargé"):
        st.markdown(f"""<div class="info-box">
            Lignes : {len(df_filtered):,} &nbsp;·&nbsp; Colonnes : {df_filtered.shape[1]}<br>
            Groupe contrôle : <strong>{col_map.get('ctrl_val', 'control')}</strong> &nbsp;·&nbsp;
            Groupe traitement : <strong>{col_map.get('trt_val', 'treatment')}</strong>
        </div>""", unsafe_allow_html=True)
        st.dataframe(df_filtered.head(100), use_container_width=True)

# ════════════════════════════════════════════
# TAB 2 — TESTS STATISTIQUES
# ════════════════════════════════════════════
with tabs[1]:
    st.markdown("### Résultats des tests statistiques")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Test du Chi² (indépendance)")
        for label, val, color_cond in [
            ("Statistique chi²", f"{chi2_stat:.4f}", None),
            ("p-value", f"{p_chi2:.6f}", p_chi2 < alpha),
            ("Degrés de liberté", str(dof), None),
        ]:
            vc = ("#00d4aa" if color_cond else "#f39c12") if color_cond is not None else "#e8eaf0"
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-val" style="color:{vc}">{val}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("**Table de contingence**")
        ct = pd.crosstab(df_filtered['_group'], df_filtered['_converted'], margins=True)
        ct.columns = ['Non converti', 'Converti', 'Total']
        ct.index   = ['Contrôle', 'Traitement', 'Total']
        st.dataframe(ct, use_container_width=True)

    with col2:
        st.markdown("#### Z-test sur proportions")
        for label, val, color_cond in [
            ("Z-score", f"{z_score:.4f}", None),
            ("p-value (bilatéral)", f"{p_ztest:.6f}", p_ztest < alpha),
            (f"IC {int((1-alpha)*100)}% (différence)", f"[{ci_low:.4f}, {ci_high:.4f}]", None),
        ]:
            vc = ("#00d4aa" if color_cond else "#f39c12") if color_cond is not None else "#e8eaf0"
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-val" style="color:{vc}; font-size:1.3rem">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Tests sur métriques secondaires")
    col1, col2 = st.columns(2)

    with col1:
        _, ctrl_time, trt_time = get_col('time_on_page')
        if ctrl_time is not None:
            t_stat, p_ttest = ttest_ind(ctrl_time.dropna(), trt_time.dropna())
            u_stat, p_mwu   = mannwhitneyu(ctrl_time.dropna(), trt_time.dropna(), alternative='two-sided')
            st.markdown("#### Temps sur la page")
            st.dataframe(pd.DataFrame({
                'Test': ['t-test (Student)', 'Mann-Whitney U'],
                'Statistique': [f"{t_stat:.4f}", f"{u_stat:.0f}"],
                'p-value': [f"{p_ttest:.6f}", f"{p_mwu:.6f}"],
                'Significatif': ['Oui' if p_ttest < alpha else 'Non',
                                 'Oui' if p_mwu < alpha else 'Non'],
            }).set_index('Test'), use_container_width=True)

    with col2:
        _, ctrl_clicks, trt_clicks = get_col('clicks')
        if ctrl_clicks is not None:
            t2, p2 = ttest_ind(ctrl_clicks.dropna(), trt_clicks.dropna())
            st.markdown("#### Nombre de clics")
            st.dataframe(pd.DataFrame({
                'Test': ['t-test (Student)'],
                'Statistique': [f"{t2:.4f}"],
                'p-value': [f"{p2:.6f}"],
                'Significatif': ['Oui' if p2 < alpha else 'Non'],
            }).set_index('Test'), use_container_width=True)

    st.markdown("---")
    st.markdown("### Interprétation")
    decision = "rejetons" if is_significant else "ne pouvons pas rejeter"
    critical_z = norm.ppf(1 - alpha/2)
    st.markdown(f"""
**H₀ (hypothèse nulle)** : Il n'y a pas de différence significative entre les taux de conversion des deux groupes.

**H₁ (hypothèse alternative)** : Le taux de conversion du groupe traitement est différent de celui du groupe contrôle.

Avec p = {p_chi2:.6f} {"<" if is_significant else ">"} α = {alpha}, nous **{decision} H₀**.

Le Z-score de {z_score:.2f} {"dépasse" if abs(z_score) > critical_z else "ne dépasse pas"} la valeur critique
de ±{critical_z:.2f} pour un seuil α = {alpha}.
    """)

# ════════════════════════════════════════════
# TAB 3 — VISUALISATIONS
# ════════════════════════════════════════════
with tabs[2]:
    st.markdown("### Visualisations analytiques")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Distribution des conversions")
        fig, ax = make_fig(figsize=(7, 4)); style_ax(ax)
        x = np.arange(2); w = 0.35
        ax.bar(x - w/2, [conv_control*100, conv_treatment*100], w,
               label='Converti', color=[MPL_STYLE['c1'], MPL_STYLE['c2']], zorder=3)
        ax.bar(x + w/2, [(1-conv_control)*100, (1-conv_treatment)*100], w,
               label='Non converti', color=['#3d3880', '#007a60'], zorder=3, alpha=0.7)
        ax.set_xticks(x); ax.set_xticklabels(['Contrôle (A)', 'Traitement (B)'], color='#e8eaf0')
        ax.set_ylabel('Pourcentage (%)', color=MPL_STYLE['text'])
        ax.legend(facecolor='#2a2d3e', edgecolor='none', labelcolor='#e8eaf0')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        _, ctrl_time, trt_time = get_col('time_on_page')
        if ctrl_time is not None:
            st.markdown("#### Distribution — Temps sur la page")
            fig, ax = make_fig(figsize=(7, 4)); style_ax(ax)
            ax.hist(ctrl_time, bins=40, alpha=0.6, color=MPL_STYLE['c1'],
                    label=f'Contrôle (μ={ctrl_time.mean():.1f}s)', density=True, zorder=3)
            ax.hist(trt_time,  bins=40, alpha=0.6, color=MPL_STYLE['c2'],
                    label=f'Traitement (μ={trt_time.mean():.1f}s)', density=True, zorder=3)
            ax.axvline(ctrl_time.mean(), color=MPL_STYLE['c1'], linestyle='--', alpha=0.8)
            ax.axvline(trt_time.mean(),  color=MPL_STYLE['c2'], linestyle='--', alpha=0.8)
            ax.set_xlabel('Temps (secondes)', color=MPL_STYLE['text'])
            ax.set_ylabel('Densité', color=MPL_STYLE['text'])
            ax.legend(facecolor='#2a2d3e', edgecolor='none', labelcolor='#e8eaf0', fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close()
        else:
            st.markdown("""<div class="warning-box">
                Colonne temps non mappée — sélectionnez-la dans la sidebar.
            </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if 'date' in df_filtered.columns:
            st.markdown("#### Evolution du taux de conversion")
            daily = df_filtered.groupby(['date', '_group'])['_converted'].mean().reset_index()
            fig, ax = make_fig(figsize=(7, 4)); style_ax(ax)
            for grp, color, label in [('control', MPL_STYLE['c1'], 'Contrôle (A)'),
                                       ('treatment', MPL_STYLE['c2'], 'Traitement (B)')]:
                data = daily[daily['_group'] == grp].sort_values('date')
                ax.plot(range(len(data)), data['_converted'] * 100,
                        color=color, linewidth=2, label=label, zorder=3)
                ax.fill_between(range(len(data)), data['_converted'] * 100, alpha=0.1, color=color)
            ax.set_xlabel('Jours', color=MPL_STYLE['text'])
            ax.set_ylabel('Taux de conversion (%)', color=MPL_STYLE['text'])
            ax.legend(facecolor='#2a2d3e', edgecolor='none', labelcolor='#e8eaf0', fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("#### Distribution bootstrap (B − A)")
        np.random.seed(42)
        boot_diffs = [
            np.random.choice(treatment['_converted'], size=min(500, n_treatment), replace=True).mean() -
            np.random.choice(control['_converted'],   size=min(500, n_control),   replace=True).mean()
            for _ in range(500)
        ]
        fig, ax = make_fig(figsize=(7, 4)); style_ax(ax)
        ax.hist(boot_diffs, bins=40, color=MPL_STYLE['c1'], alpha=0.7, density=True, zorder=3)
        ax.axvline(0, color='#e74c3c', linestyle='--', linewidth=2, label='H0: diff=0')
        ax.axvline(np.mean(boot_diffs), color=MPL_STYLE['c2'], linestyle='--', linewidth=2,
                   label=f'Diff observée: {np.mean(boot_diffs):.4f}')
        ci_lo = np.percentile(boot_diffs, alpha/2 * 100)
        ci_hi = np.percentile(boot_diffs, (1 - alpha/2) * 100)
        ax.axvspan(ci_lo, ci_hi, alpha=0.1, color=MPL_STYLE['c2'], label=f'IC {int((1-alpha)*100)}%')
        ax.set_xlabel('Différence de conversion (B − A)', color=MPL_STYLE['text'])
        ax.set_ylabel('Densité', color=MPL_STYLE['text'])
        ax.legend(facecolor='#2a2d3e', edgecolor='none', labelcolor='#e8eaf0', fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ════════════════════════════════════════════
# TAB 4 — ANALYSE SEGMENTÉE
# ════════════════════════════════════════════
with tabs[3]:
    st.markdown("### Analyse par sous-groupes (Heterogeneous Treatment Effects)")

    EXCLUDE_SEG = {'user_id','timestamp','date','hour','week','_group','_converted',
                   col_map['group'], col_map['converted']}
    segment_cols = [
        c for c in df.columns
        if c not in EXCLUDE_SEG
        and df[c].dtype == object
        and 2 <= df[c].nunique() <= 20
    ]

    if not segment_cols:
        st.markdown("""<div class="warning-box">
            <strong>Aucune colonne de segmentation disponible</strong><br>
            Le dataset ne contient pas de variables catégorielles permettant une analyse par sous-groupe.<br>
            Utilisez le dataset synthétique ou chargez un CSV enrichi (device, pays, age...).
        </div>""", unsafe_allow_html=True)
    else:
        LABEL_MAP = {'device': 'Device', 'country': 'Pays', 'age_group': "Tranche d'age",
                     'landing_page': 'Landing page', 'browser': 'Navigateur', 'os': 'OS'}
        selected_seg = st.selectbox(
            "Segmenter par",
            segment_cols,
            format_func=lambda x: LABEL_MAP.get(x, x.replace('_', ' ').title())
        )

        seg_results = []
        for seg_val in sorted(df[selected_seg].unique()):
            seg_df   = df[df[selected_seg] == seg_val]
            seg_ctrl = seg_df[seg_df['_group'] == 'control']
            seg_trt  = seg_df[seg_df['_group'] == 'treatment']
            if len(seg_ctrl) < 30 or len(seg_trt) < 30:
                continue
            seg_conv_c = seg_ctrl['_converted'].mean()
            seg_conv_t = seg_trt['_converted'].mean()
            seg_lift   = (seg_conv_t - seg_conv_c) / seg_conv_c * 100 if seg_conv_c > 0 else 0
            ct = pd.crosstab(seg_df['_group'], seg_df['_converted'])
            p_val = chi2_contingency(ct)[1] if ct.shape == (2, 2) else 1.0
            seg_results.append({
                'Segment': seg_val,
                'N Contrôle': len(seg_ctrl),
                'N Traitement': len(seg_trt),
                'Conv. Contrôle': f"{seg_conv_c*100:.2f}%",
                'Conv. Traitement': f"{seg_conv_t*100:.2f}%",
                'Lift': f"{seg_lift:+.2f}%",
                'p-value': f"{p_val:.4f}",
                'Significatif': 'Oui' if p_val < alpha else 'Non',
                '_lift_val': seg_lift,
                '_p_val': p_val,
            })

        if not seg_results:
            st.markdown("""<div class="warning-box">
                Aucun segment ne contient au moins 30 utilisateurs par groupe. Désactivez les filtres.
            </div>""", unsafe_allow_html=True)
        else:
            seg_df_r = pd.DataFrame(seg_results)
            segments = seg_df_r['Segment'].tolist()
            lifts    = seg_df_r['_lift_val'].tolist()
            p_vals   = seg_df_r['_p_val'].tolist()

            fig, axes = plt.subplots(1, 2, figsize=(12, max(3, len(seg_results) * 0.8 + 1)))
            fig.patch.set_facecolor(MPL_STYLE['fig_bg'])

            ax = axes[0]; ax.set_facecolor(MPL_STYLE['ax_bg'])
            ax.barh(segments, lifts,
                    color=['#00d4aa' if l > 0 else '#e74c3c' for l in lifts], zorder=3)
            ax.axvline(0, color=MPL_STYLE['text'], linewidth=1, linestyle='--')
            ax.set_xlabel('Lift (%)', color=MPL_STYLE['text'])
            ax.set_title(f'Lift par {selected_seg}', color='#e8eaf0', fontfamily='monospace')
            ax.tick_params(colors=MPL_STYLE['text'])
            for sp in ax.spines.values(): sp.set_color(MPL_STYLE['spine'])
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.xaxis.grid(True, color=MPL_STYLE['grid'], linestyle='--', alpha=0.5, zorder=0)
            ax.set_axisbelow(True)

            ax2 = axes[1]; ax2.set_facecolor(MPL_STYLE['ax_bg'])
            ax2.barh(segments, p_vals,
                     color=['#00d4aa' if p < alpha else '#f39c12' for p in p_vals], zorder=3)
            ax2.axvline(alpha, color='#e74c3c', linewidth=2, linestyle='--', label=f'α={alpha}')
            ax2.set_xlabel('p-value', color=MPL_STYLE['text'])
            ax2.set_title('p-value par segment', color='#e8eaf0', fontfamily='monospace')
            ax2.tick_params(colors=MPL_STYLE['text'])
            for sp in ax2.spines.values(): sp.set_color(MPL_STYLE['spine'])
            ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
            ax2.legend(facecolor='#2a2d3e', edgecolor='none', labelcolor='#e8eaf0')
            ax2.xaxis.grid(True, color=MPL_STYLE['grid'], linestyle='--', alpha=0.5, zorder=0)
            ax2.set_axisbelow(True)

            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.dataframe(seg_df_r.drop(columns=['_lift_val','_p_val']).set_index('Segment'),
                         use_container_width=True)

# ════════════════════════════════════════════
# TAB 5 — DÉCISION
# ════════════════════════════════════════════
with tabs[4]:
    st.markdown("### Rapport de décision")

    total_users_monthly      = st.number_input("Estimation utilisateurs/mois", value=50000, step=1000)
    avg_revenue_per_conv     = st.number_input("Revenu moyen par conversion (EUR)", value=45.0, step=5.0)

    monthly_delta_convs    = total_users_monthly * (conv_treatment - conv_control)
    monthly_revenue_delta  = monthly_delta_convs * avg_revenue_per_conv

    st.markdown("#### Impact business estimé (déploiement Version B)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Conversions supplémentaires/mois</div>
            <div class="metric-val" style="color:#00d4aa">{monthly_delta_convs:+,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        color = "#00d4aa" if monthly_revenue_delta > 0 else "#e74c3c"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Revenu additionnel/mois</div>
            <div class="metric-val" style="color:{color}">€{monthly_revenue_delta:+,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Revenu additionnel/an</div>
            <div class="metric-val" style="color:#00d4aa">€{monthly_revenue_delta*12:+,.0f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Recommandation finale")

    if is_significant and lift > 0:
        recommendation = "DEPLOYER la Version B (Traitement)"
        rec_color = "#00d4aa"
        rationale = f"""
        <ul style="margin:0;padding-left:20px;">
            <li>Test <strong>statistiquement significatif</strong> (p={p_chi2:.4f} &lt; α={alpha})</li>
            <li>Version B : lift de <strong>+{lift:.2f}%</strong> sur le taux de conversion</li>
            <li>IC {int((1-alpha)*100)}% exclut 0 : [{ci_low:.4f}, {ci_high:.4f}]</li>
            <li>Impact estimé : <strong>€{monthly_revenue_delta:+,.0f}/mois</strong> de revenus additionnels</li>
        </ul>"""
    elif is_significant and lift < 0:
        recommendation = "CONSERVER la Version A (Contrôle)"
        rec_color = "#e74c3c"
        rationale = f"""
        <ul style="margin:0;padding-left:20px;">
            <li>Test <strong>statistiquement significatif</strong> (p={p_chi2:.4f} &lt; α={alpha})</li>
            <li>Version B <strong>moins performante</strong> : lift de {lift:.2f}%</li>
            <li>Conserver la Version A et explorer de nouvelles hypothèses</li>
        </ul>"""
    else:
        recommendation = "CONTINUER le test — résultat non concluant"
        rec_color = "#f39c12"
        rationale = f"""
        <ul style="margin:0;padding-left:20px;">
            <li>Test <strong>non significatif</strong> (p={p_chi2:.4f} &gt; α={alpha})</li>
            <li>La différence observée peut être due au hasard</li>
            <li>Collecter davantage de données ou revoir le design de l'expérience</li>
        </ul>"""

    st.markdown(f"""<div style="background:linear-gradient(135deg,#1a1d26,#1e2130);
        border:2px solid {rec_color};border-radius:12px;padding:24px 28px;margin:16px 0;">
        <div style="font-size:1.3rem;font-weight:700;color:{rec_color};
                    font-family:'Space Mono',monospace;margin-bottom:12px;">{recommendation}</div>
        <div style="color:#e8eaf0;line-height:1.8;">{rationale}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("#### Prochaines étapes")
    for step in [
        "**Analyse de segmentation approfondie** : identifier les sous-groupes à fort effet",
        "**Test de durabilité** : vérifier que l'effet ne s'estompe pas dans le temps (novelty effect)",
        "**Tests multivariés (MVT)** : tester plusieurs éléments simultanément",
        "**Analyse coût-bénéfice** : coût d'implémentation vs gain attendu",
        "**Déploiement progressif** : rollout par paliers (10% → 50% → 100%)",
    ]:
        st.markdown(f"→ {step}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Télécharger les données filtrées (CSV)",
            data=df_filtered.to_csv(index=False),
            file_name="ab_test_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col2:
        report = "\n".join([
            "RAPPORT A/B TEST", "=" * 50,
            f"Date : {pd.Timestamp.now().strftime('%Y-%m-%d')}",
            f"Total utilisateurs : {len(df_filtered):,}", "",
            "RÉSULTATS", "-" * 30,
            f"Contrôle (A) : {n_control:,} users | Conv. : {conv_control*100:.3f}%",
            f"Traitement (B) : {n_treatment:,} users | Conv. : {conv_treatment*100:.3f}%",
            f"Lift : {lift:+.2f}%", "",
            "TESTS STATISTIQUES", "-" * 30,
            f"Chi² : {chi2_stat:.4f} | p-value : {p_chi2:.6f}",
            f"Z-score : {z_score:.4f} | p-value : {p_ztest:.6f}",
            f"Seuil alpha : {alpha}",
            f"Résultat : {'SIGNIFICATIF' if is_significant else 'NON SIGNIFICATIF'}", "",
            "RECOMMANDATION", "-" * 30, recommendation,
        ])
        st.download_button(
            label="Télécharger le rapport (TXT)",
            data=report,
            file_name="rapport_ab_test.txt",
            mime="text/plain",
            use_container_width=True
        )
