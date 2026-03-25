import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, norm
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="A/B Testing Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
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

    .winner-badge {
        display: inline-block;
        background: linear-gradient(90deg, #00d4aa, #00b894);
        color: #0d0f14;
        font-weight: 700;
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        padding: 4px 12px;
        border-radius: 20px;
        letter-spacing: 1px;
    }
    .loser-badge {
        display: inline-block;
        background: #2a2d3e;
        color: #8b90a8;
        font-weight: 700;
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        padding: 4px 12px;
        border-radius: 20px;
        letter-spacing: 1px;
    }
    .sig-badge {
        display: inline-block;
        background: linear-gradient(90deg, #6c63ff, #a29bfe);
        color: white;
        font-weight: 700;
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        padding: 4px 12px;
        border-radius: 20px;
    }
    .notsig-badge {
        display: inline-block;
        background: #f39c12;
        color: #0d0f14;
        font-weight: 700;
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        padding: 4px 12px;
        border-radius: 20px;
    }

    div[data-testid="stTabs"] button {
        font-family: 'Space Mono', monospace;
        font-size: 0.8rem;
    }

    .stSidebar { background-color: #10121a !important; }
    .stSidebar .stMarkdown { color: #e8eaf0; }

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
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('ab_data.csv')
    except FileNotFoundError:
        # Génération d'un dataset synthétique réaliste si le CSV n'est pas présent
        np.random.seed(42)
        n = 10000
        group = np.random.choice(['control', 'treatment'], size=n, p=[0.5, 0.5])
        
        # Conversion rates différentes
        conv_rate = np.where(group == 'treatment', 0.127, 0.112)
        converted = np.random.binomial(1, conv_rate)
        
        # Temps passé sur la page (secondes)
        time_on_page = np.where(
            group == 'treatment',
            np.random.gamma(shape=3.5, scale=25, size=n),
            np.random.gamma(shape=3.0, scale=22, size=n)
        )
        
        # Nombre de clics
        clicks = np.where(
            group == 'treatment',
            np.random.poisson(lam=4.2, size=n),
            np.random.poisson(lam=3.8, size=n)
        )
        
        # Revenu généré (0 si non converti)
        revenue = np.where(
            converted == 1,
            np.random.lognormal(mean=3.5, sigma=0.8, size=n),
            0
        )
        
        # Variables démographiques
        device = np.random.choice(['desktop', 'mobile', 'tablet'], size=n, p=[0.55, 0.35, 0.10])
        country = np.random.choice(['FR', 'US', 'UK', 'DE', 'ES'], size=n, p=[0.3, 0.25, 0.2, 0.15, 0.1])
        age_group = np.random.choice(['18-24', '25-34', '35-44', '45-54', '55+'], size=n, p=[0.15, 0.30, 0.25, 0.18, 0.12])
        
        # Timestamps
        timestamps = pd.date_range('2024-01-01', periods=n, freq='1min')
        np.random.shuffle(timestamps)
        
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
    
    # Feature engineering
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['week'] = df['timestamp'].dt.isocalendar().week
    
    return df

df = load_data()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.markdown("## 🧪 A/B Testing Dashboard")
st.sidebar.markdown("---")

st.sidebar.markdown("### Filtres")

# Filtre device
devices = ['Tous'] + sorted(df['device'].unique().tolist()) if 'device' in df.columns else ['Tous']
selected_device = st.sidebar.selectbox("📱 Device", devices)

# Filtre pays
if 'country' in df.columns:
    countries = ['Tous'] + sorted(df['country'].unique().tolist())
    selected_country = st.sidebar.selectbox("🌍 Pays", countries)
else:
    selected_country = 'Tous'

# Filtre âge
if 'age_group' in df.columns:
    age_groups = ['Tous'] + sorted(df['age_group'].unique().tolist())
    selected_age = st.sidebar.selectbox("👤 Tranche d'âge", age_groups)
else:
    selected_age = 'Tous'

st.sidebar.markdown("---")
st.sidebar.markdown("### Paramètres statistiques")
alpha = st.sidebar.slider("Seuil α (niveau de significativité)", 0.01, 0.10, 0.05, 0.01)
st.sidebar.markdown(f"Intervalle de confiance : **{int((1-alpha)*100)}%**")

st.sidebar.markdown("---")
st.sidebar.markdown("### À propos")
st.sidebar.markdown("""
**Projet 2 — A/B Testing**  
Expérimentation contrôlée sur landing pages.  
Analyse statistique rigoureuse pour la prise de décision.
""")

# ─────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────
df_filtered = df.copy()
if selected_device != 'Tous' and 'device' in df.columns:
    df_filtered = df_filtered[df_filtered['device'] == selected_device]
if selected_country != 'Tous' and 'country' in df.columns:
    df_filtered = df_filtered[df_filtered['country'] == selected_country]
if selected_age != 'Tous' and 'age_group' in df.columns:
    df_filtered = df_filtered[df_filtered['age_group'] == selected_age]

control = df_filtered[df_filtered['group'] == 'control']
treatment = df_filtered[df_filtered['group'] == 'treatment']

# ─────────────────────────────────────────────
# COMPUTED STATS
# ─────────────────────────────────────────────
n_control = len(control)
n_treatment = len(treatment)
conv_control = control['converted'].mean()
conv_treatment = treatment['converted'].mean()
lift = (conv_treatment - conv_control) / conv_control * 100

# Chi² test
contingency = pd.crosstab(df_filtered['group'], df_filtered['converted'])
chi2, p_chi2, dof, expected = chi2_contingency(contingency)

# Z-test for proportions
p_pool = df_filtered['converted'].mean()
se = np.sqrt(p_pool * (1 - p_pool) * (1/n_control + 1/n_treatment))
z_score = (conv_treatment - conv_control) / se if se > 0 else 0
p_ztest = 2 * (1 - norm.cdf(abs(z_score)))

# Power calculation
effect_size = abs(conv_treatment - conv_control) / np.sqrt(p_pool * (1 - p_pool))

is_significant = p_chi2 < alpha

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# 🧪 A/B Testing — Landing Page")
st.markdown(f"**{len(df_filtered):,} utilisateurs** analysés · Seuil α = {alpha}")
st.markdown("---")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tabs = st.tabs(["📊 Vue d'ensemble", "🔬 Tests statistiques", "📈 Visualisations", "🔍 Analyse segmentée", "💡 Décision"])

# ════════════════════════════════════════════
# TAB 1 — VUE D'ENSEMBLE
# ════════════════════════════════════════════
with tabs[0]:
    st.markdown("### Métriques clés")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Utilisateurs — Contrôle</div>
            <div class="metric-val">{n_control:,}</div>
            <div class="metric-delta" style="color:#8b90a8">Version A (originale)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Utilisateurs — Traitement</div>
            <div class="metric-val">{n_treatment:,}</div>
            <div class="metric-delta" style="color:#8b90a8">Version B (nouvelle)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        color = "#00d4aa" if lift > 0 else "#e74c3c"
        sign = "+" if lift > 0 else ""
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Lift (Uplift)</div>
            <div class="metric-val" style="color:{color}">{sign}{lift:.2f}%</div>
            <div class="metric-delta" style="color:#8b90a8">Variation relative</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        sig_color = "#00d4aa" if is_significant else "#f39c12"
        sig_text = "✅ Significatif" if is_significant else "⚠️ Non Significatif"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Résultat statistique</div>
            <div class="metric-val" style="color:{sig_color}; font-size:1.4rem">{sig_text}</div>
            <div class="metric-delta" style="color:#8b90a8">p-value = {p_chi2:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Taux de conversion par groupe")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor('#0d0f14')
        ax.set_facecolor('#1a1d26')
        
        groups = ['Contrôle (A)', 'Traitement (B)']
        rates = [conv_control * 100, conv_treatment * 100]
        colors = ['#6c63ff', '#00d4aa']
        bars = ax.bar(groups, rates, color=colors, width=0.5, edgecolor='none', zorder=3)
        
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{rate:.2f}%', ha='center', va='bottom',
                    color='white', fontweight='bold', fontsize=13, fontfamily='monospace')
        
        ax.set_ylabel('Taux de conversion (%)', color='#8b90a8')
        ax.set_ylim(0, max(rates) * 1.3)
        ax.tick_params(colors='#8b90a8')
        ax.spines['bottom'].set_color('#2a2d3e')
        ax.spines['left'].set_color('#2a2d3e')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, color='#2a2d3e', linestyle='--', alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        # Données tableau résumé
        summary = pd.DataFrame({
            'Groupe': ['Contrôle (A)', 'Traitement (B)'],
            'Utilisateurs': [f"{n_control:,}", f"{n_treatment:,}"],
            'Convertis': [f"{int(control['converted'].sum()):,}", f"{int(treatment['converted'].sum()):,}"],
            'Taux conv.': [f"{conv_control*100:.3f}%", f"{conv_treatment*100:.3f}%"],
        })
        if 'revenue' in df_filtered.columns:
            summary['Revenu moy.'] = [
                f"€{control['revenue'].mean():.2f}",
                f"€{treatment['revenue'].mean():.2f}"
            ]
        if 'time_on_page' in df_filtered.columns:
            summary['Tps moyen (s)'] = [
                f"{control['time_on_page'].mean():.1f}",
                f"{treatment['time_on_page'].mean():.1f}"
            ]
        st.markdown("#### Tableau récapitulatif")
        st.dataframe(summary.set_index('Groupe'), use_container_width=True)
        
        if is_significant:
            winner = 'Traitement (B)' if conv_treatment > conv_control else 'Contrôle (A)'
            st.markdown(f"""
            <div class="insight-box">
                🏆 <strong>Vainqueur détecté : {winner}</strong><br>
                Lift de <strong>{lift:+.2f}%</strong> — différence statistiquement significative (p={p_chi2:.4f} &lt; α={alpha})
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warning-box">
                ⚠️ <strong>Pas de conclusion définitive</strong><br>
                La différence observée n'est pas statistiquement significative (p={p_chi2:.4f} &gt; α={alpha}). Plus de données sont nécessaires.
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════
# TAB 2 — TESTS STATISTIQUES
# ════════════════════════════════════════════
with tabs[1]:
    st.markdown("### 🔬 Résultats des tests statistiques")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Test du Chi² (indépendance)")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Statistique χ²</div>
            <div class="metric-val">{chi2:.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">p-value</div>
            <div class="metric-val" style="color:{'#00d4aa' if p_chi2 < alpha else '#f39c12'}">{p_chi2:.6f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Degrés de liberté</div>
            <div class="metric-val">{dof}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Contingency table
        st.markdown("**Table de contingence**")
        ct = pd.crosstab(df_filtered['group'], df_filtered['converted'], margins=True)
        ct.columns = ['Non converti', 'Converti', 'Total']
        ct.index = ['Contrôle', 'Traitement', 'Total']
        st.dataframe(ct, use_container_width=True)
    
    with col2:
        st.markdown("#### Z-test sur proportions")
        ci_low = (conv_treatment - conv_control) - norm.ppf(1-alpha/2) * se
        ci_high = (conv_treatment - conv_control) + norm.ppf(1-alpha/2) * se
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Z-score</div>
            <div class="metric-val">{z_score:.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">p-value (bilatéral)</div>
            <div class="metric-val" style="color:{'#00d4aa' if p_ztest < alpha else '#f39c12'}">{p_ztest:.6f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">IC {int((1-alpha)*100)}% (différence)</div>
            <div class="metric-val" style="font-size:1.2rem">[{ci_low:.4f}, {ci_high:.4f}]</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tests métriques secondaires
    st.markdown("### Tests sur métriques secondaires")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'time_on_page' in df_filtered.columns:
            t_stat, p_ttest = ttest_ind(
                control['time_on_page'].dropna(),
                treatment['time_on_page'].dropna()
            )
            u_stat, p_mwu = mannwhitneyu(
                control['time_on_page'].dropna(),
                treatment['time_on_page'].dropna(),
                alternative='two-sided'
            )
            st.markdown("#### Temps sur la page")
            results_tbl = pd.DataFrame({
                'Test': ['t-test (Student)', 'Mann-Whitney U'],
                'Statistique': [f"{t_stat:.4f}", f"{u_stat:.0f}"],
                'p-value': [f"{p_ttest:.6f}", f"{p_mwu:.6f}"],
                'Significatif': [
                    '✅ Oui' if p_ttest < alpha else '❌ Non',
                    '✅ Oui' if p_mwu < alpha else '❌ Non'
                ]
            })
            st.dataframe(results_tbl.set_index('Test'), use_container_width=True)
    
    with col2:
        if 'clicks' in df_filtered.columns:
            t_stat2, p_ttest2 = ttest_ind(
                control['clicks'].dropna(),
                treatment['clicks'].dropna()
            )
            st.markdown("#### Nombre de clics")
            results_tbl2 = pd.DataFrame({
                'Test': ['t-test (Student)', 'Mann-Whitney U'],
                'Statistique': [f"{t_stat2:.4f}", "—"],
                'p-value': [f"{p_ttest2:.6f}", "—"],
                'Significatif': [
                    '✅ Oui' if p_ttest2 < alpha else '❌ Non',
                    '—'
                ]
            })
            st.dataframe(results_tbl2.set_index('Test'), use_container_width=True)
    
    # Interprétation des résultats
    st.markdown("---")
    st.markdown("### Interprétation")
    
    interpretation = f"""
    **H₀ (hypothèse nulle)** : Il n'y a pas de différence significative entre les taux de conversion du groupe contrôle et du groupe traitement.
    
    **H₁ (hypothèse alternative)** : Le taux de conversion du groupe traitement est différent de celui du groupe contrôle.
    
    Avec p = {p_chi2:.6f} {"< " if is_significant else "> "} α = {alpha}, nous **{'rejetons' if is_significant else 'ne pouvons pas rejeter'} H₀**.
    
    Le Z-score de {z_score:.2f} {'dépasse' if abs(z_score) > norm.ppf(1-alpha/2) else 'ne dépasse pas'} la valeur critique de ±{norm.ppf(1-alpha/2):.2f} pour un seuil α = {alpha}.
    """
    
    st.markdown(interpretation)

# ════════════════════════════════════════════
# TAB 3 — VISUALISATIONS
# ════════════════════════════════════════════
with tabs[2]:
    st.markdown("### 📈 Visualisations analytiques")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Distribution des conversions")
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor('#0d0f14')
        ax.set_facecolor('#1a1d26')
        
        groups_data = [
            ('Contrôle (A)', [conv_control, 1-conv_control], ['#6c63ff', '#2a2d3e']),
            ('Traitement (B)', [conv_treatment, 1-conv_treatment], ['#00d4aa', '#2a2d3e'])
        ]
        
        x = np.arange(2)
        width = 0.35
        bars1 = ax.bar(x - width/2, [conv_control*100, conv_treatment*100], width,
                       label='Converti', color=['#6c63ff', '#00d4aa'], zorder=3)
        bars2 = ax.bar(x + width/2, [(1-conv_control)*100, (1-conv_treatment)*100], width,
                       label='Non converti', color=['#3d3880', '#007a60'], zorder=3, alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(['Contrôle (A)', 'Traitement (B)'], color='#e8eaf0')
        ax.set_ylabel('Pourcentage (%)', color='#8b90a8')
        ax.tick_params(colors='#8b90a8')
        ax.legend(facecolor='#2a2d3e', edgecolor='none', labelcolor='#e8eaf0')
        for spine in ax.spines.values():
            spine.set_color('#2a2d3e')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, color='#2a2d3e', linestyle='--', alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        if 'time_on_page' in df_filtered.columns:
            st.markdown("#### Distribution — Temps sur la page")
            fig, ax = plt.subplots(figsize=(7, 4))
            fig.patch.set_facecolor('#0d0f14')
            ax.set_facecolor('#1a1d26')
            
            ax.hist(control['time_on_page'], bins=40, alpha=0.6, color='#6c63ff',
                    label=f'Contrôle (μ={control["time_on_page"].mean():.1f}s)', density=True, zorder=3)
            ax.hist(treatment['time_on_page'], bins=40, alpha=0.6, color='#00d4aa',
                    label=f'Traitement (μ={treatment["time_on_page"].mean():.1f}s)', density=True, zorder=3)
            
            ax.axvline(control['time_on_page'].mean(), color='#6c63ff', linestyle='--', alpha=0.8)
            ax.axvline(treatment['time_on_page'].mean(), color='#00d4aa', linestyle='--', alpha=0.8)
            
            ax.set_xlabel('Temps (secondes)', color='#8b90a8')
            ax.set_ylabel('Densité', color='#8b90a8')
            ax.tick_params(colors='#8b90a8')
            ax.legend(facecolor='#2a2d3e', edgecolor='none', labelcolor='#e8eaf0', fontsize=9)
            for spine in ax.spines.values():
                spine.set_color('#2a2d3e')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.yaxis.grid(True, color='#2a2d3e', linestyle='--', alpha=0.5, zorder=0)
            ax.set_axisbelow(True)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    # Courbe de puissance & évolution du taux de conversion
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Évolution du taux de conversion dans le temps")
        if 'date' in df_filtered.columns:
            daily = df_filtered.groupby(['date', 'group'])['converted'].mean().reset_index()
            
            fig, ax = plt.subplots(figsize=(7, 4))
            fig.patch.set_facecolor('#0d0f14')
            ax.set_facecolor('#1a1d26')
            
            for grp, color, label in [('control', '#6c63ff', 'Contrôle (A)'),
                                        ('treatment', '#00d4aa', 'Traitement (B)')]:
                data = daily[daily['group'] == grp].sort_values('date')
                ax.plot(range(len(data)), data['converted'] * 100,
                        color=color, linewidth=2, label=label, zorder=3)
                ax.fill_between(range(len(data)), data['converted'] * 100,
                                alpha=0.1, color=color)
            
            ax.set_xlabel('Jours', color='#8b90a8')
            ax.set_ylabel('Taux de conversion (%)', color='#8b90a8')
            ax.tick_params(colors='#8b90a8')
            ax.legend(facecolor='#2a2d3e', edgecolor='none', labelcolor='#e8eaf0', fontsize=9)
            for spine in ax.spines.values():
                spine.set_color('#2a2d3e')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.yaxis.grid(True, color='#2a2d3e', linestyle='--', alpha=0.5, zorder=0)
            ax.set_axisbelow(True)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    with col2:
        st.markdown("#### Distribution p-value simulée (bootstrap)")
        
        # Bootstrap distribution
        np.random.seed(42)
        n_boot = 500
        boot_diffs = []
        for _ in range(n_boot):
            s_ctrl = np.random.choice(control['converted'], size=min(500, n_control), replace=True)
            s_trt = np.random.choice(treatment['converted'], size=min(500, n_treatment), replace=True)
            boot_diffs.append(s_trt.mean() - s_ctrl.mean())
        
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor('#0d0f14')
        ax.set_facecolor('#1a1d26')
        
        ax.hist(boot_diffs, bins=40, color='#6c63ff', alpha=0.7, density=True, zorder=3)
        ax.axvline(0, color='#e74c3c', linestyle='--', linewidth=2, label='H₀: diff=0')
        ax.axvline(np.mean(boot_diffs), color='#00d4aa', linestyle='--', linewidth=2,
                   label=f'Diff observée: {np.mean(boot_diffs):.4f}')
        
        ci_lo = np.percentile(boot_diffs, alpha/2 * 100)
        ci_hi = np.percentile(boot_diffs, (1-alpha/2) * 100)
        ax.axvspan(ci_lo, ci_hi, alpha=0.1, color='#00d4aa', label=f'IC {int((1-alpha)*100)}%')
        
        ax.set_xlabel('Différence de conversion (B − A)', color='#8b90a8')
        ax.set_ylabel('Densité', color='#8b90a8')
        ax.tick_params(colors='#8b90a8')
        ax.legend(facecolor='#2a2d3e', edgecolor='none', labelcolor='#e8eaf0', fontsize=8)
        for spine in ax.spines.values():
            spine.set_color('#2a2d3e')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, color='#2a2d3e', linestyle='--', alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════
# TAB 4 — ANALYSE SEGMENTÉE
# ════════════════════════════════════════════
with tabs[3]:
    st.markdown("### 🔍 Analyse par sous-groupes (Heterogeneous Treatment Effects)")

    # Colonnes candidates : on cherche toutes les colonnes catégorielles
    # sauf les colonnes de base du test et les colonnes numériques
    EXCLUDE_COLS = {'user_id', 'timestamp', 'date', 'hour', 'week',
                    'group', 'converted', 'revenue', 'time_on_page', 'clicks'}
    LABEL_MAP = {
        'device': '📱 Device',
        'country': '🌍 Pays',
        'age_group': '👤 Tranche d\'âge',
        'landing_page': '📄 Landing page',
        'browser': '🌐 Navigateur',
        'os': '💻 OS',
    }

    # On utilise df (le dataset complet, pas df_filtered) pour la segmentation
    # afin qu'un filtre sidebar actif ne vide pas les segments
    segment_cols = [
        c for c in df.columns
        if c not in EXCLUDE_COLS
        and df[c].dtype == object
        and df[c].nunique() >= 2
        and df[c].nunique() <= 20
    ]

    if not segment_cols:
        st.markdown("""
        <div class="warning-box">
            ⚠️ <strong>Aucune colonne de segmentation disponible</strong><br>
            Le dataset chargé ne contient pas de variables catégorielles (device, pays, âge…) 
            permettant une analyse par sous-groupe.<br><br>
            Si vous utilisez le dataset Kaggle, il ne contient que <code>group</code>, 
            <code>landing_page</code> et <code>converted</code>. 
            Téléchargez le dataset enrichi ou utilisez le dataset synthétique auto-généré 
            (supprimez <code>ab_data.csv</code> pour le régénérer).
        </div>
        """, unsafe_allow_html=True)
    else:
        selected_seg = st.selectbox(
            "Segmenter par :",
            segment_cols,
            format_func=lambda x: LABEL_MAP.get(x, x.replace('_', ' ').title())
        )

        seg_results = []
        for seg_val in sorted(df[selected_seg].unique()):
            seg_df = df[df[selected_seg] == seg_val]
            seg_ctrl = seg_df[seg_df['group'] == 'control']
            seg_trt  = seg_df[seg_df['group'] == 'treatment']

            if len(seg_ctrl) < 30 or len(seg_trt) < 30:
                continue  # pas assez de données pour ce segment

            seg_conv_ctrl = seg_ctrl['converted'].mean()
            seg_conv_trt  = seg_trt['converted'].mean()
            seg_lift = (seg_conv_trt - seg_conv_ctrl) / seg_conv_ctrl * 100 if seg_conv_ctrl > 0 else 0

            ct = pd.crosstab(seg_df['group'], seg_df['converted'])
            if ct.shape == (2, 2):
                _, p_val, _, _ = chi2_contingency(ct)
            else:
                p_val = 1.0

            seg_results.append({
                'Segment':           seg_val,
                'N Contrôle':        len(seg_ctrl),
                'N Traitement':      len(seg_trt),
                'Conv. Contrôle':    f"{seg_conv_ctrl*100:.2f}%",
                'Conv. Traitement':  f"{seg_conv_trt*100:.2f}%",
                'Lift':              f"{seg_lift:+.2f}%",
                'p-value':           f"{p_val:.4f}",
                'Significatif':      '✅' if p_val < alpha else '❌',
                '_lift_val':         seg_lift,
                '_p_val':            p_val,
            })

        if not seg_results:
            st.markdown("""
            <div class="warning-box">
                ⚠️ <strong>Données insuffisantes</strong><br>
                Aucun segment ne contient au moins 30 utilisateurs par groupe 
                pour ce critère de segmentation. Essayez un autre critère ou 
                désactivez les filtres du panneau latéral.
            </div>
            """, unsafe_allow_html=True)
        else:
            seg_df_result = pd.DataFrame(seg_results)

            # ── Graphiques ──────────────────────────────────────
            fig, axes = plt.subplots(1, 2, figsize=(12, max(3, len(seg_results) * 0.7 + 1)))
            fig.patch.set_facecolor('#0d0f14')

            segments   = seg_df_result['Segment'].tolist()
            lifts      = seg_df_result['_lift_val'].tolist()
            p_vals     = seg_df_result['_p_val'].tolist()
            colors_lift = ['#00d4aa' if l > 0 else '#e74c3c' for l in lifts]
            colors_p    = ['#00d4aa' if p < alpha else '#f39c12' for p in p_vals]

            # Lift
            ax = axes[0]
            ax.set_facecolor('#1a1d26')
            ax.barh(segments, lifts, color=colors_lift, zorder=3)
            ax.axvline(0, color='#8b90a8', linewidth=1, linestyle='--')
            ax.set_xlabel('Lift (%)', color='#8b90a8')
            ax.set_title(f'Lift par {selected_seg}', color='#e8eaf0', fontfamily='monospace')
            ax.tick_params(colors='#8b90a8')
            for spine in ax.spines.values(): spine.set_color('#2a2d3e')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.xaxis.grid(True, color='#2a2d3e', linestyle='--', alpha=0.5, zorder=0)
            ax.set_axisbelow(True)

            # P-values
            ax2 = axes[1]
            ax2.set_facecolor('#1a1d26')
            ax2.barh(segments, p_vals, color=colors_p, zorder=3)
            ax2.axvline(alpha, color='#e74c3c', linewidth=2, linestyle='--', label=f'α={alpha}')
            ax2.set_xlabel('p-value', color='#8b90a8')
            ax2.set_title('p-value par segment', color='#e8eaf0', fontfamily='monospace')
            ax2.tick_params(colors='#8b90a8')
            for spine in ax2.spines.values(): spine.set_color('#2a2d3e')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.legend(facecolor='#2a2d3e', edgecolor='none', labelcolor='#e8eaf0')
            ax2.xaxis.grid(True, color='#2a2d3e', linestyle='--', alpha=0.5, zorder=0)
            ax2.set_axisbelow(True)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # ── Tableau ──────────────────────────────────────────
            display_df = seg_df_result.drop(columns=['_lift_val', '_p_val'])
            st.dataframe(display_df.set_index('Segment'), use_container_width=True)

# ════════════════════════════════════════════
# TAB 5 — DÉCISION
# ════════════════════════════════════════════
with tabs[4]:
    st.markdown("### 💡 Rapport de décision")
    
    # Calcul impact business
    total_users_monthly = st.number_input("Estimation utilisateurs/mois", value=50000, step=1000)
    avg_revenue_per_conversion = st.number_input("Revenu moyen par conversion (€)", value=45.0, step=5.0)
    
    monthly_convs_control = total_users_monthly * conv_control
    monthly_convs_treatment = total_users_monthly * conv_treatment
    monthly_revenue_control = monthly_convs_control * avg_revenue_per_conversion
    monthly_revenue_treatment = monthly_convs_treatment * avg_revenue_per_conversion
    monthly_revenue_delta = monthly_revenue_treatment - monthly_revenue_control
    
    st.markdown("#### 📊 Impact business estimé (déploiement Version B)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Conversions supplémentaires/mois</div>
            <div class="metric-val" style="color:#00d4aa">{monthly_convs_treatment - monthly_convs_control:+,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        color = "#00d4aa" if monthly_revenue_delta > 0 else "#e74c3c"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Revenu additionnel/mois</div>
            <div class="metric-val" style="color:{color}">€{monthly_revenue_delta:+,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Revenu additionnel/an</div>
            <div class="metric-val" style="color:#00d4aa">€{monthly_revenue_delta * 12:+,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### 📋 Recommandation finale")
    
    if is_significant and lift > 0:
        recommendation = "DÉPLOYER la Version B (Traitement)"
        rec_color = "#00d4aa"
        rec_icon = "✅"
        rationale = f"""
        <ul style="margin:0; padding-left:20px;">
            <li>Le test est <strong>statistiquement significatif</strong> (p={p_chi2:.4f} &lt; α={alpha})</li>
            <li>La Version B génère un <strong>lift de +{lift:.2f}%</strong> sur le taux de conversion</li>
            <li>L'intervalle de confiance à {int((1-alpha)*100)}% exclut 0 : [{ci_low:.4f}, {ci_high:.4f}]</li>
            <li>Impact business estimé : <strong>€{monthly_revenue_delta:+,.0f}/mois</strong> de revenus additionnels</li>
        </ul>
        """
    elif is_significant and lift < 0:
        recommendation = "CONSERVER la Version A (Contrôle)"
        rec_color = "#e74c3c"
        rec_icon = "🔴"
        rationale = f"""
        <ul style="margin:0; padding-left:20px;">
            <li>Le test est <strong>statistiquement significatif</strong> (p={p_chi2:.4f} &lt; α={alpha})</li>
            <li>La Version B est <strong>moins performante</strong> avec un lift de {lift:.2f}%</li>
            <li>Conserver la Version A et explorer de nouvelles hypothèses</li>
        </ul>
        """
    else:
        recommendation = "CONTINUER le test — résultat non concluant"
        rec_color = "#f39c12"
        rec_icon = "⚠️"
        rationale = f"""
        <ul style="margin:0; padding-left:20px;">
            <li>Le test <strong>n'est pas encore statistiquement significatif</strong> (p={p_chi2:.4f} &gt; α={alpha})</li>
            <li>Il est possible que la différence observée soit due au hasard</li>
            <li>Recommandation : collecter davantage de données ou revoir le design de l'expérience</li>
        </ul>
        """
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a1d26, #1e2130); 
                border: 2px solid {rec_color}; border-radius: 12px; 
                padding: 24px 28px; margin: 16px 0;">
        <div style="font-size: 1.4rem; font-weight: 700; color: {rec_color}; 
                    font-family: 'Space Mono', monospace; margin-bottom: 12px;">
            {rec_icon} {recommendation}
        </div>
        <div style="color: #e8eaf0; line-height: 1.8;">
            {rationale}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 🔜 Prochaines étapes")
    
    next_steps = [
        "**Analyse de segmentation approfondie** : identifier les sous-groupes où l'effet est le plus fort",
        "**Test de durabilité** : vérifier que l'effet ne s'estompe pas dans le temps (novelty effect)",
        "**Tests multivariés (MVT)** : tester plusieurs éléments simultanément",
        "**Analyse coût-bénéfice** : estimer le coût d'implémentation vs le gain attendu",
        "**Plan de déploiement progressif** : rollout par paliers (10% → 50% → 100%)",
    ]
    
    for step in next_steps:
        st.markdown(f"→ {step}")
    
    st.markdown("---")
    # Export
    col1, col2 = st.columns(2)
    with col1:
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="📥 Télécharger les données filtrées (CSV)",
            data=csv,
            file_name="ab_test_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        report_lines = [
            "RAPPORT A/B TEST",
            "=" * 50,
            f"Date : {pd.Timestamp.now().strftime('%Y-%m-%d')}",
            f"Total utilisateurs : {len(df_filtered):,}",
            f"",
            f"RÉSULTATS",
            f"-" * 30,
            f"Contrôle (A) : {n_control:,} users | Conv. : {conv_control*100:.3f}%",
            f"Traitement (B) : {n_treatment:,} users | Conv. : {conv_treatment*100:.3f}%",
            f"Lift : {lift:+.2f}%",
            f"",
            f"TESTS STATISTIQUES",
            f"-" * 30,
            f"Chi² : {chi2:.4f} | p-value : {p_chi2:.6f}",
            f"Z-score : {z_score:.4f} | p-value : {p_ztest:.6f}",
            f"Seuil α : {alpha}",
            f"Résultat : {'SIGNIFICATIF' if is_significant else 'NON SIGNIFICATIF'}",
            f"",
            f"RECOMMANDATION",
            f"-" * 30,
            recommendation,
        ]
        report_text = "\n".join(report_lines)
        st.download_button(
            label="📄 Télécharger le rapport (TXT)",
            data=report_text,
            file_name="rapport_ab_test.txt",
            mime="text/plain",
            use_container_width=True
        )
