import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MovieIQ – Film Success Analytics",
    page_icon="🎬",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-val { font-size: 2rem; font-weight: 700; color: #e94560; }
    .metric-lbl { font-size: 0.8rem; color: #a0aec0; letter-spacing: 0.05em; text-transform: uppercase; }
    h1, h2, h3 { color: #e2e8f0 !important; }
    .stTabs [data-baseweb="tab"] { font-size: 0.9rem; }
    .insight-box {
        background: #0d1117;
        border-left: 4px solid #e94560;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.4rem 0;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
PALETTE = {"Successful": "#2ecc71", "Unsuccessful": "#e74c3c"}
PLT_STYLE = "dark_background"

def fmt_millions(x, _): return f"${x/1e6:.0f}M"

def load_data(uploaded):
    df = pd.read_csv(uploaded)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Normalise column names
    rename = {}
    for c in df.columns:
        if "budget" in c:       rename[c] = "budget"
        if "revenue" in c:      rename[c] = "revenue"
        if "popularity" in c:   rename[c] = "popularity"
        if "runtime" in c:      rename[c] = "runtime"
        if "vote_average" in c or "rating" in c: rename[c] = "vote_average"
        if "vote_count" in c:   rename[c] = "vote_count"
        if "genre" in c:        rename[c] = "main_genre"
        if "title" in c:        rename[c] = "title"
        if "release" in c or "year" in c: rename[c] = "release_year"
    df.rename(columns=rename, inplace=True)

    # Derived columns
    if "budget" in df.columns and "revenue" in df.columns:
        df["budget"]  = pd.to_numeric(df["budget"],  errors="coerce").fillna(0)
        df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0)
        df = df[(df["budget"] > 0) & (df["revenue"] > 0)]
        df["roi"]     = (df["revenue"] - df["budget"]) / df["budget"] * 100
        df["profit"]  = df["revenue"] - df["budget"]
        df["success"] = (df["revenue"] > df["budget"]).astype(int)
        df["success_label"] = df["success"].map({1: "Successful", 0: "Unsuccessful"})

    if "release_year" in df.columns:
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
        df = df[df["release_year"].between(1950, 2030, inclusive="both")]

    if "main_genre" in df.columns:
        df["main_genre"] = df["main_genre"].astype(str).str.strip().str.title()

    return df

# ── Sidebar ───────────────────────────────────────────────────────────────────
df_raw = None
sel_genre = []
rating_range = (0.0, 10.0)
year_range = (1950, 2024)

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/clapperboard.png", width=60)
    st.title("MovieIQ")
    st.caption("Film Success Analytics Dashboard")
    st.divider()

    uploaded = st.file_uploader("Upload movies CSV", type=["csv"])

    if uploaded:
        df_raw = load_data(uploaded)

        genres = ["All"] + sorted(df_raw["main_genre"].dropna().unique().tolist()) \
            if "main_genre" in df_raw.columns else ["All"]
        sel_genre = st.multiselect("Genre Filter", genres[1:], default=[])

        min_rating, max_rating = 0.0, 10.0
        if "vote_average" in df_raw.columns:
            min_rating = float(df_raw["vote_average"].min())
            max_rating = float(df_raw["vote_average"].max())
        rating_range = st.slider("Vote Average Range", min_rating, max_rating,
                                 (min_rating, max_rating), 0.1)

        year_min, year_max = 1950, 2024
        if "release_year" in df_raw.columns:
            year_min = int(df_raw["release_year"].min())
            year_max = int(df_raw["release_year"].max())
        year_range = st.slider("Release Year Range", year_min, year_max,
                               (year_min, year_max))

        st.divider()
        st.caption("Built with Streamlit · MovieIQ v2.0")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🎬 MovieIQ — Film Success Analytics")
st.markdown("Explore what drives box-office success through **EDA, statistical tests, and revenue forecasting**.")

if not uploaded:
    st.info("👈 Upload your `movies.csv` file from the sidebar to get started.")
    st.stop()

# Apply filters
df = df_raw.copy()
if sel_genre:
    df = df[df["main_genre"].isin(sel_genre)]
if "vote_average" in df.columns:
    df = df[df["vote_average"].between(*rating_range)]
if "release_year" in df.columns:
    df = df[df["release_year"].between(*year_range)]

if df.empty:
    st.warning("No data matches your filters. Adjust the sidebar options.")
    st.stop()

# ── KPI Row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
kpis = [
    ("Total Movies",    f"{len(df):,}"),
    ("Success Rate",    f"{df['success'].mean()*100:.1f}%" if "success" in df.columns else "—"),
    ("Avg ROI",         f"{df['roi'].mean():.1f}%" if "roi" in df.columns else "—"),
    ("Avg Revenue",     f"${df['revenue'].mean()/1e6:.1f}M" if "revenue" in df.columns else "—"),
    ("Unique Genres",   f"{df['main_genre'].nunique()}" if "main_genre" in df.columns else "—"),
]
for col, (lbl, val) in zip([k1,k2,k3,k4,k5], kpis):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{val}</div>
        <div class="metric-lbl">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs(["📊 Overview", "🔍 EDA", "📈 Forecasting", "🧪 Statistics", "💡 Insights"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 – OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Dataset Overview")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Sample Data**")
        st.dataframe(df.head(10), use_container_width=True)
    with c2:
        st.markdown("**Descriptive Statistics**")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        st.dataframe(df[num_cols].describe().T.round(2), use_container_width=True)

    # Missing values
    st.markdown("**Data Quality — Missing Values**")
    miss = df.isnull().sum().reset_index()
    miss.columns = ["Column", "Missing"]
    miss["% Missing"] = (miss["Missing"] / len(df) * 100).round(2)
    miss = miss[miss["Missing"] > 0]
    if miss.empty:
        st.success("✅ No missing values in filtered dataset.")
    else:
        st.dataframe(miss, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 – EDA
# ════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Exploratory Data Analysis")
    plt.style.use(PLT_STYLE)

    # Row 1
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("**Budget vs Revenue**")
        fig, ax = plt.subplots(figsize=(6, 4))
        if "success_label" in df.columns:
            for label, grp in df.groupby("success_label"):
                ax.scatter(grp["budget"]/1e6, grp["revenue"]/1e6,
                           alpha=0.5, s=18, label=label,
                           color=PALETTE.get(label, "#aaa"))
            ax.legend(fontsize=8)
        else:
            ax.scatter(df["budget"]/1e6, df["revenue"]/1e6, alpha=0.5, s=18, color="#e94560")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:.0f}M"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:.0f}M"))
        ax.set_xlabel("Budget"); ax.set_ylabel("Revenue")
        ax.set_title("Budget vs Revenue")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with r1c2:
        st.markdown("**Success Rate by Genre**")
        if "main_genre" in df.columns and "success" in df.columns:
            genre_s = df.groupby("main_genre")["success"].mean().sort_values(ascending=False).head(12)
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.barh(genre_s.index, genre_s.values * 100,
                           color=plt.cm.RdYlGn(genre_s.values))
            ax.set_xlabel("Success Rate (%)")
            ax.set_title("Success Rate by Genre")
            for bar, val in zip(bars, genre_s.values):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        f"{val*100:.1f}%", va="center", fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    # Row 2
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown("**ROI Distribution**")
        if "roi" in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            clipped = df["roi"].clip(-200, 500)
            ax.hist(clipped, bins=40, color="#e94560", edgecolor="none", alpha=0.85)
            ax.axvline(0, color="white", linestyle="--", linewidth=1, label="Break-even")
            ax.axvline(df["roi"].mean(), color="#f39c12", linestyle="--",
                       linewidth=1, label=f"Mean ROI {df['roi'].mean():.0f}%")
            ax.set_xlabel("ROI (%)"); ax.set_ylabel("Count")
            ax.set_title("ROI Distribution")
            ax.legend(fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with r2c2:
        st.markdown("**Vote Average by Outcome**")
        if "vote_average" in df.columns and "success_label" in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            for label, grp in df.groupby("success_label"):
                ax.hist(grp["vote_average"], bins=25, alpha=0.65,
                        label=label, color=PALETTE.get(label, "#aaa"))
            ax.set_xlabel("Vote Average"); ax.set_ylabel("Count")
            ax.set_title("Rating Distribution by Outcome")
            ax.legend(fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    # Row 3
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        st.markdown("**Correlation Heatmap**")
        num_df = df[["budget","revenue","roi","vote_average","popularity","runtime"]
                    ].dropna() if all(c in df.columns for c in ["budget","revenue","roi"]) \
                    else df.select_dtypes(include=np.number).dropna()
        if not num_df.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                        linewidths=0.5, ax=ax, annot_kws={"size": 8})
            ax.set_title("Feature Correlation Matrix")
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with r3c2:
        st.markdown("**Outlier Detection (ROI)**")
        if "roi" in df.columns and "title" in df.columns:
            q1, q3 = df["roi"].quantile(0.25), df["roi"].quantile(0.75)
            iqr = q3 - q1
            outliers = df[(df["roi"] < q1 - 1.5*iqr) | (df["roi"] > q3 + 1.5*iqr)]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(df.index, df["roi"], s=10, alpha=0.4, color="#7f8c8d", label="Normal")
            ax.scatter(outliers.index, outliers["roi"], s=25, color="#e94560",
                       alpha=0.9, label=f"Outliers ({len(outliers)})")
            ax.axhline(q3 + 1.5*iqr, color="#f39c12", linestyle="--", linewidth=0.8)
            ax.axhline(q1 - 1.5*iqr, color="#f39c12", linestyle="--", linewidth=0.8)
            ax.set_xlabel("Movie Index"); ax.set_ylabel("ROI (%)")
            ax.set_title("ROI Outlier Detection (IQR Method)")
            ax.legend(fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.caption(f"Found **{len(outliers)} outliers** out of {len(df)} movies "
                       f"({len(outliers)/len(df)*100:.1f}%)")

    # Genre trend over time
    if "release_year" in df.columns and "main_genre" in df.columns and "revenue" in df.columns:
        st.markdown("**Genre Revenue Trend Over Time**")
        top_genres = df.groupby("main_genre")["revenue"].sum().nlargest(5).index.tolist()
        trend = df[df["main_genre"].isin(top_genres)].groupby(
            ["release_year","main_genre"])["revenue"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(12, 4))
        colors = ["#e94560","#2ecc71","#3498db","#f39c12","#9b59b6"]
        for i, g in enumerate(top_genres):
            sub = trend[trend["main_genre"]==g]
            ax.plot(sub["release_year"], sub["revenue"]/1e6,
                    label=g, color=colors[i % len(colors)], linewidth=2)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:.0f}M"))
        ax.set_xlabel("Year"); ax.set_ylabel("Avg Revenue")
        ax.set_title("Average Revenue by Genre Over Time (Top 5)")
        ax.legend(fontsize=8, loc="upper left")
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 – FORECASTING
# ════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("📈 Revenue Forecasting")
    st.markdown("Using **Linear Regression** to model historical revenue trends and project the next 5 years.")

    if "release_year" not in df.columns or "revenue" not in df.columns:
        st.warning("Need `release_year` and `revenue` columns for forecasting.")
    else:
        yearly = df.groupby("release_year").agg(
            avg_revenue=("revenue","mean"),
            total_revenue=("revenue","sum"),
            movie_count=("revenue","count")
        ).reset_index()
        yearly = yearly[yearly["movie_count"] >= 3]  # at least 3 movies per year

        X = yearly["release_year"].values.reshape(-1,1)
        y_avg = yearly["avg_revenue"].values
        y_total = yearly["total_revenue"].values

        model_avg = LinearRegression().fit(X, y_avg)
        model_total = LinearRegression().fit(X, y_total)

        last_year = int(yearly["release_year"].max())
        future_years = np.arange(last_year + 1, last_year + 6).reshape(-1,1)
        all_years = np.vstack([X, future_years])

        pred_avg   = model_avg.predict(all_years)
        pred_total = model_total.predict(all_years)

        # Confidence interval (±1 std of residuals)
        resid = y_avg - model_avg.predict(X)
        ci = resid.std() * 1.96

        fc1, fc2 = st.columns(2)

        with fc1:
            st.markdown("**Average Revenue Forecast (per movie)**")
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(yearly["release_year"], yearly["avg_revenue"]/1e6,
                    color="#3498db", linewidth=2, label="Historical Avg Revenue")
            hist_x = X.flatten()
            fut_x  = future_years.flatten()
            all_x  = all_years.flatten()
            ax.plot(all_x, pred_avg/1e6, "--", color="#e94560", linewidth=1.5, label="Trend Line")
            ax.fill_between(all_x,
                            (pred_avg - ci)/1e6,
                            (pred_avg + ci)/1e6,
                            alpha=0.15, color="#e94560", label="95% CI")
            ax.axvspan(last_year, fut_x[-1], alpha=0.05, color="#f39c12")
            for yr, val in zip(fut_x, model_avg.predict(future_years)):
                ax.annotate(f"${val/1e6:.0f}M", (yr, val/1e6),
                            textcoords="offset points", xytext=(0,6),
                            fontsize=7, color="#f39c12", ha="center")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:.0f}M"))
            ax.set_xlabel("Year"); ax.set_ylabel("Avg Revenue")
            ax.set_title("Avg Revenue per Movie — Forecast")
            ax.legend(fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with fc2:
            st.markdown("**Forecast Summary Table**")
            fc_df = pd.DataFrame({
                "Year": fut_x,
                "Predicted Avg Revenue": [f"${v/1e6:.1f}M" for v in model_avg.predict(future_years)],
                "Predicted Total Revenue": [f"${v/1e6:.0f}M" for v in model_total.predict(future_years)],
            })
            st.dataframe(fc_df, use_container_width=True, hide_index=True)

            r2 = model_avg.score(X, y_avg)
            slope = model_avg.coef_[0]
            st.markdown(f"""
            <div class="insight-box">📐 <b>Model R² = {r2:.3f}</b> — explains {r2*100:.1f}% of revenue variance</div>
            <div class="insight-box">📉 <b>Trend slope:</b> ${slope/1e6:.2f}M per year change in avg revenue</div>
            <div class="insight-box">⚠️ Forecast assumes <b>linear continuation</b> of historical patterns</div>
            """, unsafe_allow_html=True)

        # Success rate forecast
        st.markdown("---")
        st.markdown("**Success Rate Trend & Forecast**")
        if "success" in df.columns:
            sr_yearly = df.groupby("release_year")["success"].mean().reset_index()
            sr_yearly = sr_yearly[sr_yearly["release_year"].isin(yearly["release_year"])]
            Xs = sr_yearly["release_year"].values.reshape(-1,1)
            ys = sr_yearly["success"].values
            model_sr = LinearRegression().fit(Xs, ys)
            pred_sr = model_sr.predict(np.vstack([Xs, future_years]))

            fig, ax = plt.subplots(figsize=(12, 3.5))
            ax.plot(sr_yearly["release_year"], ys*100,
                    color="#2ecc71", linewidth=2, label="Historical Success Rate")
            ax.plot(np.vstack([Xs, future_years]).flatten(), pred_sr*100,
                    "--", color="#e94560", linewidth=1.5, label="Forecast")
            ax.axvspan(last_year, future_years[-1][0], alpha=0.05, color="#f39c12")
            ax.set_ylabel("Success Rate (%)")
            ax.set_xlabel("Year")
            ax.set_title("Industry Success Rate — Historical & Forecast")
            ax.legend(fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 – STATISTICS
# ════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("🧪 Statistical Tests")
    st.markdown("Formal hypothesis tests to validate observations from the EDA.")

    if "success" in df.columns:
        success_grp    = df[df["success"] == 1]
        no_success_grp = df[df["success"] == 0]

        tests = []

        # T-Test: Budget
        if "budget" in df.columns and len(success_grp) > 1 and len(no_success_grp) > 1:
            t, p = stats.ttest_ind(success_grp["budget"].dropna(),
                                   no_success_grp["budget"].dropna())
            tests.append(("T-Test", "Budget: Success vs Unsuccessful",
                           f"t = {t:.3f}", f"p = {p:.4f}",
                           "✅ Significant" if p < 0.05 else "❌ Not Significant"))

        # T-Test: Vote Average
        if "vote_average" in df.columns and len(success_grp) > 1:
            t2, p2 = stats.ttest_ind(success_grp["vote_average"].dropna(),
                                     no_success_grp["vote_average"].dropna())
            tests.append(("T-Test", "Vote Average: Success vs Unsuccessful",
                           f"t = {t2:.3f}", f"p = {p2:.4f}",
                           "✅ Significant" if p2 < 0.05 else "❌ Not Significant"))

        # T-Test: ROI
        if "roi" in df.columns:
            t3, p3 = stats.ttest_ind(success_grp["roi"].dropna(),
                                     no_success_grp["roi"].dropna())
            tests.append(("T-Test", "ROI: Success vs Unsuccessful",
                           f"t = {t3:.3f}", f"p = {p3:.4f}",
                           "✅ Significant" if p3 < 0.05 else "❌ Not Significant"))

        # Chi-Square: Genre vs Success
        if "main_genre" in df.columns:
            ct = pd.crosstab(df["main_genre"], df["success"])
            if ct.shape[0] > 1 and ct.shape[1] > 1:
                chi2, pchi, _, _ = stats.chi2_contingency(ct)
                tests.append(("Chi-Square", "Genre vs Success",
                               f"χ² = {chi2:.3f}", f"p = {pchi:.4f}",
                               "✅ Significant" if pchi < 0.05 else "❌ Not Significant"))

        # Pearson: Budget vs Revenue
        if "budget" in df.columns and "revenue" in df.columns:
            r, pr = stats.pearsonr(df["budget"].dropna(), df["revenue"].dropna())
            tests.append(("Pearson r", "Budget ↔ Revenue",
                           f"r = {r:.3f}", f"p = {pr:.4f}",
                           "✅ Significant" if pr < 0.05 else "❌ Not Significant"))

        # Pearson: Vote Average vs ROI
        if "vote_average" in df.columns and "roi" in df.columns:
            r2, pr2 = stats.pearsonr(df[["vote_average","roi"]].dropna()["vote_average"],
                                     df[["vote_average","roi"]].dropna()["roi"])
            tests.append(("Pearson r", "Vote Average ↔ ROI",
                           f"r = {r2:.3f}", f"p = {pr2:.4f}",
                           "✅ Significant" if pr2 < 0.05 else "❌ Not Significant"))

        results_df = pd.DataFrame(tests, columns=["Test", "Variables", "Statistic", "p-value", "Result"])
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        st.markdown("""
        > **Interpretation guide:** p < 0.05 means the result is statistically significant 
        > (less than 5% probability it occurred by chance). Pearson r closer to ±1 = stronger correlation.
        """)

        # Visual: correlation bar chart
        if "budget" in df.columns and "revenue" in df.columns and "roi" in df.columns:
            st.markdown("**Feature Correlation with Revenue**")
            num_cols = df.select_dtypes(include=np.number).drop(
                columns=["success"], errors="ignore").columns
            corr_with_rev = df[num_cols].corr()["revenue"].drop("revenue").sort_values()
            fig, ax = plt.subplots(figsize=(8, 3))
            colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in corr_with_rev.values]
            ax.barh(corr_with_rev.index, corr_with_rev.values, color=colors)
            ax.axvline(0, color="white", linewidth=0.8)
            ax.set_xlabel("Pearson Correlation with Revenue")
            ax.set_title("What Correlates Most with Revenue?")
            plt.tight_layout(); st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 – INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("💡 Key Insights")

    ins = []

    if "main_genre" in df.columns and "success" in df.columns:
        top_genre = df.groupby("main_genre")["success"].mean().idxmax()
        top_rate  = df.groupby("main_genre")["success"].mean().max() * 100
        ins.append(f"🎭 <b>Top Genre:</b> <b>{top_genre}</b> leads with a {top_rate:.1f}% success rate.")

    if "roi" in df.columns:
        avg_roi = df["roi"].mean()
        ins.append(f"💰 <b>Average ROI:</b> Movies in this dataset return an average of <b>{avg_roi:.1f}%</b> on investment.")

    if "budget" in df.columns and "revenue" in df.columns:
        corr_br = df["budget"].corr(df["revenue"])
        ins.append(f"📊 <b>Budget–Revenue Correlation:</b> r = {corr_br:.2f} — {'strong' if abs(corr_br)>0.6 else 'moderate'} positive relationship.")

    if "vote_average" in df.columns and "success" in df.columns:
        avg_s  = df[df["success"]==1]["vote_average"].mean()
        avg_ns = df[df["success"]==0]["vote_average"].mean()
        ins.append(f"⭐ <b>Ratings & Success:</b> Successful movies average <b>{avg_s:.2f}</b> vs <b>{avg_ns:.2f}</b> for unsuccessful ones.")

    if "roi" in df.columns:
        q1, q3 = df["roi"].quantile(0.25), df["roi"].quantile(0.75)
        iqr = q3 - q1
        n_out = len(df[(df["roi"] < q1 - 1.5*iqr) | (df["roi"] > q3 + 1.5*iqr)])
        ins.append(f"🔍 <b>Outliers:</b> {n_out} movies ({n_out/len(df)*100:.1f}%) have extreme ROI values outside the normal range.")

    if "release_year" in df.columns and "revenue" in df.columns:
        recent = df[df["release_year"] >= df["release_year"].quantile(0.75)]
        older  = df[df["release_year"] <  df["release_year"].quantile(0.75)]
        if len(recent) > 0 and len(older) > 0:
            diff = (recent["revenue"].mean() - older["revenue"].mean()) / 1e6
            direction = "higher" if diff > 0 else "lower"
            ins.append(f"📅 <b>Recent vs Older:</b> Recent movies earn <b>${abs(diff):.0f}M {direction}</b> on average than older films.")

    for i in ins:
        st.markdown(f'<div class="insight-box">{i}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Download Filtered Data")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Filtered CSV", data=csv,
                       file_name="movieiq_filtered.csv", mime="text/csv")
