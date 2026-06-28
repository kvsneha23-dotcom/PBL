# India Digital Labour Market Intelligence Platform
## Startup Validation — Analytics Dashboard

A comprehensive Streamlit analytics dashboard built to validate a business idea:
**a Digital Labour Market Intelligence Platform targeting India's informal F&B sector.**

---

## Dashboard Tabs

| Tab | Contents |
|-----|----------|
| 📊 Descriptive Analytics | Frequency distributions, cross-tabulations, WTP analysis, geographic breakdown |
| 📈 EDA | Histograms, box plots, violin plots, scatter plots, heatmaps, pair plots, pain-point analysis |
| 🔍 Diagnostic Analytics | Adoption drivers, business size analysis, hiring channels, barrier analysis, price sensitivity |
| ⚙️ Feature Engineering | 6 composite business scores, MI-based feature selection, ML feature matrix |
| 🤖 ML Models | KNN, Decision Tree, Random Forest, Gradient Boost — accuracy, F1, ROC, confusion matrices |
| 📋 Findings | Market validation summary, stakeholder recommendations, platform roadmap |

---

## Composite Business Features Engineered

| Score | Description |
|-------|-------------|
| **DRI** | Digital Readiness Index — platform adoption capacity |
| **HDS** | Hiring Difficulty Score — employer pain intensity (employer only) |
| **TS** | Trust Score — institutional trust level |
| **LMFS** | Labour Market Friction Score — market inefficiency measure |
| **EAS** | Employer Adoption Score — employer predisposition (employer only) |
| **WES** | Worker Employability Score — worker fit for platform (worker only) |

---

## ML Models

- **KNN** — K=7 neighbours
- **Decision Tree** — max_depth=6
- **Random Forest** — 150 estimators, max_depth=8
- **Gradient Boosting** — 150 estimators, lr=0.1, max_depth=4

**Target:** `adoption_intent = (adopt_likelihood >= 4).astype(int)`  
**Split:** 75% train / 25% test, stratified, 5-fold CV

---

## Local Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/india-labour-platform-validation.git
cd india-labour-platform-validation

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

> **Note:** The cleaned dataset (`data/india_fb_platform_cleaned.csv`) is included.  
> You can also upload your own CSV via the sidebar file uploader.

---

## Deploy to Streamlit Cloud

1. Push this repository to GitHub (public repo)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy**

The app will be live at: `https://YOUR_USERNAME-india-labour-platform.streamlit.app`

---

## Dataset

- **N = 1,000** synthetic respondents mimicking real survey responses
- **4 stakeholder types:** Employer (280), Worker (450), Contractor (180), Govt (90)
- **15 Indian states** represented
- **161 features** (post-cleaning)
- Structural NaN = inapplicable cross-survey columns (not true missing data)

---

## Project Context

**Business Idea:** Digital Labour Market Intelligence Platform for India's informal sector  
**Sector:** Food & Beverage (F&B), Hospitality, Quick-Service Restaurants  
**Problem:** Severe hiring inefficiencies, lack of verified worker profiles, agent exploitation, no real-time labour data  
**Solution:** AI-powered matching platform with govt verification, digital payment tracking, and market analytics  
**Market:** ~2.8M MSE employers + 450M informal workers in India

---

## Survey Sections Covered

1. Demographics & Business Profile
2. Digital Readiness Assessment
3. Current Hiring Practices (Employers)
4. Labour Challenges & Pain Points
5. Platform Feature Demand
6. Willingness to Pay
7. Adoption Readiness & Timeline
8. Government Data Integration
9. Contractor Business Intelligence
10. Trust & Data Privacy (DPDPA 2023)

---

*Built with Streamlit, Plotly, Scikit-learn, Pandas | Synthetic data generated for startup validation*
