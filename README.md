# 🎬 MovieIQ — Predictive Analytics on Film Success

> A data-driven project that analyzes and predicts movie success using machine learning and exploratory data analysis.

---

## 📁 Project Structure

```
MovieIQ-Predictive-Analytics-on-Film-Success/
│
├── assets/                    # Images, plots, and static resources
├── app.py                     # Main application script (Streamlit / Flask)
├── MovieIQ_Analysis.ipynb     # Jupyter Notebook — EDA & ML pipeline
├── movies.csv                 # Raw movie dataset
├── movies_cleaned.csv         # Preprocessed / cleaned dataset
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 🧠 Project Overview

**MovieIQ** explores what makes a movie successful by analyzing a dataset of films across various features — budget, popularity, runtime, vote average, and ROI. The project includes:

- 📊 **Exploratory Data Analysis (EDA)** — correlation heatmaps, feature distributions, and trend analysis
- 🤖 **Predictive Modeling** — ML models to forecast movie success
- 📈 **Key Insight** — `ROI` has the highest absolute correlation (r = 0.693) with movie success, far ahead of popularity, runtime, budget, and vote average

---

## 🔍 Key Features

| Feature        | Correlation with Success (r) |
|----------------|------------------------------|
| ROI            | 0.693                        |
| Popularity     | 0.046                        |
| Runtime        | 0.035                        |
| Budget         | 0.032                        |
| Vote Average   | 0.023                        |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/MovieIQ-Predictive-Analytics-on-Film-Success.git
cd MovieIQ-Predictive-Analytics-on-Film-Success
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

Open `MovieIQ_Analysis.ipynb` in Jupyter or VS Code to explore the analysis.

### 4. Launch the app

```bash
python app.py
```

---

## 🛠️ Tech Stack

- **Python 3.14**
- **Pandas** — data manipulation
- **Matplotlib / Seaborn** — data visualization
- **Scikit-learn** — machine learning
- **Jupyter Notebook** — analysis & EDA
- **Streamlit / Flask** — web application (`app.py`)

---

## 📌 Future Improvements

- [ ] Add genre-based success segmentation
- [ ] Incorporate NLP on movie overviews/reviews
- [ ] Deploy app to Streamlit Cloud or Render
- [ ] Add model comparison dashboard

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

> Made with ❤️ and data
