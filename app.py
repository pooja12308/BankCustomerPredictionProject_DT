import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Bank Marketing Prediction App",
    layout="wide"
)

st.title("🏦 Bank Marketing – Term Deposit Prediction")
st.markdown("Predict whether a customer is **likely to subscribe** to a term deposit.")

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("bank_marketing_dataset.csv")

df = load_data()

# --------------------------------------------------
# Show Dataset
# --------------------------------------------------
with st.expander("📊 View Dataset"):
    st.dataframe(df)
    st.write("Shape:", df.shape)

# --------------------------------------------------
# Prepare Data
# --------------------------------------------------
X = df.drop("deposit", axis=1)
y = df["deposit"].map({"yes": 1, "no": 0})

cat_cols = X.select_dtypes(include="object").columns
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# Train Model
# --------------------------------------------------
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

# --------------------------------------------------
# Model Evaluation
# --------------------------------------------------
st.header("📈 Model Performance")

y_pred = model.predict(X_test)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Accuracy")
    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")

with col2:
    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

# --------------------------------------------------
# Decision Tree Visualization
# --------------------------------------------------
st.header("🌳 Decision Tree Visualization")

fig, ax = plt.subplots(figsize=(22, 10))
plot_tree(
    model,
    feature_names=X_encoded.columns,
    class_names=["Not Subscribe", "Subscribe"],
    filled=True,
    ax=ax
)
st.pyplot(fig)

# --------------------------------------------------
# Feature Importance
# --------------------------------------------------
st.header("⭐ Feature Importance")

importance = pd.Series(
    model.feature_importances_,
    index=X_encoded.columns
).sort_values(ascending=False)

st.bar_chart(importance.head(10))

# --------------------------------------------------
# Prediction Section
# --------------------------------------------------
st.header("🔮 Predict for a New Customer")

input_data = {}

for col in X.columns:
    if X[col].dtype == "object":
        input_data[col] = st.selectbox(
            f"{col}",
            options=sorted(X[col].unique())
        )
    else:
        input_data[col] = st.number_input(
            f"{col}",
            value=float(X[col].median())
        )

input_df = pd.DataFrame([input_data])
input_encoded = pd.get_dummies(input_df)

# Align columns with training data
input_encoded = input_encoded.reindex(
    columns=X_encoded.columns,
    fill_value=0
)

if st.button("Predict"):
    prediction = model.predict(input_encoded)[0]
    if prediction == 1:
        st.success("✅ Likely to Subscribe")
    else:
        st.error("❌ Not Likely to Subscribe")
