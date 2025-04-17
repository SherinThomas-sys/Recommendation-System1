

!pip install -q streamlit pyngrok faiss-cpu sentence-transformers requests pillow pandas numpy torch

import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from google.colab import files
from pyngrok import ngrok
import re

print("Upload cleaned_main_product_data.csv")
uploaded_main = files.upload()
main_file = next(iter(uploaded_main))
df = pd.read_csv(main_file)

print("Upload cleaned_all_accessories.csv")
uploaded_acc = files.upload()
acc_file = next(iter(uploaded_acc))
df1 = pd.read_csv(acc_file)

df['model'].head(10)

def extract_from_model(model_name):
    if pd.isna(model_name):
        return pd.Series([None, None, None, None, None])

    model_name = str(model_name).strip().title()

    category = None
    variant = None
    series = None
    model_number = None
    device = None

    # Apple
    if 'Iphone' in model_name:
        category = 'Smartphone'
        device = 'iPhone'
        if 'Pro Max' in model_name:
            variant = 'Pro Max'
        elif 'Pro' in model_name:
            variant = 'Pro'
        elif 'Plus' in model_name:
            variant = 'Plus'
        elif 'E' in model_name:
            variant = 'e'

        match = re.search(r'Iphone\s*(\d+)', model_name)
        if match:
            model_number = match.group(1)

    elif 'Macbook' in model_name:
        category = 'Laptop'
        device = 'MacBook'
        variant = 'Air' if 'Air' in model_name else 'Pro' if 'Pro' in model_name else None

    # Samsung
    elif 'Samsung' in model_name:
        category = 'Smartphone'
        device = 'Samsung'
        if 'Book' in model_name:
            category = 'Laptop'
            device = 'Samsung Book'
            variant_match = re.search(r'Book\s*(\d+)', model_name)
            if variant_match:
                variant = f"Book{variant_match.group(1)}"
        match = re.search(r'Samsung\s*([A-Z])(\d+)', model_name)
        if match:
            series = match.group(1)
            model_number = match.group(2)

    return pd.Series([category, variant, series, model_number, device])

# Clean model column
df['model'] = df['model'].astype(str).str.strip().str.title()

# Apply updated function
df[['category_extracted', 'variant_extracted', 'series_extracted', 'model_number_extracted', 'device_extracted']] = df['model'].apply(extract_from_model)

# Check result
df[['model', 'category_extracted', 'variant_extracted', 'series_extracted', 'model_number_extracted', 'device_extracted']].head(10)

print("Upload extracted.csv")
uploaded_ext= files.upload()
ext_file = next(iter(uploaded_ext))
df2 = pd.read_csv(ext_file)

df.to_csv('extracted_main_product_data.csv', index=False)
files.download('extracted_main_product_data.csv')

def extract_specs(title):
    ram_match = re.search(r'(\d+)\s*GB\s*RAM', title, re.IGNORECASE)
    storage_match = re.search(r'(\d+)\s*GB(?!\s*RAM)', title, re.IGNORECASE)
    ram = int(ram_match.group(1)) if ram_match else 0
    storage = int(storage_match.group(1)) if storage_match else 0
    return ram, storage

# Apply and create new DataFrame
df[['RAM_GB', 'Storage_GB']] = df['product_title'].apply(lambda x: pd.Series(extract_specs(str(x))))

# Save to CSV
df[['product_title', 'RAM_GB', 'Storage_GB']].to_csv("spec_extraction.csv", index=False)

# Download the CSV file to your local machine
files.download("spec_extraction.csv")

print("Upload extracted.csv")
uploaded_spec= files.upload()
spec_ext = next(iter(uploaded_spec))
df3 = pd.read_csv(spec_ext)

with open("app.py", "w") as f:
    f.write("""
import streamlit as st
import pandas as pd
import random

# Load datasets
df = pd.read_csv("cleaned_main_product_data.csv")
df1 = pd.read_csv("cleaned_all_accessories.csv")
df2 = pd.read_csv("extracted_main_product_data.csv")
df3 = pd.read_csv("spec_extraction.csv")

# Merge feature-extracted and spec-extracted data
df_full = df.merge(df2, on=['product_id', 'category', 'product_title', 'brand', 'model', 'features', 'price', 'rating', 'review_count', 'image_url'], how='left')
df_full = df_full.merge(df3, on='product_title', how='left')

# Fill missing values for comparison
for col in ['variant_extracted', 'series_extracted', 'device_extracted']:
    df_full[col] = df_full[col].fillna('')
df_full['RAM_GB'] = df_full['RAM_GB'].fillna(0)
df_full['Storage_GB'] = df_full['Storage_GB'].fillna(0)

# Streamlit UI
st.title("✨RECOMMENDATION SYSTEM")

category_option = st.selectbox("Choose your category 🛍️", df_full['category'].dropna().unique())
model_option = st.selectbox("Choose your model 📱", df_full[df_full['category'] == category_option]['model'].dropna().unique())

def display_product(prod, title):
    st.subheader(f"🌟 {title} 🌟")
    # Two-column layout: left for image, right for details
    cols = st.columns([1, 3])
    with cols[0]:
        st.image(prod['image_url'], width=150)
    with cols[1]:
        # Display details as per format: bullet title, price | rating, then brand and model.
        st.markdown(f"🔹 **{prod['product_title']}**")
        st.markdown(f"💰 ₹{prod['price']}  |  ⭐ {prod['rating']}")
        st.markdown(f"🔍 Brand: {prod['brand']}  |  📦 Model: {prod['model']}")
        st.markdown(f"**Category:** {prod['category']}")

if st.button("Get Recommendations 🔍"):
    selected_product = df_full[(df_full['category'] == category_option) & (df_full['model'] == model_option)].iloc[0]
    display_product(selected_product, "Selected Product")

    # Filter for similar specs
    similar = df_full[
        (df_full['category'] == selected_product['category']) &
        (df_full['variant_extracted'] == selected_product['variant_extracted']) &
        (df_full['series_extracted'] == selected_product['series_extracted']) &
        (df_full['device_extracted'] == selected_product['device_extracted']) &
        (df_full['model'] != selected_product['model'])
    ]

    # Better alternative
    def get_better_alternative():
        better = similar[
            ((similar['RAM_GB'] > selected_product['RAM_GB']) | (similar['Storage_GB'] > selected_product['Storage_GB'])) &
            (similar['rating'] >= selected_product['rating']) &
            (similar['price'] <= selected_product['price'])
        ]
        if better.empty:
            better = similar[
                (similar['RAM_GB'] == selected_product['RAM_GB']) &
                (similar['Storage_GB'] == selected_product['Storage_GB']) &
                ((similar['rating'] > selected_product['rating']) | (similar['price'] < selected_product['price']))
            ]
        return better.sort_values(by=['rating', 'price'], ascending=[False, True]).drop_duplicates('model').head(1)

    # Budget alternative
    def get_budget_alternative():
        budget = similar[
            (similar['RAM_GB'] == selected_product['RAM_GB']) &
            (similar['Storage_GB'] == selected_product['Storage_GB']) &
            (similar['price'] < selected_product['price'])
        ]
        if budget.empty:
            budget = similar[
                (similar['price'] < selected_product['price']) & (similar['rating'] >= 3)
            ]
        return budget.sort_values(by='price').drop_duplicates('model').head(1)

    # Trending pick
    def get_trending_alternative():
        trending = similar.sort_values(by=['rating', 'review_count'], ascending=[False, False])
        return trending.drop_duplicates('model').head(1)

    better = get_better_alternative()
    budget = get_budget_alternative()
    trending = get_trending_alternative()

    if not better.empty:
        display_product(better.iloc[0], "🎯 Better Alternative")
    if not budget.empty:
        display_product(budget.iloc[0], "💸 Budget-Friendly Alternative")
    if not trending.empty:
        display_product(trending.iloc[0], "🔥 Trending Pick")

    # Accessory recommendations
    st.subheader("🛒 Recommended Accessories")
    accessories = []

    def get_accessories_for_smartphones():
        matched_model = df1[(df1['category'].str.contains("case|protector", case=False)) & (df1['model'] == selected_product['model'])]
        if not matched_model.empty:
            accessories.extend(matched_model.sample(min(2, len(matched_model))).to_dict('records'))
        headphones = df1[(df1['category'].str.contains("headphone", case=False)) & (df1['brand'] == selected_product['brand'])]
        if not headphones.empty:
            accessories.append(headphones.sample(1).iloc[0].to_dict())
        else:
            random_headphone = df1[df1['category'].str.contains("headphone", case=False)].sample(1).iloc[0].to_dict()
            accessories.append(random_headphone)
        if "iphone" in selected_product['product_title'].lower():
            iphone_model = ''.join(filter(str.isdigit, selected_product['model']))
            if iphone_model and iphone_model.isdigit() and int(iphone_model) in range(11, 16):
                charger = df1[df1['product_title'].str.lower().str.contains("apple charger")].sample(1)
                if not charger.empty:
                    accessories.append(charger.iloc[0].to_dict())

    def get_accessories_for_laptops():
        for item in ["mouse", "bag", "charger"]:
            found = df1[df1['category'].str.contains(item, case=False)]
            if not found.empty:
                accessories.append(found.sample(1).iloc[0].to_dict())

    if "phone" in selected_product['category'].lower():
        get_accessories_for_smartphones()
    elif "laptop" in selected_product['category'].lower():
        get_accessories_for_laptops()

    accessories = accessories[:4]
    for i, acc in enumerate(accessories):
        st.markdown(f"**Accessory {i+1} 🛍️:**")
        # For accessories, we can also reuse our two-column layout
        cols = st.columns([1, 3])
        with cols[0]:
            st.image(acc['image_url'], width=150)
        with cols[1]:
            st.markdown(f"🔹 **{acc['product_title']}**")
            st.markdown(f"💰 ₹{acc['price']}  |  ⭐ {acc['rating']}")
            st.markdown(f"🔍 Brand: {acc['brand']}  |  📦 Model: {acc['model']}")
            st.markdown(f"**Category:** {acc['category']}")

""")

from pyngrok import ngrok

# STEP 1: Set your authtoken
ngrok.set_auth_token("2vdayuRqKE4qLJ9cHDXEq5xs3GY_4oHek7NCMbtV1BbxxxWqf")

public_url = ngrok.connect(8501, "http")
print("Streamlit app is live at:", public_url)


# STEP 3: Start Streamlit in background
!streamlit run app.py &

print(f"✅ Your app is live at: {public_url}")