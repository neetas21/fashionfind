import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="FashionFind 🛍️",
    page_icon="👗",
    layout="centered"
)

# ─────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #fdf6f0; }
    .stTextInput > div > div > input {
        border-radius: 20px;
        border: 2px solid #e91e8c;
    }
    .stButton > button {
        background-color: #e91e8c;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #c2185b;
        color: white;
    }
    .product-card {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid #e91e8c;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Load Real Myntra Data
# ─────────────────────────────────────────
@st.cache_data
def load_catalog():
    # Load real cleaned Myntra data
    df = pd.read_csv('https://raw.githubusercontent.com/neetas21/fashionfind/main/myntra_product_data.csv')
    # Make sure description has no nulls (safety check)
    df = df[df['description'].notna()].reset_index(drop=True)
    return df

# ─────────────────────────────────────────
# Load Sentence Transformer + FAISS index
# ─────────────────────────────────────────
@st.cache_resource
def load_model_and_index(_df):
    # Why all-MiniLM-L6-v2? It's fast, lightweight, and great for
    # semantic similarity — perfect for product search
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Convert every product description into a 384-dimensional vector
    embeddings = model.encode(_df['description'].tolist(), show_progress_bar=False)
    embedding_matrix = np.array(embeddings).astype('float32')

    # Build FAISS index — IndexFlatL2 uses L2 (Euclidean) distance
    # Smaller distance = more similar products
    dimension = embedding_matrix.shape[1]  # 384
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_matrix)

    return model, index

# ─────────────────────────────────────────
# RAG Core Function
# ─────────────────────────────────────────
def fashion_assistant(user_query, df, model, index, budget=None, groq_api_key=None):
    # Step 1: Convert user query to embedding
    query_embedding = model.encode([user_query]).astype('float32')

    # Step 2: Search FAISS for top 5 similar products
    distances, indices = index.search(query_embedding, k=5)

    # Step 3: Filter by budget if set, build context for LLM
    results = []
    matched_products = []
    for idx in indices[0]:
        product = df.iloc[idx]
        if budget is None or product['price'] <= budget:
            results.append(
                f"- {product['name']} by {product['seller']}: "
                f"₹{product['price']} | Rating: {product['rating']}/5 | "
                f"Category: {product['category']}"
            )
            matched_products.append(product)

    if not results:
        return "Sorry, no products found within your budget! Try increasing your budget.", []

    context = "\n".join(results)

    # Step 4: Send to Groq LLaMA 3.3 to generate natural response
    client = Groq(api_key=groq_api_key.strip())
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are FashionFind, a friendly AI fashion assistant for Myntra. Be warm, enthusiastic, and give concise recommendations. Always mention brand names and prices."
            },
            {
                "role": "user",
                "content": f"""A customer asked: "{user_query}"

Here are the most relevant products from our catalog:
{context}

Give a friendly, helpful recommendation in 3-4 sentences. Mention specific product names and brands naturally."""
            }
        ]
    )

    return response.choices[0].message.content, matched_products

# ─────────────────────────────────────────
# UI Layout
# ─────────────────────────────────────────
st.markdown("# 👗 FashionFind AI")
st.markdown("##### Your personal AI fashion assistant — 500+ real Myntra products powered by RAG + LLaMA 3.3")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    groq_api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    st.markdown("Get your free key at [groq.com](https://console.groq.com)")
    st.markdown("---")
    budget = st.slider("💰 Max Budget (₹)", min_value=500, max_value=15000, value=15000, step=500)
    st.markdown(f"**Budget:** ₹{budget}")
    st.markdown("---")
    st.markdown("**Built by Neeta Singh** 👩‍💻")
    st.markdown("**Data:** Real Myntra catalog (cleaned from 1M+ products)")
    st.markdown("**Stack:**")
    st.markdown("- 🤗 Sentence Transformers")
    st.markdown("- ⚡ FAISS Vector Search")
    st.markdown("- 🦙 Groq LLaMA 3.3")
    st.markdown("- 🐼 Pandas")

# Load data
df = load_catalog()

# Show dataset stats
col1, col2, col3 = st.columns(3)
col1.metric("🛍️ Products", f"{len(df):,}")
col2.metric("⭐ Avg Rating", f"{df['rating'].mean():.1f}/5")
col3.metric("💰 Price Range", f"₹{df['price'].min()}-₹{df['price'].max():,}")

st.markdown("---")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hi! 👋 I'm FashionFind, your personal AI stylist with access to 500+ real Myntra products. Tell me what you're looking for — an occasion, a vibe, a brand, or a budget — and I'll find the perfect outfit! 🛍️"
    })

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant", avatar="👗"):
            st.write(msg["content"])
            if "products" in msg and msg["products"]:
                for p in msg["products"]:
                    st.markdown(f"""
                    <div class="product-card">
                        <b>{p['name']}</b> &nbsp;|&nbsp; 
                        <b>{p['seller']}</b> &nbsp;|&nbsp; 
                        ₹{p['price']} &nbsp;|&nbsp; 
                        ⭐ {p['rating']}/5<br>
                        <small>Category: {p['category']}</small><br>
                        <small><a href="{p['purl']}" target="_blank">View on Myntra →</a></small>
                    </div>
                    """, unsafe_allow_html=True)

# Suggested prompts on first load
if len(st.session_state.messages) == 1:
    st.markdown("##### 💡 Try asking:")
    cols = st.columns(2)
    prompts = [
        "Show me top rated products 🌟",
        "Casual outfit under ₹1000 👕",
        "Something from Roadster 🏷️",
        "Best rated jeans 👖"
    ]
    for i, prompt in enumerate(prompts):
        if cols[i % 2].button(prompt):
            st.session_state.pending_prompt = prompt
            st.rerun()

# Handle suggested prompt
if "pending_prompt" in st.session_state:
    user_input = st.session_state.pop("pending_prompt")
else:
    user_input = st.chat_input("Ask me anything about fashion... 👗")

# Process input
if user_input:
    if not groq_api_key:
        st.error("⚠️ Please enter your Groq API key in the sidebar!")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant", avatar="👗"):
            with st.spinner("Searching 500+ Myntra products for you...✨"):
                model, index = load_model_and_index(df)
                reply, matched_products = fashion_assistant(
                    user_input, df, model, index,
                    budget=budget if budget < 15000 else None,
                    groq_api_key=groq_api_key
                )
                st.write(reply)
                if matched_products:
                    for p in matched_products:
                        st.markdown(f"""
                        <div class="product-card">
                            <b>{p['name']}</b> &nbsp;|&nbsp;
                            <b>{p['seller']}</b> &nbsp;|&nbsp;
                            ₹{p['price']} &nbsp;|&nbsp;
                            ⭐ {p['rating']}/5<br>
                            <small>Category: {p['category']}</small><br>
                            <small><a href="{p['purl']}" target="_blank">View on Myntra →</a></small>
                        </div>
                        """, unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": reply,
            "products": [p.to_dict() for p in matched_products] if matched_products else []
        })
