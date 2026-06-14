import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
import os
groq_api_key = st.secrets["GROQ_API_KEY"]

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
    .chat-message {
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
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
# Product Catalog
# ─────────────────────────────────────────
@st.cache_data
def load_catalog():
    data = {
        "product_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "name": [
            "Floral Kurta", "Slim Fit Jeans", "Embroidered Anarkali",
            "Casual White T-Shirt", "Silk Saree", "Denim Jacket",
            "Printed Maxi Dress", "Palazzo Pants", "Woolen Sweater", "Ethnic Lehenga"
        ],
        "category": [
            "Kurta", "Jeans", "Kurta", "T-Shirt", "Saree",
            "Jacket", "Dress", "Pants", "Sweater", "Lehenga"
        ],
        "price": [999, 1499, 2999, 499, 4999, 1999, 1799, 899, 1299, 5999],
        "color": [
            "Pink", "Blue", "Red", "White", "Golden",
            "Blue", "Multicolor", "Black", "Grey", "Pink"
        ],
        "description": [
            "A beautiful pink floral kurta perfect for casual outings and festivals",
            "Classic slim fit blue jeans suitable for everyday casual wear",
            "Elegant red embroidered anarkali suit perfect for weddings and parties",
            "Simple and comfortable white t-shirt for daily casual wear",
            "Gorgeous golden silk saree ideal for weddings and special occasions",
            "Trendy blue denim jacket perfect for winter casual outings",
            "Vibrant multicolor printed maxi dress great for beach and vacations",
            "Comfortable black palazzo pants suitable for office and casual wear",
            "Warm grey woolen sweater perfect for cold weather and winters",
            "Stunning pink ethnic lehenga choli perfect for weddings and festivals"
        ]
    }
    return pd.DataFrame(data)

# ─────────────────────────────────────────
# Load model & FAISS index
# ─────────────────────────────────────────
@st.cache_resource
def load_model_and_index(df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['description'].tolist())
    embedding_matrix = np.array(embeddings).astype('float32')
    dimension = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_matrix)
    return model, index

# ─────────────────────────────────────────
# RAG Core Function
# ─────────────────────────────────────────
def fashion_assistant(user_query, df, model, index, budget=None, groq_api_key=None):
    # Step 1: Embed query
    query_embedding = model.encode([user_query]).astype('float32')

    # Step 2: FAISS search
    distances, indices = index.search(query_embedding, k=3)

    # Step 3: Filter & build context
    results = []
    matched_products = []
    for idx in indices[0]:
        product = df.iloc[idx]
        if budget is None or product['price'] <= budget:
            results.append(
                f"- {product['name']} in {product['color']}: ₹{product['price']} — {product['description']}"
            )
            matched_products.append(product)

    if not results:
        return "Sorry, no products found within your budget! Try increasing your budget.", []

    context = "\n".join(results)

    # Step 4: LLM via Groq
    client = Groq(api_key=groq_api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are FashionFind, a friendly AI fashion assistant for an Indian fashion store. You help customers find perfect outfits. Be warm, enthusiastic, and give concise recommendations."
            },
            {
                "role": "user",
                "content": f"""A customer asked: "{user_query}"

Here are relevant products from our catalog:
{context}

Give a friendly recommendation in 3-4 sentences. Mention the product names naturally."""
            }
        ]
    )

    return response.choices[0].message.content, matched_products

# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────
st.markdown("# 👗 FashionFind AI")
st.markdown("##### Your personal AI fashion assistant — powered by RAG + LLaMA 3.3")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    groq_api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    st.markdown("Get your free key at [groq.com](https://console.groq.com)")
    st.markdown("---")
    budget = st.slider("💰 Max Budget (₹)", min_value=500, max_value=7000, value=7000, step=500)
    st.markdown(f"**Budget set to:** ₹{budget}")
    st.markdown("---")
    st.markdown("**Built by Neeta Singh** 👩‍💻")
    st.markdown("RAG Pipeline with:")
    st.markdown("- 🤗 Sentence Transformers")
    st.markdown("- ⚡ FAISS Vector Search")
    st.markdown("- 🦙 Groq LLaMA 3.3")
    st.markdown("- 🐼 Pandas")

# Load data
df = load_catalog()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hi! 👋 I'm FashionFind, your personal AI stylist. Tell me what you're looking for — an occasion, a vibe, a color, or a budget — and I'll find the perfect outfit for you! 🛍️"
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
                        <b>{p['name']}</b> &nbsp;|&nbsp; {p['color']} &nbsp;|&nbsp; ₹{p['price']}<br>
                        <small>{p['description']}</small>
                    </div>
                    """, unsafe_allow_html=True)

# Suggested prompts
if len(st.session_state.messages) == 1:
    st.markdown("##### 💡 Try asking:")
    cols = st.columns(2)
    prompts = [
        "Something for a wedding 💍",
        "Casual outfit under ₹1000 👕",
        "Show me something pink 🌸",
        "What to wear in winter? 🧥"
    ]
    for i, prompt in enumerate(prompts):
        if cols[i % 2].button(prompt):
            st.session_state.pending_prompt = prompt
            st.rerun()

# Handle suggested prompt click
if "pending_prompt" in st.session_state:
    user_input = st.session_state.pop("pending_prompt")
else:
    user_input = st.chat_input("Ask me anything about fashion... 👗")

# Process input
if user_input:
    if not groq_api_key:
        st.error("⚠️ Please enter your Groq API key in the sidebar to get started!")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant", avatar="👗"):
            with st.spinner("Finding perfect outfits for you...✨"):
                model, index = load_model_and_index(df)
                reply, matched_products = fashion_assistant(
                    user_input, df, model, index,
                    budget=budget if budget < 7000 else None,
                    groq_api_key=groq_api_key
                )
                st.write(reply)
                if matched_products:
                    for p in matched_products:
                        st.markdown(f"""
                        <div class="product-card">
                            <b>{p['name']}</b> &nbsp;|&nbsp; {p['color']} &nbsp;|&nbsp; ₹{p['price']}<br>
                            <small>{p['description']}</small>
                        </div>
                        """, unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": reply,
            "products": [p.to_dict() for p in matched_products] if matched_products else []
        })
