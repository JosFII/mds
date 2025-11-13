import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import MDS
from sklearn.datasets import make_blobs
st.markdown(
'''
<style>
    .stApp {
   background-color: white;
    }
 
       .stWrite,.stMarkdown,.stTextInput,h1, h2, h3, h4, h5, h6 {
            color: purple !important;
        }
</style>
''',
unsafe_allow_html=True
)

st.title("MDS")
st.markdown('dibuat oleh: Joseph F.H. (20234920002)')


st.header("contoh 1")
st.markdown("""
berikut kode yang digunakan: \n
        data = load_digits()
            X, y = data.data, data.target

            st.write('Original Dimesnion of X = ', X.shape)

            n_components = 2  
            mds = MDS(n_components=n_components, random_state=209)

            X_reduced = mds.fit_transform(X)

            fig=plt.figure(figsize=(8, 6))
            plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=plt.cm.get_cmap("jet", 10))
            plt.colorbar(label='Digit Label', ticks=range(10))
            plt.title("MDS Visualization of Digits Dataset")
            plt.xlabel("MDS Dimension 1")
            plt.ylabel("MDS Dimension 2")
            st.pyplot(fig)    
""")
data = load_digits()
X, y = data.data, data.target

st.write('Original Dimesnion of X = ', X.shape)
# Create an MDS model with the desired number of dimensions
# Number of dimensions for visualization
n_components = 2  
mds = MDS(n_components=n_components, random_state=209)

# Fit the MDS model to your data
X_reduced = mds.fit_transform(X)

st.write('Dimesnion of X after MDS = ',X_reduced.shape)

# Visualize the reduced data
fig=plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(label='Digit Label', ticks=range(10))
plt.title("MDS Visualization of Digits Dataset")
plt.xlabel("MDS Dimension 1")
plt.ylabel("MDS Dimension 2")
st.pyplot(fig)

st.write("""
bisa dilihat bahwa data yang awalnya memiliki 64 dimensi dijadikan menjadi dua dimensi saja. \n
dari hasil scatterplot mds, dapat dilihat bahwa setiap digit berkumpul dengan digit yang sama, kecuali untuk digit 5 yang agak tersebar
pada plot, dan digit 8 yang cukup tersebar tapi masih cukup berpusat di tengah plot. 
""")



st.header("contoh 2")
st.markdown('''
berikut kode yang digunakan: \n
            X, _ = make_blobs(n_samples=100, n_features=3, centers=2, random_state=42)

            st.write('Original Dimension of X : ', X.shape)


            mds = MDS(n_components=2, random_state=42)
            X_2d = mds.fit_transform(X)

            st.write('Dimension of X after MDS : ', X_2d.shape)

            fig2=plt.figure(figsize=(8, 6))
            plt.scatter(X_2d[:, 0], X_2d[:, 1])
            plt.title("MDS Visualization")
            plt.xlabel("MDS Dimension 1")
            plt.ylabel("MDS Dimension 2")
            st.pyplot(fig2)
''')

# Generate a sample dataset (you can replace this with your own data)
X, _ = make_blobs(n_samples=100, n_features=3, centers=2, random_state=42)

st.write('Original Dimension of X : ', X.shape)
# Perform MDS to reduce the dimensionality to 2D

mds = MDS(n_components=2, random_state=42)
X_2d = mds.fit_transform(X)

st.write('Dimension of X after MDS : ', X_2d.shape)

# Plot the results
fig2=plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1])
plt.title("MDS Visualization")
plt.xlabel("MDS Dimension 1")
plt.ylabel("MDS Dimension 2")
st.pyplot(fig2)

st.write("""
pada data ini, mds telah membuat data yang memiliki tiga dimensi menjadi dua dimensi. \n
di plot tersebut bisa dilihat bahwa data terbagi menjadi dua kelompok, satu di kiri bawah plot, dan yang kedua di kanan atas plot.
""")