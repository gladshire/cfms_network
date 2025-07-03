import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Elution Trace Viewer", layout="wide")
st.title("Elution Trace Visualizer")

elut_data_file = st.file_uploader("Upload CF-MS elution score CSV", type=["csv"])

if elut_data_file is not None:
    # Load data
    df = pd.read_csv(elut_data_file)

    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Filter Options")

    # Protein ID filters
    protein_query = st.sidebar.text_input("Search by Protein ID (id1 or id2)").upper()
    
    # Score filter
    pearson_range = st.sidebar.slider("Pearson correlation range", -1.0, 1.0, (-1.0, 1.0), 0.01)
    euclidean_range = st.sidebar.slider("Euclidean distance range", 0.0, 2.0, (0.0, 1.0), 0.01)
    #confidence_range = st.sidebar.slider("Confidence score range", 0.0, 1.0, (0.5, 1.0), 0.01)

    # Interaction label filter
    label_options = st.sidebar.multiselect(
            "Interaction Label",
            options=sorted(df["label"].unique()),
            default=sorted(df["label"].unique()),
            format_func=lambda x: f"{x} (Interact)" if x == 0 else f"{x} (Non-Interact)"
    )

    # --- SORTING OPTIONS ---
    st.sidebar.header(" Sorting Options")

    sort_by = st.sidebar.selectbox(
            "Sort data by:",
            options=["pearson", "euclidean", "confidence"]
    )

    sort_ascending = st.sidebar.radio(
            "Sort direction:",
            options=["Ascending", "Descending"]
    )

    ascending = sort_ascending == "Ascending"

    # --- APPLY FILTERS ---
    filtered_df = df[
            (df["pearson"].between(*pearson_range)) &
            (df["euclidean"].between(*euclidean_range)) &
            #(df["confidence"].between(*confidence_range)) &
            (df["label"].isin(label_options))
    ]

    if protein_query:
        filtered_df = filtered_df[
            filtered_df["id1"].str.contains(protein_query) |
            filtered_df["id2"].str.contains(protein_query)
        ]

    # Add row index for mapping back
    filtered_df = filtered_df.copy()
    filtered_df["row_index"] = filtered_df.index

    # Sort data table by user-selected field
    filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending)

    # Display table
    st.markdown("### Filtered Protein Pairs")
    st.dataframe(filtered_df[["id1", "id2", "pearson", "euclidean", "confidence", "label"]],
                 use_container_width=True, height=300)

    if not filtered_df.empty:
        selected_row = st.selectbox(
                "Select a protein pair to view elution traces",
                options=filtered_df["row_index"],
                format_func=lambda idx: f"{df.loc[idx, 'id1']} vs {df.loc[idx, 'id2']} | "
                                        f"Pearson: {df.loc[idx, 'pearson']:.2f}, "
                                        f"Euclidean: {df.loc[idx, 'euclidean']:.2f}, "
                                        f"Confidence: {df.loc[idx, 'confidence']:.2f}"
        )

        row = df.loc[selected_row]

        
        # Extract traces
        trace1 = row[[col for col in df.columns if col.startswith("trace1_")]].values
        trace2 = row[[col for col in df.columns if col.startswith("trace2_")]].values

        # Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(trace1, label=f"Trace 1: {row['id1']}")
        ax.plot(trace2, label=f"Trace 2: {row['id2']}")
        ax.set_title("Elution Trace Pair")
        ax.set_xlabel("Fraction")
        ax.set_ylabel("Intensity")
        ax.legend()
        st.pyplot(fig)

        # Display metadata
        st.markdown("### Metadata")
        st.write({
            "Protein 1": row["id1"],
            "Protein 2": row["id2"],
            "Experiment": row["experiment"],
            "Euclidean Dist.": row["euclidean"],
            "Pearson  Corr.": row["pearson"],
            "Confidence": row["confidence"],
            "Label (0=pppi, 1=nppi)": row["label"]
        })
    else:
        st.warning("No rows match the current filters. Try adjusting your criteria.")

