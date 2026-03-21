import os
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

@st.cache_data
def load_data(base_folder, date_choice):

    frames = []

    if date_choice == "All Dates":
        folders = [
            "February_10",
            "February_11",
            "February_12",
            "February_13",
            "February_14"
        ]
    else:
        folders = [date_choice]

    for folder in folders:
        folder_path = os.path.join(base_folder, folder)

        for file in os.listdir(folder_path):
            path = os.path.join(folder_path, file)

            try:
                df = pq.read_table(path).to_pandas()
                frames.append(df)
            except:
                continue

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)

    # decode event
    df["event"] = df["event"].apply(
        lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
    )

    return df