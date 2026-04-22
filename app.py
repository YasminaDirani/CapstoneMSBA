import os
import tempfile

import streamlit as st

from utils.ui import apply_design_system


def main() -> None:
    """Run the Streamlit app entrypoint on every rerun."""
    os.environ.setdefault(
        "MPLCONFIGDIR",
        os.path.join(tempfile.gettempdir(), "yasmina_matplotlib"),
    )

    st.set_page_config(
        page_title="Financial Aid Analytics",
        page_icon="📊",
        layout="wide",
    )

    apply_design_system(max_width=1500)

    navigation = st.navigation(
        [
            st.Page(
                "pages/Data_Overview.py",
                title="Data Overview",
                icon="🗂️",
                default=True,
            ),
            st.Page(
                "pages/Decision_Insights.py",
                title="Decision Insights",
                icon="🧠",
            ),
        ],
        position="sidebar",
    )

    navigation.run()


if __name__ == "__main__":
    main()
