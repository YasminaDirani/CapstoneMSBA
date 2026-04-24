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
                "pages/Executive_Summary.py",
                title="Start Here",
                icon="🏁",
                default=True,
            ),
            st.Page(
                "pages/Data_Reliability_QA.py",
                title="Data Reliability & QA",
                icon="🛡️",
            ),
            st.Page(
                "pages/Data_Overview.py",
                title="Evidence Overview",
                icon="🗂️",
            ),
            st.Page(
                "pages/Decision_Insights.py",
                title="Evidence Insights",
                icon="🧠",
            ),
            st.Page(
                "pages/Eligibility_Model.py",
                title="Eligibility Model",
                icon="🎯",
            ),
            st.Page(
                "pages/Aid_Percentage_Model.py",
                title="Aid Percentage Model",
                icon="📈",
            ),
            st.Page(
                "pages/Purchasing_Power_Experiment.py",
                title="Purchasing Power",
                icon="🪙",
            ),
            st.Page(
                "pages/Decision_Engine.py",
                title="Decision Engine",
                icon="⚖️",
            ),
            st.Page(
                "pages/Review_Queue.py",
                title="Action Queue",
                icon="🧾",
            ),
            st.Page(
                "pages/Final_Recommendation.py",
                title="Final Recommendation",
                icon="✅",
            ),
        ],
        position="sidebar",
    )

    navigation.run()


if __name__ == "__main__":
    main()
