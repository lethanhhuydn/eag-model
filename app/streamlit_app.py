"""EAG-Lab: a point-and-click front end for the eagmodel package.

Run locally with:   streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eagmodel import EAGData, SimConfig, fit, simulate          # noqa: E402
from eagmodel import plots                                       # noqa: E402
from eagmodel.functions import kappa_from_rho, rho_from_kappa    # noqa: E402
from eagmodel.power import power_analysis                        # noqa: E402
from eagmodel.report import to_html                              # noqa: E402
from eagmodel.validate import validate                           # noqa: E402

T = {
    "en": dict(
        title="EAG-Lab", tagline="Simulate, check and plan studies with the Evidence-Aggregation-Growth model of competence",
        lang="Language", tab_check="Check my data", tab_sim="Simulate", tab_power="Plan sample size", tab_help="Data format",
        up_a="Rubric scores (analytic): student, occasion, component, indicator, score",
        up_h="Holistic ratings: student, occasion, rater, score",
        up_s="Students (optional): student, group",
        example="Use the built-in example data", run="Run the model check", thr="Readiness threshold on the competence scale",
        use_thr="Report readiness against a threshold", working="Fitting the model and running the checks ...",
        need="Upload the rubric scores and the holistic ratings, or tick the example box.",
        checks="Checks", agg="Aggregation function", tables="Tables", students="Students", dl_html="Download the full report (HTML)",
        dl_students="Download the student table (CSV)", n="Number of students", kappa="Compensation index (kappa)",
        kappa_help="1 = sum score, 0.5 = geometric mean, towards 0 = the weakest component decides",
        noise="Rater noise (SD in rating points)", corr="Correlation between components", seed="Random seed",
        gen="Generate data", dl_zip="Download the three CSV files (zip)", preview="Preview",
        sizes="Sample sizes to compare (comma separated)", reps="Replications per sample size",
        run_power="Run the simulation", power_note="Each replication simulates a cohort and fits the model, so this can take a few minutes.",
        power_cols="Power is the share of replications in which the sum-score model is rejected at the 5% level.",
    ),
    "vi": dict(
        title="EAG-Lab", tagline="Mô phỏng, kiểm định và lập kế hoạch nghiên cứu với mô hình năng lực Bằng chứng - Tổng hợp - Tăng trưởng (EAG)",
        lang="Ngôn ngữ", tab_check="Kiểm định dữ liệu", tab_sim="Mô phỏng", tab_power="Tính cỡ mẫu", tab_help="Định dạng dữ liệu",
        up_a="Điểm rubric (phân tích): student, occasion, component, indicator, score",
        up_h="Điểm tổng thể: student, occasion, rater, score",
        up_s="Học sinh (không bắt buộc): student, group",
        example="Dùng dữ liệu ví dụ có sẵn", run="Chạy kiểm định mô hình", thr="Ngưỡng sẵn sàng trên thang năng lực",
        use_thr="Báo cáo mức sẵn sàng theo ngưỡng", working="Đang ước lượng mô hình và chạy các phép kiểm định ...",
        need="Hãy tải lên điểm rubric và điểm tổng thể, hoặc chọn dữ liệu ví dụ.",
        checks="Các phép kiểm định", agg="Hàm tổng hợp", tables="Bảng kết quả", students="Học sinh", dl_html="Tải báo cáo đầy đủ (HTML)",
        dl_students="Tải bảng học sinh (CSV)", n="Số học sinh", kappa="Chỉ số bù trừ (kappa)",
        kappa_help="1 = cộng điểm, 0.5 = trung bình nhân, tiến về 0 = thành tố yếu nhất quyết định",
        noise="Nhiễu của người chấm (độ lệch chuẩn, điểm)", corr="Tương quan giữa các thành tố", seed="Hạt giống ngẫu nhiên",
        gen="Tạo dữ liệu", dl_zip="Tải ba file CSV (zip)", preview="Xem trước",
        sizes="Các cỡ mẫu cần so sánh (cách nhau bằng dấu phẩy)", reps="Số lần lặp cho mỗi cỡ mẫu",
        run_power="Chạy mô phỏng", power_note="Mỗi lần lặp mô phỏng một nhóm học sinh rồi ước lượng mô hình, nên có thể mất vài phút.",
        power_cols="Lực kiểm định là tỉ lệ số lần lặp bác bỏ mô hình cộng điểm ở mức 5%.",
    ),
}

HELP = {
    "en": """
**Three long-format tables** (CSV or Excel). Column names must match; the order of columns is free.

| File | Columns | Notes |
|---|---|---|
| Rubric scores | `student`, `occasion`, `component`, `indicator`, `score` | whole numbers from 0 (lowest rubric level); at least two indicators per component |
| Holistic ratings | `student`, `occasion`, `rater`, `score` | any numeric scale, for example 0 to 100 |
| Students (optional) | `student`, `group` | exactly two groups, for example comparison and intervention |

`occasion` is a number (design cycle 0, 1, 2, ...). One occasion is enough to check the aggregation function;
two or more are needed for the growth function.
""",
    "vi": """
**Ba bảng dạng dài** (CSV hoặc Excel). Tên cột phải đúng; thứ tự cột tuỳ ý.

| File | Cột | Ghi chú |
|---|---|---|
| Điểm rubric | `student`, `occasion`, `component`, `indicator`, `score` | số nguyên từ 0 (mức thấp nhất); mỗi thành tố có ít nhất hai chỉ báo |
| Điểm tổng thể | `student`, `occasion`, `rater`, `score` | thang số bất kỳ, ví dụ 0 đến 100 |
| Học sinh (không bắt buộc) | `student`, `group` | đúng hai nhóm, ví dụ đối chứng và thực nghiệm |

`occasion` là một con số (chu trình thiết kế 0, 1, 2, ...). Một lần đo là đủ để kiểm định hàm tổng hợp;
cần từ hai lần đo trở lên để ước lượng hàm tăng trưởng.
""",
}

BADGE = {"pass": "🟢", "caution": "🟡", "fail": "🔴", "info": "🔵"}


def read_table(upload):
    return pd.read_excel(upload) if upload.name.lower().endswith((".xlsx", ".xls")) else pd.read_csv(upload)


@st.cache_data(show_spinner=False)
def run_check(a_csv: str, h_csv: str, s_csv, threshold, seed: int):
    a, h = pd.read_csv(io.StringIO(a_csv)), pd.read_csv(io.StringIO(h_csv))
    s = None if s_csv is None else pd.read_csv(io.StringIO(s_csv))
    data = EAGData.from_frames(a, h, s)
    f = fit(data, seed=seed, threshold=threshold, extra_models=True)
    rep = validate(f, threshold=threshold, seed=seed)
    free = f.free
    figs = dict(profile=plots.profile_figure(rep.profile, free.kappa, free.kappa_ci),
                form=plots.form_figure(rep.tables["functional_form"]),
                weights=plots.weights_figure(rep.tables["weights"]))
    if f.has_growth:
        figs["growth"] = plots.trajectory_figure(f, rep.tables["students"])
    tables = {k: rep.tables[k] for k in ("models", "components", "items", "raters", "students", "weights")}
    return dict(headline=rep.headline, checks=rep.checks_frame(), figs=figs, tables=tables,
                html=to_html(f, rep), seconds=f.seconds)


def main():
    st.set_page_config(page_title="EAG-Lab", layout="centered")
    lang = st.sidebar.radio("Language / Ngôn ngữ", ["en", "vi"], format_func=lambda v: {"en": "English", "vi": "Tiếng Việt"}[v])
    t = T[lang]
    st.title(t["title"])
    st.caption(t["tagline"])
    tab_check, tab_sim, tab_power, tab_help = st.tabs([t["tab_check"], t["tab_sim"], t["tab_power"], t["tab_help"]])

    # ---------------------------------------------------------------- check
    with tab_check:
        up_a = st.file_uploader(t["up_a"], type=["csv", "xlsx"], key="up_a")
        up_h = st.file_uploader(t["up_h"], type=["csv", "xlsx"], key="up_h")
        up_s = st.file_uploader(t["up_s"], type=["csv", "xlsx"], key="up_s")
        example = st.checkbox(t["example"], key="example")
        use_thr = st.checkbox(t["use_thr"], value=True, key="use_thr")
        thr = st.slider(t["thr"], 0.2, 0.9, 0.6, 0.05, key="thr") if use_thr else None
        if st.button(t["run"], type="primary", key="run"):
            if example:
                data, _ = simulate(SimConfig(n_students=200), seed=2026)
                a, h, s = data.to_frames()
            elif up_a is not None and up_h is not None:
                a, h = read_table(up_a), read_table(up_h)
                s = read_table(up_s) if up_s is not None else None
            else:
                a = None
                st.warning(t["need"])
            if a is not None:
                try:
                    with st.spinner(t["working"]):
                        st.session_state["result"] = run_check(a.to_csv(index=False), h.to_csv(index=False),
                                                               None if s is None else s.to_csv(index=False), thr, 0)
                except Exception as exc:                      # show data problems in plain words
                    st.error(str(exc))
        res = st.session_state.get("result")
        if res:
            st.success(res["headline"])
            st.subheader(t["checks"])
            ch = res["checks"].copy()
            ch["verdict"] = ch["verdict"].map(lambda v: f"{BADGE[v]} {v}")
            st.dataframe(ch, hide_index=True)
            st.subheader(t["agg"])
            for key in ("profile", "form", "weights", "growth"):
                if key in res["figs"]:
                    st.image(res["figs"][key])
            st.subheader(t["tables"])
            for key in ("models", "components", "items", "raters"):
                st.dataframe(res["tables"][key].round(3), hide_index=True)
            st.subheader(t["students"])
            st.dataframe(res["tables"]["students"].round(3), hide_index=True)
            st.download_button(t["dl_html"], res["html"], "eag_report.html", "text/html", key="dl_html")
            st.download_button(t["dl_students"], res["tables"]["students"].to_csv(index=False), "students.csv",
                               "text/csv", key="dl_students")

    # ---------------------------------------------------------------- simulate
    with tab_sim:
        c1, c2 = st.columns(2)
        n = c1.number_input(t["n"], 40, 2000, 200, 20, key="sim_n")
        kappa = c2.slider(t["kappa"], 0.05, 1.5, 0.25, 0.05, help=t["kappa_help"], key="sim_kappa")
        noise = c1.slider(t["noise"], 2.0, 20.0, 8.0, 1.0, key="sim_noise")
        corr = c2.slider(t["corr"], 0.0, 0.9, 0.5, 0.1, key="sim_corr")
        seed = c1.number_input(t["seed"], 0, 10 ** 6, 1, key="sim_seed")
        if st.button(t["gen"], key="gen"):
            cfg = SimConfig(n_students=int(n), rho=float(rho_from_kappa(kappa)), h_sd=float(noise), corr_a=float(corr))
            data, truth = simulate(cfg, seed=int(seed))
            a, h, s = data.to_frames()
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr("analytic.csv", a.to_csv(index=False))
                z.writestr("holistic.csv", h.to_csv(index=False))
                z.writestr("students.csv", s.to_csv(index=False))
            st.session_state["sim"] = dict(zip=buf.getvalue(), a=a.head(12), h=h.head(8),
                                           c=pd.DataFrame(truth["C"], columns=[f"cycle {v:g}" for v in data.times]).describe().round(3))
        sim = st.session_state.get("sim")
        if sim:
            st.download_button(t["dl_zip"], sim["zip"], "eag_simulated_data.zip", "application/zip", key="dl_zip")
            st.subheader(t["preview"])
            st.dataframe(sim["a"], hide_index=True)
            st.dataframe(sim["h"], hide_index=True)
            st.dataframe(sim["c"])

    # ---------------------------------------------------------------- power
    with tab_power:
        st.caption(t["power_note"])
        sizes = st.text_input(t["sizes"], "100, 200", key="pw_sizes")
        reps = st.number_input(t["reps"], 5, 500, 20, 5, key="pw_reps")
        kappa_p = st.slider(t["kappa"], 0.05, 1.5, 0.25, 0.05, help=t["kappa_help"], key="pw_kappa")
        if st.button(t["run_power"], key="run_power"):
            try:
                ns = [int(v) for v in sizes.replace(";", ",").split(",") if v.strip()]
                bar = st.progress(0.0)
                st.session_state["power"] = power_analysis(SimConfig(rho=float(rho_from_kappa(kappa_p))), ns, int(reps),
                                                           progress=lambda f: bar.progress(min(f, 1.0)))
            except Exception as exc:
                st.error(str(exc))
        pw = st.session_state.get("power")
        if pw is not None:
            st.dataframe(pw.round(3), hide_index=True)
            st.caption(t["power_cols"])

    with tab_help:
        st.markdown(HELP[lang])
        st.latex(r"C_i(t)=\Big(\sum_k w_k\,\pi_{ik}(t)^{\rho}\Big)^{1/\rho},\qquad \pi_{ik}(t)=\frac{1}{1+e^{-(a_{ik}+b_{ik}t)}}")


main()
