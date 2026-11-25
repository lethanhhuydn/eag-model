# eag-model

**Evidence-Aggregation-Growth (EAG) model of competence: simulate data, check the model against your own data, plan your sample size.**

The EAG model writes a competence (here: *research-design competence* in AI- and IoT-integrated STEM projects) as a composition of three functions:

| Function | What it does | Formula |
|---|---|---|
| Evidence | links rubric scores to the proficiency of each component | rating scale model for indicator scores given `theta_k` |
| Growth | describes how each component develops over design cycles | `theta_k(t) = a_k + b_k * t`, mastery `pi_k = logistic(theta_k)` |
| Aggregation | combines components into one competence | `C = (sum_k w_k * pi_k ** rho) ** (1 / rho)` |

The substitution parameter `rho` is reported as the **compensation index** `kappa` (`kappa = 1 / (2 - rho)` for `rho <= 1`):

| kappa | Meaning |
|---|---|
| 1 | fully compensatory: competence is a weighted sum, as assumed whenever rubric points are added up |
| 0.5 | weighted geometric mean |
| towards 0 | conjunctive: competence follows the weakest component |

The package estimates `kappa` and the weights from ordinary classroom data (rubric scores plus holistic ratings), tests whether adding up rubric points is defensible, and turns the result into teaching information (which component to teach next for each student, and how many design cycles remain before a readiness threshold is reached).

## What you can do

1. **Simulate** data with known parameters, for teaching, method development and sample-size planning.
2. **Check your own data.** Upload three tables and receive a report with about 20 checks (rubric fit, reliability, rater agreement, test of sum-score aggregation, posterior predictive check, cross-validation, growth), each with its criterion and a verdict.
3. **Plan a study.** Estimate by simulation how many students are needed to detect non-compensatory structure.

## Install

```bash
pip install "git+https://github.com/lethanhhuydn/eag-model.git"
# with the point-and-click app:
pip install "eagmodel[app] @ git+https://github.com/lethanhhuydn/eag-model.git"
```

Python 3.9 or later. Dependencies: numpy, scipy, pandas, matplotlib.

## Use

### Point and click (EAG-Lab)

```bash
git clone https://github.com/lethanhhuydn/eag-model.git
cd eag-model
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The app has an English and a Vietnamese interface. It can also be hosted free of charge on Streamlit Community Cloud: sign in at share.streamlit.io with GitHub, choose this repository and the file `app/streamlit_app.py`.

### Command line

```bash
eag simulate --out mydata --n 200 --rho -2          # synthetic data in the input format
eag check --analytic mydata/analytic.csv --holistic mydata/holistic.csv \
          --students mydata/students.csv --out report --threshold 0.6
eag power --n 100 200 400 --rho -2 --reps 100        # sample-size planning
```

`eag check` writes `report.html` (self-contained), `summary.json`, `students.csv` and further tables.

### Python

```python
from eagmodel import EAGData, fit
from eagmodel.validate import validate
from eagmodel.report import write_report

data = EAGData.from_files("analytic.csv", "holistic.csv", "students.csv")
model = fit(data, threshold=0.6, extra_models=True)
print(model.free.kappa, model.free.kappa_ci, model.free.weights)
report = validate(model, threshold=0.6)
print(report.headline)
write_report(model, report, "report")
```

### Google Colab

Open the notebook in Colab with [this link](https://colab.research.google.com/github/lethanhhuydn/eag-model/blob/main/notebooks/eag_colab.ipynb) and run all cells. No installation on your computer is needed.

## Input format

Three long-format tables (CSV or Excel):

| File | Columns | Notes |
|---|---|---|
| Rubric scores | `student`, `occasion`, `component`, `indicator`, `score` | whole numbers from 0; at least two indicators per component |
| Holistic ratings | `student`, `occasion`, `rater`, `score` | any numeric scale (for example 0 to 100); two or more raters recommended |
| Students (optional) | `student`, `group` | exactly two groups, for example comparison and intervention |

`occasion` is numeric (design cycle 0, 1, 2, ...). One occasion is enough to check the aggregation function; two or more are needed for the growth function. Missing scores are allowed. A complete example is in `examples/data` (simulated data; `truth.json` holds the values used to generate them), and the report it produces is in `examples/report`.

## Reading the report

* **Verdicts.** PASS, CAUTION and FAIL apply the criterion printed beside each check. FINDING marks results that are informative and not a matter of passing.
* **Test of sum-score aggregation.** A small p-value means that the data contradict the assumption behind adding up rubric points.
* **`teach_next`** in `students.csv` is the component with the largest instructional leverage `dC / d theta_k`, that is, the component whose improvement raises that student's competence most.
* **`cycles_to_ready`** is the forecast number of further design cycles before the student's competence reaches the threshold, at the student's current growth rates.

## How it is estimated

1. Rubric parameters by conditional maximum likelihood, then growth parameters by marginal maximum likelihood on a grid, one component at a time.
2. The joint distribution of all components by an EM algorithm with importance sampling.
3. Weights, compensation index, rating link and rater severity by simulated maximum likelihood with analytic gradients.

Details are in `docs/model.md`. The stages run in sequence so that the meaning of each component is fixed by the rubric evidence alone.

## Reproducing the simulation study of the article

```bash
python paper/run_montecarlo.py --reps 200 --sens-reps 100 --workers 2 --out paper/results/montecarlo.csv
python paper/summarise.py          # tables in paper/results
python paper/extra_checks.py       # further checks reported in the article (paper/results/extra_checks.json)
python paper/make_figures.py       # figures in paper/figures
```

Seeds are fixed, so the numbers in the article are reproduced exactly on the same software versions (listed in `paper/results/environment.txt`). The full study (2,900 replications) takes about 50 minutes on two processor cores. The stored output of every replication is in `paper/results/montecarlo.csv`.

## Planning an empirical study

`docs/protocol.md` gives a step-by-step protocol for validating the model with real students (design, sample size, rubric, rater design, analysis plan and decision rules). `docs/rubric.md` contains the 20-indicator rubric.

## Limitations

The model assumes that rubric indicators follow a rating scale model within each component, that proficiency grows linearly on the logit scale, that latent variables are normally distributed and that holistic ratings are linear in competence with normal errors. The checks in the report probe several of these assumptions, and passing them does not prove the model. The estimator is sequential and approximate: in the simulation study it overestimated `kappa` by 3% to 9% for intermediate structures, its 95% confidence intervals covered the true value less often than 95% at the two ends of the range (prefer the likelihood-ratio test for decisions about the sum score), and in about 1% of cohorts of 100 students its importance sampling broke down (the report flags this under "Importance sampling"; do not use the results in that case). The results in the article come from simulated data; empirical validation is the purpose of the protocol.

## Tests

```bash
pip install -e ".[test]"
pytest -q
```

## Licence and citation

MIT licence. Please cite the article and this software (see `CITATION.cff`).

---

## Hướng dẫn nhanh (Tiếng Việt)

**eag-model** là phần mềm mã nguồn mở cho mô hình năng lực EAG (Bằng chứng, Tổng hợp, Tăng trưởng). Phần mềm làm ba việc:

1. **Mô phỏng** dữ liệu với tham số biết trước.
2. **Kiểm định dữ liệu của bạn**: tải lên ba bảng (điểm rubric, điểm tổng thể của người chấm, danh sách học sinh) và nhận báo cáo khoảng 20 phép kiểm định, mỗi phép có tiêu chí và kết luận (đạt, lưu ý, không đạt).
3. **Tính cỡ mẫu** bằng mô phỏng trước khi làm nghiên cứu thực nghiệm.

Cách dùng dễ nhất:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Chọn "Tiếng Việt" ở thanh bên trái. Thẻ "Kiểm định dữ liệu" có nút dùng dữ liệu ví dụ để xem trước báo cáo.

Ý nghĩa của **chỉ số bù trừ kappa**: bằng 1 nghĩa là có thể cộng điểm các thành tố; càng gần 0 thì thành tố yếu nhất càng quyết định năng lực chung, khi đó cộng điểm sẽ đánh giá sai những học sinh có hồ sơ năng lực lệch. Cột `teach_next` trong bảng học sinh cho biết thành tố nên dạy tiếp theo cho từng em.

Đề cương nghiên cứu thực nghiệm để các nhóm khác làm theo nằm ở `docs/protocol.md`; bộ rubric 20 chỉ báo nằm ở `docs/rubric.md`.
