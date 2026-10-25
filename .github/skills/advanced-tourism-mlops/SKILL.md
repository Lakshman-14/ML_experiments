---
name: advanced-tourism-mlops
description: Complete the Visit with Us Tourism Package Prediction project end to end. Use when building or reviewing its customer purchase prediction pipeline, notebook, GitHub Actions CI/CD, or Streamlit deployment.
---

# Advanced Machine Learning and MLOps: Tourism Package Prediction

Complete the project in the current workspace, following its rubric and preserving existing user work. Treat the executed notebook HTML as the submission deliverable; the supporting repository should make the workflow reproducible.

## Project context

Build a binary classifier for Visit with Us to identify customers likely to purchase a tourism package before a marketing contact. The supplied dataset is `tourism.csv`. The target is `ProdTaken` (0/1); `CustomerID` is an identifier, not a model feature. The CSV may contain an exported unnamed index column, which should be removed.

The brief refers to a new Wellness Tourism Package, but the available label only says whether a customer purchased a package. Do not claim the model predicts Wellness-specific purchases unless the data or a verified label supports that claim. Clearly state this limitation in the notebook and app.

Dataset fields:

- Customer and contact: `CustomerID`, `Age`, `TypeofContact`, `CityTier`, `Occupation`, `Gender`, `NumberOfPersonVisiting`, `PreferredPropertyStar`, `MaritalStatus`, `NumberOfTrips`, `Passport`, `OwnCar`, `NumberOfChildrenVisiting`, `Designation`, `MonthlyIncome`
- Sales interaction: `PitchSatisfactionScore`, `ProductPitched`, `NumberOfFollowups`, `DurationOfPitch`
- Target: `ProdTaken`

## Required workflow

1. **Inspect before changing files.** Check the workspace, existing notebook/template, project conventions, dependencies, Git status, and dataset path. Preserve user edits and avoid replacing a provided template without reason. If deployment or GitHub actions require secrets, credentials, or account access, request them rather than inventing values.
2. **Register and validate data.** Create a clear project root with a `data/` directory and place the source CSV there when appropriate. Add a validation script that checks the required columns, reports missing or unexpected fields, and prints a useful data summary (shape, target counts, missingness). Keep the original data intact.
3. **Prepare data reproducibly.** Load from the repository's `data/` directory, remove only justified columns (including an exported index and `CustomerID` from features), handle missing values and categorical/numeric columns in a leakage-safe preprocessing pipeline, and split train/test with a fixed seed and stratification when valid. Save splits locally and make them workflow artifacts for the next job. Do not fit preprocessing on test data.
4. **Train and track experiments.** Load the workflow artifacts, define and tune one or more rubric-approved models (Decision Tree, Bagging, Random Forest, AdaBoost, Gradient Boosting, or XGBoost if available). Use a reproducible search and appropriate cross-validation. Record every tried/tuned parameter and selection metric in a durable, inspectable form or an established experiment tracker. Report class balance and relevant classification metrics; include confusion matrix and threshold-aware precision/recall where useful. Select the model using a stated criterion, evaluate once on the held-out test set, and save the complete preprocessing-plus-model pipeline.
5. **Deploy with Streamlit.** Build an app that loads the committed model artifact, gathers the relevant customer inputs, constructs a one-row dataframe with the exact training feature names and types, and shows a prediction with a clear explanation that it is an estimate. Include dependency files suitable for Streamlit Community Cloud. Document the deployment steps; do not claim deployment is live unless it has been verified.
6. **Automate with GitHub Actions.** Add `.github/workflows/pipeline.yml` (or preserve a clear existing equivalent). The workflow should run data validation, preparation, and model training in ordered jobs; pass train/test files using workflow artifacts; run relevant checks; and retain or commit the selected model so deployment can load it from the repository. Pin or constrain dependencies sensibly. If automatically committing generated model updates to `main` is required, use narrowly scoped `contents: write`, a bot identity, and a loop-safe trigger strategy; never commit credentials, datasets with restricted data, or unrelated generated files. Avoid destructive force pushes.
7. **Create the notebook submission.** Provide a well-structured, sequentially executable Python notebook with business framing, data registration/validation, cleaning and preparation, exploratory analysis, modeling/tuning, evaluation, conclusions, and business recommendations. Include visible outputs and concise observations grounded in results. Export the fully executed notebook to HTML only after running all cells successfully. Include the GitHub repository and verified Streamlit Community Cloud URL in the notebook; if a link is not yet available, label it as pending rather than fabricating it.
8. **Verify against the rubric.** Run the smallest relevant tests/checks, execute the notebook end to end, validate workflow YAML and app startup where possible, and check that the app's input schema matches the saved model pipeline. Summarize what was actually verified and what still requires external account/deployment access.

## Rubric completion checklist

- **Data registration (3):** project root, `data/` folder, CSV, validation script with required-column checks and printed summary.
- **Data preparation (7):** repository-relative loading, documented cleaning, justified feature removal, train/test split saved locally, and workflow artifact handoff.
- **Model building and experiment tracking (13):** artifact loading, approved model, parameter tuning, complete tuned-parameter logging, held-out evaluation, and saved best pipeline.
- **Model deployment (11):** Streamlit app, correctly shaped input dataframe, deployment dependencies, and documented/verified Streamlit Community Cloud deployment.
- **GitHub Actions MLOps pipeline (16):** `pipeline.yml`, ordered end-to-end jobs, artifact passing, reproducible automation, and safe model-update behavior on `main`.
- **Output evaluation (4):** GitHub repository link, evidence of folder structure and successful workflow, Streamlit Community Cloud link, and app screenshot. Do not fabricate screenshots or successful run evidence.
- **Notebook quality (6):** clear sections and flow, readable/comments where useful, all cells executed without errors, visible outputs, conclusions, and repository links.

## Completion behavior

- Implement the complete requested scope rather than stopping at a plan, unless blocked by missing user decisions, credentials, or external access.
- Keep data, model, and app feature schemas consistent; test serialization and loading.
- Do not use the held-out test set for tuning or repeatedly adjusting the model.
- Do not hide errors with broad exception handling or report success for unverified deployments/actions.
- Follow existing project style and dependency choices; avoid unrelated changes.
- When implementation is requested, report changed files, checks run and their outcomes, and any external steps still needed for GitHub or Streamlit deployment.
