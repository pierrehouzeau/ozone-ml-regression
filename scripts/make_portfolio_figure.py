import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def load_data(path: str = 'ozone.csv', target: str = 'O3obs'):
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {path}. Columns: {df.columns.tolist()}")
    df = df.dropna(subset=[target])
    y = df[target]
    X = df.drop(columns=[target])
    return X, y


def build_model(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # For robustness, ignore unseen/rare categories by using 'most_frequent' imputation only
    # (dataset expected mostly numeric; keep simple here)
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
    ], remainder='drop')  # drop non-numeric if any

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model),
    ])
    return pipe


def parity_plot(y_true, y_pred, out_svg='figures/ozone_portfolio.svg', out_png='figures/ozone_portfolio.png'):
    os.makedirs(os.path.dirname(out_svg), exist_ok=True)
    # Compute RMSE, compatible with older sklearn versions
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))

    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['font.size'] = 11

    plt.figure(figsize=(6, 6), dpi=180)
    plt.scatter(y_true, y_pred, s=14, alpha=0.55, edgecolor='none')
    lims = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    plt.plot(lims, lims, 'r--', lw=2, label='y = x (idéal)')
    plt.xlim(lims); plt.ylim(lims); plt.gca().set_aspect('equal', 'box')
    plt.xlabel('O3obs réel')
    plt.ylabel('O3obs prédit')
    plt.title(f'Ozone — Vrai vs Prédit (RMSE = {rmse:.2f})')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_svg, format='svg', bbox_inches='tight', facecolor='white')
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    return out_svg, out_png, rmse


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = build_model(X)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    svg, png, rmse = parity_plot(y_test, y_pred)
    print(f"Saved: {svg} and {png} (RMSE={rmse:.3f})")


if __name__ == '__main__':
    main()
