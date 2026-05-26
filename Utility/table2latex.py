from pathlib import Path
import pandas as pd


def export_newton_table(filein, fileout, cols, formatters):
    df = pd.read_csv(
        filein,
        sep=r"\s+",
        engine="python",
        skiprows=[0, 2],
    )
    print(f"{df.columns=}\n {df=}")
    df = df[cols]

    df = df.rename(columns={
        "it": r"$k$",
        "mer": r"$\Phi_k$",
        "rhomer": r"$\Phi_k/\Phi_{k-1}$",
        "eta": r"$\eta_k$",
        "|F_h|": r"$\|F_k\|_{X'}$",
        "N": r"$N_k$",
        "meshiter": r"$\underline{\ell}_k$",
        "alpha": r"$\alpha_k$",
    })

    latex = df.to_latex(
        index=False,
        escape=False,
        formatters=formatters,
        column_format="r" * len(df.columns),
    )
    # latex = df.to_latex(
    #     index=False,
    #     escape=False,
    #     float_format=lambda x: f"{x:.3e}",
    #     column_format="r" * len(df.columns),
    # )

    with open(fileout, "w") as f:
        f.write(latex)


#===============================================================
if __name__ == "__main__":
    filein = Path("..") / Path("FEM1D") / "newton_history.txt"

    fileout = Path("/Users/becker/Nextcloud/Latex/Projects/ODE") / "example1_k3.tex"

    cols = [
        "it",
        "mer",
        "rhomer",
        "eta",
        "|F_h|",
        "N",
        "meshiter",
        "alpha",
    ]


    def fmt_exp(x):
        if pd.isna(x):
            return "--"

        m, e = f"{x:.2e}".split("e")
        e = int(e)

        return rf"${m}{{{e:+d}}}$"

    def fmt_alpha(x):
        if pd.isna(x):
            return "--"
        if x >= 0.01:
            return f"{x:.2f}"
        return "0.01"
    formatters = {
        r"$\alpha_k$": fmt_alpha, r"$\Phi_k/\Phi_{k-1}$": fmt_alpha
    }
    for c in [
        r"$\Phi_k$",
        r"$\eta_k$",
        r"$\|F_k\|_{X'}$",
    ]:
        formatters[c] = fmt_exp

    export_newton_table(
        filein=filein,
        fileout=fileout,
        cols=cols,
        formatters=formatters
    )