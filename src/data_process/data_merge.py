import pandas as pd


def d_merge(version_dir):
    first = pd.read_csv(version_dir / "matrix.csv")
    second = pd.read_csv(version_dir / "matrix_diffusion.csv")
    big_df = pd.merge(first, second, how='outer')
    big_df.to_csv(version_dir / "matrix_merge_dif.csv", index=False)

def s_merge(version_dir):
    first = pd.read_csv(version_dir / "matrix.csv")
    second = pd.read_csv(version_dir / "matrix_smote.csv")
    big_df = pd.merge(first, second, how='outer')
    big_df.to_csv(version_dir / "matrix_merge_smote.csv", index=False)

def g_merge(version_dir):
    first = pd.read_csv(version_dir / "matrix.csv")
    second = pd.read_csv(version_dir / "matrix_gan.csv")
    big_df = pd.merge(first, second, how='outer')
    big_df.to_csv(version_dir / "matrix_merge_gan.csv", index=False)

def b_merge(version_dir):
    first = pd.read_csv(version_dir / "matrix.csv")
    second = pd.read_csv(version_dir / "matrix_bert.csv")
    big_df = pd.merge(first, second, how='outer')
    big_df.to_csv(version_dir / "matrix_merge_bert.csv", index=False)

def v_merge(version_dir):
    first = pd.read_csv(version_dir / "matrix.csv")
    second = pd.read_csv(version_dir / "matrix_vae.csv")
    big_df = pd.merge(first, second, how='outer')
    big_df.to_csv(version_dir / "matrix_merge_vae.csv", index=False)

def gpt_merge(version_dir):
    first = pd.read_csv(version_dir / "matrix.csv")
    second = pd.read_csv(version_dir / "matrix_gpt.csv")
    big_df = pd.merge(first, second, how='outer')
    big_df.to_csv(version_dir / "matrix_merge_gpt.csv", index=False)

if __name__ == "__main__":
    d_merge()
