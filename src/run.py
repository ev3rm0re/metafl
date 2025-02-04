import os
import sys
import pandas as pd
from data_process import data_merge, testcase_aug, testcase_diffusion, testcase_smote, testcase_bert, testcase_vae, testcase_gpt
from calculate_suspiciousness import rank
from calculate_suspiciousness import contrast_l
from data_process.data_undersampling.undersampling import UndersamplingData
from read_data.Defects4JDataLoader import Defects4JDataLoader
from paths import Paths
from natsort import natsorted


class Pipeline:
    def __init__(self, project_dir, program, bug_id):
        self.project_dir = project_dir
        self.program = program
        self.bug_id = bug_id
        self.dataloader = self._choose_dataloader_obj()
        self.column_raw = self._choose_dataloader_obj().column_raw

    def run(self):
        self._run_task()
        return self.df, self.column_raw

    def _dynamic_choose(self, loader):
        self.dataset_dir = self.project_dir
        data_obj = loader(self.dataset_dir, self.program, self.bug_id)
        data_obj.load()
        return data_obj

    def _choose_dataloader_obj(self):
        return self._dynamic_choose(Defects4JDataLoader)

    def _run_task(self):
        self.data_obj = UndersamplingData(self.dataloader)
        self.df = self.data_obj.process()


def undersam(program, version, methods):
    print(f"\n****Calculating undersample version for program `{program}` and version `{version}` rank")
    project_dir = Paths.DatasetRoot
    version_dir = Paths.get_d4j_version_dir(program, version)
    pl = Pipeline(project_dir, program, version)
    df, column_raw = pl.run()
    column_raw.append('error')
    df.columns = column_raw
    under_sam_file_path = version_dir / "under_sam.csv"
    df.to_csv(under_sam_file_path, index=0)
    data = pd.read_csv(under_sam_file_path, index_col=0)
    for method in methods:
        rank_temporary = f"rank_{method}_undersam_temp.csv"
        rank_final_method = f"rank_{method}_undersam.csv"
        rank.rank(data, method, version_dir / rank_temporary)
        contrast_l.contrast_line(
            version_dir / rank_temporary,
            version_dir / rank_final_method,
            version_dir / "bugline.csv"
        )


def cal_diffusion(program, version, methods):
    print(f"\n****Calculating diffusion version for program `{program}` and version `{version}` rank")
    version_dir = Paths.get_d4j_version_dir(program, version)
    testcase_diffusion.main(program, version)       # 扩散模型
    data_merge.d_merge(version_dir)
    data = pd.read_csv(version_dir / "matrix_merge_dif.csv", index_col=0)
    for method in methods:
        rank_temporary = f"rank_{method}_dif_temp.csv"
        rank_final_method = f"rank_{method}_dif.csv"
        rank.rank(data, method, version_dir / rank_temporary)
        contrast_l.contrast_line(
            version_dir / rank_temporary,
            version_dir / rank_final_method,
            version_dir / "bugline.csv",
        )

def cal_gan(program, version, methods):
    print(f"\n****Calculating GAN version for program `{program}` and version `{version}` rank")
    version_dir = Paths.get_d4j_version_dir(program, version)
    testcase_aug.main(program, version)             # gan模型
    data_merge.g_merge(version_dir)
    data = pd.read_csv(version_dir / "matrix_merge_gan.csv", index_col=0)
    for method in methods:
        rank_temporary = f"rank_{method}_gan_temp.csv"
        rank_final_method = f"rank_{method}_gan.csv"
        rank.rank(data, method, version_dir / rank_temporary)
        contrast_l.contrast_line(
            version_dir / rank_temporary,
            version_dir / rank_final_method,
            version_dir / "bugline.csv",
        )

def cal_smote(program, version, methods):
    print(f"\n****Calculating smote version for program `{program}` and version `{version}` rank")
    version_dir = Paths.get_d4j_version_dir(program, version)
    testcase_smote.main(program, version)   # smote模型
    data_merge.s_merge(version_dir)
    data = pd.read_csv(version_dir / "matrix_merge_smote.csv", index_col=0)
    for method in methods:
        rank_temporary = f"rank_{method}_smote_temp.csv"
        rank_final_method = f"rank_{method}_smote.csv"
        rank.rank(data, method, version_dir / rank_temporary)
        contrast_l.contrast_line(
            version_dir / rank_temporary,
            version_dir / rank_final_method,
            version_dir / "bugline.csv",
        )

def cal_bert(program, version, methods):
    print(f"\n****Calculating BERT version for program `{program}` and version `{version}` rank")
    version_dir = Paths.get_d4j_version_dir(program, version)
    testcase_bert.main(program, version)   # bert模型
    data_merge.b_merge(version_dir)
    data = pd.read_csv(version_dir / "matrix_merge_bert.csv", index_col=0)
    for method in methods:
        rank_temporary = f"rank_{method}_bert_temp.csv"
        rank_final_method = f"rank_{method}_bert.csv"
        rank.rank(data, method, version_dir / rank_temporary)
        contrast_l.contrast_line(
            version_dir / rank_temporary,
            version_dir / rank_final_method,
            version_dir / "bugline.csv",
        )

def cal_vae(program, version, methods):
    print(f"\n****Calculating VAE version for program `{program}` and version `{version}` rank")
    version_dir = Paths.get_d4j_version_dir(program, version)
    testcase_vae.main(program, version)   # vae模型
    data_merge.v_merge(version_dir)
    data = pd.read_csv(version_dir / "matrix_merge_vae.csv", index_col=0)
    for method in methods:
        rank_temporary = f"rank_{method}_vae_temp.csv"
        rank_final_method = f"rank_{method}_vae.csv"
        rank.rank(data, method, version_dir / rank_temporary)
        contrast_l.contrast_line(
            version_dir / rank_temporary,
            version_dir / rank_final_method,
            version_dir / "bugline.csv",
        )

def cal_gpt(program, version, methods):
    print(f"\n****Calculating GPT version for program `{program}` and version `{version}` rank")
    version_dir = Paths.get_d4j_version_dir(program, version)
    testcase_gpt.main(program, version)   # gpt模型
    data_merge.gpt_merge(version_dir)
    data = pd.read_csv(version_dir / "matrix_merge_gpt.csv", index_col=0)
    for method in methods:
        rank_temporary = f"rank_{method}_gpt_temp.csv"
        rank_final_method = f"rank_{method}_gpt.csv"
        rank.rank(data, method, version_dir / rank_temporary)
        contrast_l.contrast_line(
            version_dir / rank_temporary,
            version_dir / rank_final_method,
            version_dir / "bugline.csv",
        )

def cal_matrix(program, version, methods):
    print(f"\n****Calculating program `{program}` and version `{version}` rank")
    version_dir = Paths.get_d4j_version_dir(program, version)
    data = pd.read_csv(version_dir / "matrix.csv", index_col=0)
    for method in methods:
        rank_temporary = f"rank_{method}_temp.csv"
        rank_final_method = f"rank_{method}.csv"
        rank.rank(data, method, version_dir / rank_temporary)
        contrast_l.contrast_line(
            version_dir / rank_temporary,
            version_dir / rank_final_method,
            version_dir / "bugline.csv",
        )

def main():
    programs = sys.argv[1:]
    for program in programs:
        print(f"Processing program `{program}`")
        program_data_path = Paths.get_d4j_program_data_dir(program)
        versions = os.listdir(program_data_path)
        versions = natsorted(versions)
        methods = [
            'ochiai',
            'dstar',
            'barinel',
            'MLP',
            'CNN',
            'RNN',
        ]
        for version in versions:
            print(f"\nProcessing buggy-version `{version}` of program `{program}`")
            # undersam(program, version, methods)
            # cal_matrix(program, version, methods)
            # cal_diffusion(program, version, methods)
            # cal_gan(program, version, methods)
            # cal_smote(program, version, methods)
            cal_vae(program, version, methods)
            # cal_gpt(program, version, methods)
            print(f"Done processing buggy-version `{version}` of program `{program}`")
        print(f"Done processing program `{program}`\n")


if __name__ == '__main__':
    main()
