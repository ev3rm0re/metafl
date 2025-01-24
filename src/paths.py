from pathlib import Path


class Paths:
    LabRoot = Path(__file__).parent.parent

    DatasetRoot = LabRoot / "data"

    D4JDatasetRoot = DatasetRoot / "d4j"

    @classmethod
    def get_d4j_buggy_lines(cls, program_name: str, version_name: str):
        return cls.D4JDatasetRoot / "buggy-lines" / f"{program_name}-{version_name}.buggy.lines"

    @classmethod
    def get_d4j_program_data_dir(cls, program_name: str):
        return cls.D4JDatasetRoot / "data" / program_name

    @classmethod
    def get_d4j_program_rank_dir(cls, program_name: str):
        return cls.D4JDatasetRoot / "rank" / program_name

    @classmethod
    def get_d4j_version_dir(cls, program_name: str, version_name: str):
        return cls.D4JDatasetRoot / "data" / program_name / version_name / "gzoltars" / program_name / version_name

    MetaFLDatasetRoot = DatasetRoot / "MetaFL"

    @classmethod
    def get_metalfl_program_data_dir(cls, program_name: str):
        return cls.MetaFLDatasetRoot / program_name
