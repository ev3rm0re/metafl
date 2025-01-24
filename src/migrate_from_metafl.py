from pathlib import Path
from argparse import ArgumentParser
from typing import List, IO
from natsort import natsorted
from paths import Paths
import os
import logging
from itertools import chain

_logger = logging.getLogger(Path(__file__).name)


def write_line(f: IO, line: str):
    f.write(line)
    f.write("\n")


def migrate_components(components: List[int], target: Path):
    """ components  => spectra """
    with open(target, "w") as file:
        file.writelines(map(lambda component: f"MetaFL#{component}\n", components))


def migrate_test_result(components: List[int], cm: Path, fv: Path, target: Path, target_csv: Path):
    """ components, coverageMatrix.txt, failureVector.txt => matrix, matrix.csv """
    csv_header = ",".join(chain(map(str, components), ["error"]))
    with open(cm, "r") as f_cm, \
            open(fv, "r") as f_fv, \
            open(target, "w") as f, \
            open(target_csv, "w") as f_csv:
        write_line(f_csv, csv_header)
        while True:
            coverage = f_cm.readline().strip()
            failure = f_fv.readline().strip()
            if len(failure) == 0:
                break
            write_line(f, f"{coverage} {'+' if failure == '1' else '-'}")
            csv_row = ",".join(chain(coverage.split(" "), [failure]))
            write_line(f_csv, csv_row)


def migrate_one_version_from_metafl(
        faulty_component: int,
        source_version_dir: Path,
        target_version_dir: Path,
        target_buggy_lines_file: Path,
):
    with open(source_version_dir / "componentList.txt", "r") as file:
        components = file.read().strip().split(" ")
    components = list(map(int, components))

    migrate_components(components, target_version_dir / "spectra")

    migrate_test_result(
        components,
        source_version_dir / "coverageMatrix.txt",
        source_version_dir / "failureVector.txt",
        target_version_dir / "matrix",
        target_version_dir / "matrix.csv",
    )

    # faulty_components => .buggy.lines, bugline.csv
    with open(target_buggy_lines_file, "w") as file:
        write_line(file, f"MetaFL#{faulty_component}#MetaFL")
    with open(target_version_dir / "bugline.csv", "w") as file:
        write_line(file, f"{faulty_component}\n")


def migrate_from_metafl(
        program_name: str,
        results_id: str,
):
    """ migrate coverage matrix and failure vector from MetaFL to d4j """
    metafl_program_data_dir = Paths.get_metalfl_program_data_dir(program_name)
    fault_list_file = metafl_program_data_dir / "backup" / f"faultList{results_id}.txt"
    faulty_components = list(map(int, open(fault_list_file).read().strip().split("\n")))
    results_dir = metafl_program_data_dir / "backup" / f"results{results_id}"
    versions = natsorted(os.listdir(results_dir))

    for vid, version in enumerate(versions):
        schemes = natsorted(os.listdir(results_dir / version))
        for scheme in schemes:
            source_dir = results_dir / version / scheme
            assert source_dir.exists()
            target_version = f"{version}-{scheme}"
            target_dir = Paths.get_d4j_version_dir(program_name, target_version)
            target_dir.mkdir(parents=True, exist_ok=True)
            _logger.info(f"migrating version {target_version} from `{source_dir}` to `{target_dir}`")
            target_faults_file = Paths.get_d4j_buggy_lines(program_name, target_version)
            migrate_one_version_from_metafl(
                faulty_components[vid], source_dir,
                target_dir, target_faults_file
            )


def main():
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        level=logging.DEBUG,
    )
    parser = ArgumentParser()
    parser.add_argument("program_name")
    parser.add_argument("results_id")
    args = parser.parse_args()
    migrate_from_metafl(args.program_name, args.results_id)


if __name__ == '__main__':
    main()
